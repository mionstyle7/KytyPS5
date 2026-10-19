#include "graphics/host_gpu/renderer/pipeline/pipelineWarmup.h"

#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/cpuAffinity.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <type_traits>
#include <xxhash.h>

namespace Libs::Graphics::PipelineWarmup {

namespace {

constexpr uint32_t FileMagic     = 0x4c50574bu; // "KWPL"
constexpr uint32_t FormatVersion = 1;
// Recipes no session used for this many sessions are dropped when the file is opened.
constexpr uint32_t KeepSessions = 4;

static_assert(std::is_trivially_copyable_v<vk::VertexInputBindingDescription>);
static_assert(std::is_trivially_copyable_v<vk::VertexInputAttributeDescription>);
static_assert(std::is_trivially_copyable_v<vk::PipelineColorBlendAttachmentState>);
static_assert(std::is_trivially_copyable_v<vk::DynamicState>);
static_assert(std::is_trivially_copyable_v<vk::Format>);

struct Writer {
	std::vector<uint8_t> bytes;
	void                 Raw(const void* data, size_t size) {
        const auto* begin = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), begin, begin + size);
	}
	template <typename T>
	void Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Raw(&value, sizeof(T));
	}
	void U32(uint32_t value) { Put(value); }
	void U8(bool value) { Put<uint8_t>(value ? 1 : 0); }
};

struct Reader {
	std::span<const uint8_t> data;
	size_t                   offset = 0;
	bool                     ok     = true;
	bool                     Raw(void* out, size_t size) {
        if (!ok || size > data.size() - offset) {
            ok = false;
            return false;
        }
        std::memcpy(out, data.data() + offset, size);
        offset += size;
        return true;
	}
	template <typename T>
	T Get() {
		T value {};
		Raw(&value, sizeof(T));
		return value;
	}
	uint32_t U32() { return Get<uint32_t>(); }
	bool     U8() { return Get<uint8_t>() != 0; }
	std::span<const uint8_t> Bytes(size_t size) {
		if (!ok || size > data.size() - offset) {
			ok = false;
			return {};
		}
		const auto result = data.subspan(offset, size);
		offset += size;
		return result;
	}
};

// Module registry: the SPIR-V of every shader module the program cache created.
struct Registry {
	std::mutex                                                              mutex;
	std::unordered_map<VkShaderModule, std::shared_ptr<const Store::Module>> modules;
};
Registry& GetRegistry() {
	static Registry registry;
	return registry;
}

std::shared_ptr<const Store::Module> MakeModule(std::span<const uint32_t> words) {
	auto module   = std::make_shared<Store::Module>();
	module->words.assign(words.begin(), words.end());
	module->hash = XXH3_64bits(words.data(), words.size_bytes());
	return module;
}

std::shared_ptr<const Store::Module> FindModule(vk::ShaderModule module,
                                                const GraphicsPipelineBuild* build) {
	if (module == nullptr) {
		return nullptr;
	}
	if (build != nullptr) {
		if (module == build->tess_control && !build->tess_control_spirv.empty()) {
			return MakeModule(build->tess_control_spirv);
		}
		if (module == build->tess_eval && !build->tess_eval_spirv.empty()) {
			return MakeModule(build->tess_eval_spirv);
		}
	}
	auto&           registry = GetRegistry();
	std::lock_guard lock(registry.mutex);
	const auto      found = registry.modules.find(static_cast<VkShaderModule>(module));
	return found != registry.modules.end() ? found->second : nullptr;
}

void PutLayout(Writer& w, const PipelineCache::Pipeline& pipeline) {
	Writer layout;
	layout.U32(static_cast<uint32_t>(pipeline.layout_bindings.size()));
	for (const auto& binding: pipeline.layout_bindings) {
		layout.U32(binding.binding);
		layout.U32(static_cast<uint32_t>(binding.descriptorType));
		layout.U32(binding.descriptorCount);
		layout.U32(static_cast<uint32_t>(binding.stageFlags));
	}
	layout.U8(pipeline.uses_push_descriptors);
	layout.U8(pipeline.uses_bindless);
	layout.U32(static_cast<uint32_t>(pipeline.push_constant_stages));
	layout.U32(ShaderRecompiler::IR::NativePushConstantSize);
	w.U32(static_cast<uint32_t>(layout.bytes.size()));
	w.Raw(layout.bytes.data(), layout.bytes.size());
}

// The create info of a graphics build, with modules named by their SPIR-V hash. False when a
// stage's SPIR-V is unknown.
bool SerializeGraphics(const GraphicsPipelineBuild& b, const PipelineCache::Pipeline& pipeline,
                       std::vector<uint8_t>& out,
                       std::vector<std::shared_ptr<const Store::Module>>& modules) {
	const auto& info = b.pipeline_info;
	Writer      w;
	w.Put<char>('G');
	PutLayout(w, pipeline);
	w.U32(static_cast<uint32_t>(info.flags));
	w.U32(info.stageCount);
	for (uint32_t i = 0; i < info.stageCount; i++) {
		const auto& stage  = b.stages[i];
		auto        module = FindModule(stage.module, &b);
		if (module == nullptr) {
			return false;
		}
		w.U32(static_cast<uint32_t>(stage.stage));
		w.Put(module->hash);
		w.U32(stage.pNext == &b.subgroup_sizes[i] ? b.subgroup_sizes[i].requiredSubgroupSize : 0u);
		modules.push_back(std::move(module));
	}
	w.U8(info.pVertexInputState != nullptr);
	if (info.pVertexInputState != nullptr) {
		const auto& vi = b.vertex_input_info;
		w.U32(vi.vertexBindingDescriptionCount);
		w.Raw(b.input_desc.data(), vi.vertexBindingDescriptionCount * sizeof(b.input_desc[0]));
		w.U32(vi.vertexAttributeDescriptionCount);
		w.Raw(b.input_attr.data(), vi.vertexAttributeDescriptionCount * sizeof(b.input_attr[0]));
	}
	w.U8(info.pInputAssemblyState != nullptr);
	w.U32(static_cast<uint32_t>(b.input_assembly.topology));
	w.U32(b.input_assembly.primitiveRestartEnable);
	w.U32(b.depth_clip_control.negativeOneToOne);
	const auto& rs             = b.rasterizer;
	const bool  has_provoking  = rs.pNext == &b.provoking_vertex;
	const void* after_provoking = has_provoking ? b.provoking_vertex.pNext : rs.pNext;
	w.U32(rs.depthClampEnable);
	w.U8(after_provoking == &b.clip_ext);
	w.U32(b.clip_ext.depthClipEnable);
	w.U8(has_provoking);
	w.U32(static_cast<uint32_t>(b.provoking_vertex.provokingVertexMode));
	w.U32(static_cast<uint32_t>(rs.cullMode));
	w.U32(static_cast<uint32_t>(rs.frontFace));
	w.U32(static_cast<uint32_t>(rs.polygonMode));
	w.Put(rs.lineWidth);
	w.U32(b.multisampling.sampleShadingEnable);
	w.U32(static_cast<uint32_t>(b.multisampling.rasterizationSamples));
	w.Put(b.multisampling.minSampleShading);
	const auto& cb = b.color_blending;
	w.U32(cb.attachmentCount);
	w.Raw(b.color_blend.data(), cb.attachmentCount * sizeof(b.color_blend[0]));
	w.U32(static_cast<uint32_t>(cb.logicOp));
	w.U8(cb.pNext == &b.color_write);
	w.U32(b.color_write.attachmentCount);
	w.Raw(b.color_write_enable.data(), b.color_write.attachmentCount * sizeof(vk::Bool32));
	w.U8(info.pDepthStencilState != nullptr);
	w.U32(b.depth_stencil.depthBoundsTestEnable);
	w.Put(b.depth_stencil.minDepthBounds);
	w.Put(b.depth_stencil.maxDepthBounds);
	w.U32(b.dynamic_state.dynamicStateCount);
	w.Raw(b.dynamic_states.data(), b.dynamic_states.size() * sizeof(vk::DynamicState));
	w.U8(info.pTessellationState != nullptr);
	w.U32(b.tessellation.patchControlPoints);
	w.U32(b.rendering.color_count);
	w.Raw(b.rendering.color_formats.data(), b.rendering.color_count * sizeof(vk::Format));
	w.Put(b.rendering.depth_format);
	w.Put(b.rendering.stencil_format);
	out = std::move(w.bytes);
	return true;
}

// Fills `b` from SerializeGraphics' bytes, wired as PrepareGraphicsPipeline wires it. The
// modules come from `module_for`; the layout is set by the caller.
template <typename ModuleFor>
bool DeserializeGraphics(std::span<const uint8_t> bytes, GraphicsPipelineBuild& b,
                         std::span<const uint8_t>& layout_bytes, ModuleFor&& module_for) {
	Reader r {bytes};
	if (r.Get<char>() != 'G') {
		return false;
	}
	layout_bytes = r.Bytes(r.U32());
	auto& info   = b.pipeline_info;
	info.flags   = vk::PipelineCreateFlags {r.U32()};
	const auto stage_count = r.U32();
	if (!r.ok || stage_count > b.stages.size()) {
		return false;
	}
	for (uint32_t i = 0; i < stage_count; i++) {
		auto& stage    = b.stages[i];
		stage.stage    = static_cast<vk::ShaderStageFlagBits>(r.U32());
		const auto hash = r.Get<uint64_t>();
		stage.module   = module_for(hash);
		stage.pName    = "main";
		if (const auto size = r.U32(); size != 0) {
			b.subgroup_sizes[i].requiredSubgroupSize = size;
			stage.pNext                              = &b.subgroup_sizes[i];
		}
		if (stage.module == nullptr) {
			return false;
		}
	}
	const bool has_vertex_input = r.U8();
	if (has_vertex_input) {
		const auto bindings = r.U32();
		if (bindings > b.input_desc.size()) {
			return false;
		}
		r.Raw(b.input_desc.data(), bindings * sizeof(b.input_desc[0]));
		const auto attributes = r.U32();
		if (attributes > b.input_attr.size()) {
			return false;
		}
		r.Raw(b.input_attr.data(), attributes * sizeof(b.input_attr[0]));
		b.vertex_input_info.vertexBindingDescriptionCount   = bindings;
		b.vertex_input_info.pVertexBindingDescriptions      = b.input_desc.data();
		b.vertex_input_info.vertexAttributeDescriptionCount = attributes;
		b.vertex_input_info.pVertexAttributeDescriptions    = b.input_attr.data();
	}
	const bool has_input_assembly                = r.U8();
	b.input_assembly.topology                    = static_cast<vk::PrimitiveTopology>(r.U32());
	b.input_assembly.primitiveRestartEnable      = r.U32();
	b.depth_clip_control.negativeOneToOne        = r.U32();
	b.viewport_state.pNext                       = &b.depth_clip_control;
	auto& rs                                     = b.rasterizer;
	rs.depthClampEnable                          = r.U32();
	const bool has_clip                          = r.U8();
	b.clip_ext.depthClipEnable                   = r.U32();
	const bool has_provoking                     = r.U8();
	b.provoking_vertex.provokingVertexMode       = static_cast<vk::ProvokingVertexModeEXT>(r.U32());
	if (has_clip) {
		rs.pNext = &b.clip_ext;
	}
	if (has_provoking) {
		b.provoking_vertex.pNext = rs.pNext;
		rs.pNext                 = &b.provoking_vertex;
	}
	rs.cullMode                              = vk::CullModeFlags {r.U32()};
	rs.frontFace                             = static_cast<vk::FrontFace>(r.U32());
	rs.polygonMode                           = static_cast<vk::PolygonMode>(r.U32());
	rs.lineWidth                             = r.Get<float>();
	b.multisampling.sampleShadingEnable      = r.U32();
	b.multisampling.rasterizationSamples     = static_cast<vk::SampleCountFlagBits>(r.U32());
	b.multisampling.minSampleShading         = r.Get<float>();
	auto&      cb                            = b.color_blending;
	const auto attachments                   = r.U32();
	if (attachments > b.color_blend.size()) {
		return false;
	}
	r.Raw(b.color_blend.data(), attachments * sizeof(b.color_blend[0]));
	cb.attachmentCount = attachments;
	cb.pAttachments    = b.color_blend.data();
	cb.logicOp         = static_cast<vk::LogicOp>(r.U32());
	if (r.U8()) {
		cb.pNext = &b.color_write;
	}
	const auto write_count = r.U32();
	if (write_count > b.color_write_enable.size()) {
		return false;
	}
	r.Raw(b.color_write_enable.data(), write_count * sizeof(vk::Bool32));
	b.color_write.attachmentCount    = write_count;
	b.color_write.pColorWriteEnables = b.color_write_enable.data();
	const bool has_depth_stencil              = r.U8();
	b.depth_stencil.depthBoundsTestEnable     = r.U32();
	b.depth_stencil.minDepthBounds            = r.Get<float>();
	b.depth_stencil.maxDepthBounds            = r.Get<float>();
	const auto dynamic_count                  = r.U32();
	if (!r.ok || dynamic_count > 64) {
		return false;
	}
	b.dynamic_states.resize(dynamic_count);
	r.Raw(b.dynamic_states.data(), dynamic_count * sizeof(vk::DynamicState));
	b.dynamic_state.dynamicStateCount = dynamic_count;
	b.dynamic_state.pDynamicStates    = b.dynamic_states.data();
	const bool has_tessellation       = r.U8();
	b.tessellation.patchControlPoints = r.U32();
	b.rendering.color_count           = r.U32();
	if (b.rendering.color_count > b.rendering.color_formats.size()) {
		return false;
	}
	r.Raw(b.rendering.color_formats.data(), b.rendering.color_count * sizeof(vk::Format));
	b.rendering.depth_format   = r.Get<vk::Format>();
	b.rendering.stencil_format = r.Get<vk::Format>();
	if (!r.ok || r.offset != bytes.size()) {
		return false;
	}
	b.rendering_info.colorAttachmentCount    = b.rendering.color_count;
	b.rendering_info.pColorAttachmentFormats = b.rendering.color_formats.data();
	b.rendering_info.depthAttachmentFormat   = b.rendering.depth_format;
	b.rendering_info.stencilAttachmentFormat = b.rendering.stencil_format;
	info.pNext                               = &b.rendering_info;
	info.stageCount                          = stage_count;
	info.pStages                             = b.stages.data();
	info.pVertexInputState                   = has_vertex_input ? &b.vertex_input_info : nullptr;
	info.pInputAssemblyState                 = has_input_assembly ? &b.input_assembly : nullptr;
	info.pTessellationState                  = has_tessellation ? &b.tessellation : nullptr;
	info.pViewportState                      = &b.viewport_state;
	info.pRasterizationState                 = &b.rasterizer;
	info.pMultisampleState                   = &b.multisampling;
	info.pDepthStencilState                  = has_depth_stencil ? &b.depth_stencil : nullptr;
	info.pColorBlendState                    = &b.color_blending;
	info.pDynamicState                       = &b.dynamic_state;
	info.basePipelineIndex                   = -1;
	return true;
}

std::vector<uint8_t> SerializeCompute(const PipelineCache::Pipeline& pipeline, uint64_t module_hash,
                                      uint32_t wave_size, uint64_t shader_hash) {
	Writer w;
	w.Put<char>('C');
	PutLayout(w, pipeline);
	w.Put(module_hash);
	w.U32(wave_size);
	w.Put(shader_hash);
	return std::move(w.bytes);
}

// The modules a recipe names (their hashes), in order.
std::vector<uint64_t> RecipeModules(std::span<const uint8_t> bytes) {
	std::vector<uint64_t> result;
	Reader                r {bytes};
	const auto            kind = r.Get<char>();
	(void)r.Bytes(r.U32());
	if (kind == 'C') {
		result.push_back(r.Get<uint64_t>());
	} else if (kind == 'G') {
		(void)r.U32(); // flags
		const auto count = r.U32();
		for (uint32_t i = 0; r.ok && i < count && i < 8; i++) {
			(void)r.U32();
			result.push_back(r.Get<uint64_t>());
			(void)r.U32();
		}
	}
	if (!r.ok) {
		result.clear();
	}
	return result;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PIPELINE_WARMUP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void RegisterModule(vk::ShaderModule module, std::span<const uint32_t> spirv) {
	if (!Enabled() || module == nullptr) {
		return;
	}
	auto            entry    = MakeModule(spirv);
	auto&           registry = GetRegistry();
	std::lock_guard lock(registry.mutex);
	registry.modules[static_cast<VkShaderModule>(module)] = std::move(entry);
}

// File: a header {magic, version, session}, then records {type, hash, size, bytes}:
//   'S' a session starts (hash = its number): the records after it were seen in it;
//   'M' a module's SPIR-V (hash = its XXH3);
//   'G'/'C' a graphics or compute recipe (hash = XXH3 of the bytes);
//   'U' a recipe used again (no bytes).
struct FileHeader {
	uint32_t magic   = FileMagic;
	uint32_t version = FormatVersion;
	uint32_t session = 0;
};

Store::Store(GraphicContext& graphics, vk::PipelineCache driver_cache, std::filesystem::path path)
    : m_graphics(graphics), m_driver_cache(driver_cache), m_path(std::move(path)) {
	Load();
}

void Store::Load() {
	std::error_code error;
	std::filesystem::create_directories(m_path.parent_path(), error);
	const auto path_text = Common::PathToString(m_path);
	uint32_t   last      = 0;
	if (std::FILE* in = std::fopen(path_text.c_str(), "rb"); in != nullptr) {
		FileHeader header {};
		if (std::fread(&header, sizeof(header), 1, in) == 1 && header.magic == FileMagic &&
		    header.version == FormatVersion) {
			last            = header.session;
			uint32_t current = 0;
			for (;;) {
				char     type = 0;
				uint64_t hash = 0;
				uint32_t size = 0;
				if (std::fread(&type, 1, 1, in) != 1 || std::fread(&hash, sizeof(hash), 1, in) != 1 ||
				    std::fread(&size, sizeof(size), 1, in) != 1 || size > (256u << 20u)) {
					break;
				}
				std::vector<uint8_t> bytes(size);
				if (size != 0 && std::fread(bytes.data(), size, 1, in) != 1) {
					break; // cut short when a session ended mid-write
				}
				switch (type) {
					case 'S':
						current = static_cast<uint32_t>(hash);
						last    = std::max(last, current);
						break;
					case 'M': {
						std::vector<uint32_t> words(size / 4);
						std::memcpy(words.data(), bytes.data(), words.size() * 4);
						m_module_words[hash] = std::move(words);
						break;
					}
					case 'G':
					case 'C': {
						auto& recipe        = m_recipes[hash];
						recipe.modules      = RecipeModules(bytes);
						recipe.bytes        = std::move(bytes);
						recipe.last_session = std::max(recipe.last_session, current);
						break;
					}
					case 'U':
						if (auto found = m_recipes.find(hash); found != m_recipes.end()) {
							found->second.last_session = std::max(found->second.last_session, current);
						}
						break;
					default: break;
				}
			}
		}
		std::fclose(in);
	}
	m_session = last + 1;
	// Drops recipes unused for KeepSessions sessions and the modules no recipe names, and any
	// recipe whose modules are missing.
	std::unordered_map<uint64_t, bool> needed;
	for (auto it = m_recipes.begin(); it != m_recipes.end();) {
		auto& recipe  = it->second;
		bool  missing = recipe.bytes.empty() || recipe.modules.empty();
		for (const auto hash: recipe.modules) {
			missing = missing || !m_module_words.contains(hash);
		}
		if (missing || recipe.last_session + KeepSessions < m_session) {
			it = m_recipes.erase(it);
			continue;
		}
		for (const auto hash: recipe.modules) {
			needed[hash] = true;
		}
		++it;
	}
	std::erase_if(m_module_words, [&](const auto& entry) { return !needed.contains(entry.first); });
	// Written again compacted: the modules, then the recipes grouped by the session that last
	// used them.
	m_file = std::fopen(path_text.c_str(), "wb");
	if (m_file == nullptr) {
		LOGF("Pipeline warmup: cannot write %s\n", path_text.c_str());
		return;
	}
	const FileHeader header {.session = m_session};
	std::fwrite(&header, sizeof(header), 1, m_file);
	for (const auto& [hash, words]: m_module_words) {
		WriteRecord('M', hash,
		            {reinterpret_cast<const uint8_t*>(words.data()), words.size() * sizeof(uint32_t)});
	}
	std::vector<std::pair<uint32_t, uint64_t>> order;
	order.reserve(m_recipes.size());
	for (const auto& [hash, recipe]: m_recipes) {
		order.emplace_back(recipe.last_session, hash);
	}
	std::sort(order.begin(), order.end());
	uint32_t session = 0;
	for (const auto& [last_session, hash]: order) {
		if (last_session != session) {
			session = last_session;
			WriteRecord('S', session, {});
		}
		auto& recipe = m_recipes[hash];
		WriteRecord(recipe.bytes.front() == 'C' ? 'C' : 'G', hash, recipe.bytes);
		recipe.state = State::Queued;
		m_queue.push_back(hash);
	}
	WriteRecord('S', m_session, {});
	std::fflush(m_file);
	// The most recently used first: they are the likeliest to be drawn soon.
	std::reverse(m_queue.begin(), m_queue.end());
	LOGF("Pipeline warmup: %zu recorded pipelines (%zu shader modules) in %s, session %u\n",
	     m_recipes.size(), m_module_words.size(), path_text.c_str(), m_session);
}

Store::~Store() {
	m_stop.store(true);
	{
		std::lock_guard lock(m_mutex);
		m_changed.notify_all();
	}
	for (auto& thread: m_threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
	const auto device = m_graphics.device;
	for (auto& [hash, recipe]: m_recipes) {
		(void)hash;
		if (recipe.state == State::Built && recipe.pipeline != nullptr) {
			device.destroyPipeline(recipe.pipeline, nullptr);
		}
	}
	for (const auto& [hash, module]: m_modules) {
		(void)hash;
		device.destroyShaderModule(module, nullptr);
	}
	for (const auto& [hash, objects]: m_layouts) {
		(void)hash;
		device.destroyPipelineLayout(objects.layout, nullptr);
		device.destroyDescriptorSetLayout(objects.set, nullptr);
	}
	if (m_file != nullptr) {
		std::fclose(m_file);
	}
}

void Store::WriteRecord(char type, uint64_t hash, std::span<const uint8_t> bytes) {
	if (m_file == nullptr) {
		return;
	}
	const uint32_t size = static_cast<uint32_t>(bytes.size());
	std::fwrite(&type, 1, 1, m_file);
	std::fwrite(&hash, sizeof(hash), 1, m_file);
	std::fwrite(&size, sizeof(size), 1, m_file);
	if (size != 0) {
		std::fwrite(bytes.data(), size, 1, m_file);
	}
}

void Store::WriteModule(const Module& module) {
	if (m_module_words.contains(module.hash)) {
		return;
	}
	m_module_words[module.hash] = module.words;
	WriteRecord('M', module.hash,
	            {reinterpret_cast<const uint8_t*>(module.words.data()),
	             module.words.size() * sizeof(uint32_t)});
}

void Store::Start() {
	std::lock_guard lock(m_mutex);
	if (m_started) {
		return;
	}
	m_started = true;
	m_begin   = std::chrono::steady_clock::now();
	if (m_queue.empty()) {
		return;
	}
	const char* value    = std::getenv("KYTY_PIPELINE_WARMUP_THREADS");
	const auto  hardware = std::max(std::thread::hardware_concurrency(), 2u);
	const auto  threads  = std::max<uint32_t>(
        value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
                           : (hardware > 6u ? hardware / 2u - 1u : 1u),
        1u);
	LOGF("Pipeline warmup: building %zu pipelines on %u threads\n", m_queue.size(), threads);
	for (uint32_t i = 0; i < threads; i++) {
		m_threads.emplace_back([this] { Worker(); });
	}
	// KYTY_PIPELINE_WARMUP_KEEP_S (default 120; 0 keeps them all): prebuilt pipelines no draw took
	// within this many seconds are destroyed. Building them already filled the driver's cache, so
	// one needed later is created from it in about a millisecond; kept, ~1,300 pipelines of a
	// session held video memory the late gameplay needed (3-5 fps against ~14 without them,
	// 2026-10-19).
	static const uint64_t keep_s = [] {
		const char* keep = std::getenv("KYTY_PIPELINE_WARMUP_KEEP_S");
		return keep != nullptr ? std::strtoull(keep, nullptr, 10) : uint64_t {120};
	}();
	if (keep_s != 0) {
		m_threads.emplace_back([this] {
			std::unique_lock lock(m_mutex);
			const auto deadline = m_begin + std::chrono::seconds(keep_s);
			m_changed.wait_until(lock, deadline, [this] { return m_stop.load(); });
			if (m_stop.load()) {
				return;
			}
			// Nothing more is started; builds still running finish first.
			for (auto& [hash, recipe]: m_recipes) {
				(void)hash;
				if (recipe.state == State::Queued) {
					recipe.state = State::Taken;
				}
			}
			m_changed.wait(lock, [this] { return m_stop.load() || m_active == 0; });
			if (m_stop.load()) {
				return;
			}
			uint64_t dropped = 0;
			for (auto& [hash, recipe]: m_recipes) {
				(void)hash;
				if (recipe.state == State::Built && recipe.pipeline != nullptr) {
					m_graphics.device.destroyPipeline(recipe.pipeline, nullptr);
					recipe.pipeline = nullptr;
					recipe.state    = State::Taken;
					dropped++;
				}
			}
			LOGF("Pipeline warmup: %" PRIu64 " prebuilt pipelines unused after %" PRIu64
			     " s destroyed (%" PRIu64 " were taken by draws)\n",
			     dropped, keep_s, m_taken);
		});
	}
}

void Store::Worker() {
	Common::LowerBelowGameThreads();
	for (;;) {
		uint64_t hash = 0;
		Recipe   copy;
		{
			std::lock_guard lock(m_mutex);
			while (m_next < m_queue.size() &&
			       m_recipes[m_queue[m_next]].state != State::Queued) {
				m_next++;
			}
			if (m_stop.load() || m_next >= m_queue.size()) {
				break;
			}
			hash          = m_queue[m_next++];
			auto& recipe  = m_recipes[hash];
			recipe.state  = State::Building;
			copy.bytes    = recipe.bytes;
			copy.modules  = recipe.modules;
			m_active++;
		}
		const auto pipeline = Build(hash, copy);
		std::lock_guard lock(m_mutex);
		m_active--;
		auto& recipe = m_recipes[hash];
		if (pipeline != nullptr) {
			recipe.state    = State::Built;
			recipe.pipeline = pipeline;
			m_built++;
		} else {
			recipe.state = State::Failed;
			m_failed++;
		}
		m_changed.notify_all();
		if (m_active == 0 && m_next >= m_queue.size()) {
			LogProgress();
			// Every build is done: the modules are no longer needed.
			for (const auto& [module_hash, module]: m_modules) {
				(void)module_hash;
				m_graphics.device.destroyShaderModule(module, nullptr);
			}
			m_modules.clear();
		}
	}
}

void Store::LogProgress() {
	const auto seconds =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - m_begin).count();
	LOGF("Pipeline warmup: %" PRIu64 " pipelines built, %" PRIu64 " failed, in %.1f s; %" PRIu64
	     " taken by draws so far (%" PRIu64 " waited for their build), %" PRIu64
	     " new pipelines recorded\n",
	     m_built, m_failed, seconds, m_taken, m_waited, m_recorded);
}

// Worker threads, m_mutex not held: the modules and layouts are created under it.
vk::ShaderModule Store::ModuleFor(uint64_t hash) {
	std::lock_guard lock(m_mutex);
	if (auto found = m_modules.find(hash); found != m_modules.end()) {
		return found->second;
	}
	const auto words = m_module_words.find(hash);
	if (words == m_module_words.end() || words->second.empty()) {
		return nullptr;
	}
	const auto module = CompileSPV(words->second, m_graphics.device);
	if (module != nullptr) {
		m_modules[hash] = module;
	}
	return module;
}

vk::PipelineLayout Store::LayoutFor(std::span<const uint8_t> layout_bytes, bool& ok) {
	ok               = false;
	const auto hash  = XXH3_64bits(layout_bytes.data(), layout_bytes.size());
	std::lock_guard lock(m_mutex);
	if (auto found = m_layouts.find(hash); found != m_layouts.end()) {
		ok = true;
		return found->second.layout;
	}
	Reader                                      r {layout_bytes};
	std::vector<vk::DescriptorSetLayoutBinding> bindings(r.U32());
	if (!r.ok || bindings.size() > 4096) {
		return nullptr;
	}
	for (auto& binding: bindings) {
		binding.binding         = r.U32();
		binding.descriptorType  = static_cast<vk::DescriptorType>(r.U32());
		binding.descriptorCount = r.U32();
		binding.stageFlags      = vk::ShaderStageFlags {r.U32()};
	}
	const bool push_descriptors = r.U8();
	const bool bindless         = r.U8();
	const auto push_stages      = vk::ShaderStageFlags {r.U32()};
	const auto push_size        = r.U32();
	if (!r.ok || (bindless && m_graphics.bindless_layout == nullptr)) {
		return nullptr;
	}
	LayoutObjects                     objects;
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = push_descriptors ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                                         : vk::DescriptorSetLayoutCreateFlags {};
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (m_graphics.device.createDescriptorSetLayout(&set_info, nullptr, &objects.set) !=
	    vk::Result::eSuccess) {
		return nullptr;
	}
	std::array<vk::DescriptorSetLayout, 2> sets {objects.set, m_graphics.bindless_layout};
	const vk::PushConstantRange            push {push_stages, 0, push_size};
	vk::PipelineLayoutCreateInfo           layout_info {};
	layout_info.setLayoutCount         = bindless ? 2u : 1u;
	layout_info.pSetLayouts            = sets.data();
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (m_graphics.device.createPipelineLayout(&layout_info, nullptr, &objects.layout) !=
	    vk::Result::eSuccess) {
		m_graphics.device.destroyDescriptorSetLayout(objects.set, nullptr);
		return nullptr;
	}
	m_layouts[hash] = objects;
	ok              = true;
	return objects.layout;
}

vk::Pipeline Store::Build(uint64_t hash, const Recipe& recipe) {
	(void)hash;
	if (recipe.bytes.empty()) {
		return nullptr;
	}
	vk::Pipeline pipeline = nullptr;
	if (recipe.bytes.front() == 'C') {
		Reader r {recipe.bytes};
		(void)r.Get<char>();
		const auto layout_bytes = r.Bytes(r.U32());
		const auto module_hash  = r.Get<uint64_t>();
		const auto wave_size    = r.U32();
		const auto shader_hash  = r.Get<uint64_t>();
		if (!r.ok) {
			return nullptr;
		}
		bool       ok     = false;
		const auto layout = LayoutFor(layout_bytes, ok);
		const auto module = ModuleFor(module_hash);
		if (!ok || module == nullptr) {
			return nullptr;
		}
		if (CreateComputePipelineHandle(m_graphics, layout, wave_size, shader_hash, module,
		                                m_driver_cache, {}, &pipeline) != vk::Result::eSuccess) {
			return nullptr;
		}
		return pipeline;
	}
	GraphicsPipelineBuild    build;
	build.device = m_graphics.device;
	std::span<const uint8_t> layout_bytes;
	if (!DeserializeGraphics(recipe.bytes, build, layout_bytes,
	                         [this](uint64_t module_hash) { return ModuleFor(module_hash); })) {
		return nullptr;
	}
	bool ok                    = false;
	build.pipeline_info.layout = LayoutFor(layout_bytes, ok);
	if (!ok) {
		return nullptr;
	}
	if (CreateGraphicsPipeline(build, m_driver_cache, &pipeline) != vk::Result::eSuccess) {
		return nullptr;
	}
	return pipeline;
}

vk::Pipeline Store::Take(uint64_t hash, std::vector<uint8_t>&& bytes,
                         std::vector<std::shared_ptr<const Module>>&& modules) {
	std::unique_lock lock(m_mutex);
	auto             found = m_recipes.find(hash);
	if (found == m_recipes.end()) {
		// New: recorded (its modules first) for the next session.
		for (const auto& module: modules) {
			WriteModule(*module);
		}
		WriteRecord(bytes.front(), hash, bytes);
		std::fflush(m_file);
		auto& recipe        = m_recipes[hash];
		recipe.modules      = RecipeModules(bytes);
		recipe.bytes        = std::move(bytes);
		recipe.last_session = m_session;
		recipe.used_logged  = true;
		recipe.state        = State::Taken;
		m_recorded++;
		return nullptr;
	}
	auto& recipe = found->second;
	if (!recipe.used_logged) {
		recipe.used_logged = true;
		WriteRecord('U', hash, {});
		std::fflush(m_file);
	}
	if (recipe.state == State::Building) {
		// Its build is under way: waiting for it costs less than starting another.
		m_waited++;
		m_changed.wait(lock, [&] { return recipe.state != State::Building || m_stop.load(); });
	}
	vk::Pipeline result = nullptr;
	if (recipe.state == State::Built) {
		result          = recipe.pipeline;
		recipe.pipeline = nullptr;
		m_taken++;
	}
	recipe.state = State::Taken;
	return result;
}

vk::Pipeline Store::TakeGraphics(const GraphicsPipelineBuild& build,
                                 const PipelineCache::Pipeline& pipeline) {
	std::vector<uint8_t>                       bytes;
	std::vector<std::shared_ptr<const Module>> modules;
	if (!SerializeGraphics(build, pipeline, bytes, modules)) {
		return nullptr;
	}
	const auto hash = XXH3_64bits(bytes.data(), bytes.size());
	return Take(hash, std::move(bytes), std::move(modules));
}

vk::Pipeline Store::TakeCompute(const PipelineCache::Pipeline& pipeline, vk::ShaderModule module,
                                uint32_t wave_size, uint64_t shader_hash) {
	auto found = FindModule(module, nullptr);
	if (found == nullptr) {
		return nullptr;
	}
	auto       bytes = SerializeCompute(pipeline, found->hash, wave_size, shader_hash);
	const auto hash  = XXH3_64bits(bytes.data(), bytes.size());
	std::vector<std::shared_ptr<const Module>> modules {std::move(found)};
	return Take(hash, std::move(bytes), std::move(modules));
}

} // namespace Libs::Graphics::PipelineWarmup
