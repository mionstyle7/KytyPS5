#include "graphics/host_gpu/renderer/pipeline/shaderWarmup.h"

#include "common/logging/log.h"
#include "common/stringUtils.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics::ShaderWarmup {

namespace {

static_assert(std::is_trivially_copyable_v<ShaderVertexInputInfo>);
static_assert(std::is_trivially_copyable_v<ShaderPixelInputInfo>);
static_assert(std::is_trivially_copyable_v<ShaderComputeInputInfo>);
static_assert(std::is_trivially_copyable_v<ShaderRecompiler::IR::ResourceSpecialization::Buffer>);
static_assert(std::is_trivially_copyable_v<ShaderRecompiler::IR::ResourceSpecialization::Image>);
static_assert(std::is_trivially_copyable_v<ShaderRecompiler::IR::ResourceSpecialization::Sampler>);

constexpr uint32_t FileMagic     = 0x4d52574bu; // "KWRM"
constexpr uint32_t FormatVersion = 1;

// The layouts the records were written with: another build whose structures differ starts a new
// file (the records are plain copies of them).
struct FileHeader {
	uint32_t magic         = FileMagic;
	uint32_t version       = FormatVersion;
	uint32_t vertex_size   = sizeof(ShaderVertexInputInfo);
	uint32_t pixel_size    = sizeof(ShaderPixelInputInfo);
	uint32_t compute_size  = sizeof(ShaderComputeInputInfo);
	uint32_t buffer_size   = sizeof(ShaderRecompiler::IR::ResourceSpecialization::Buffer);
	uint32_t image_size    = sizeof(ShaderRecompiler::IR::ResourceSpecialization::Image);
	uint32_t sampler_size  = sizeof(ShaderRecompiler::IR::ResourceSpecialization::Sampler);
	bool     operator==(const FileHeader&) const = default;
};

struct Writer {
	std::vector<uint8_t> bytes;
	void Raw(const void* data, size_t size) {
		const auto* begin = static_cast<const uint8_t*>(data);
		bytes.insert(bytes.end(), begin, begin + size);
	}
	template <typename T>
	void Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Raw(&value, sizeof(T));
	}
	template <typename T>
	void PutVector(const std::vector<T>& values) {
		Put<uint32_t>(static_cast<uint32_t>(values.size()));
		if (!values.empty()) {
			Raw(values.data(), values.size() * sizeof(T));
		}
	}
};

struct Reader {
	std::span<const uint8_t> data;
	size_t                   offset = 0;
	bool                     ok     = true;
	bool Raw(void* out, size_t size) {
		if (!ok || size > data.size() - offset) {
			ok = false;
			return false;
		}
		std::memcpy(out, data.data() + offset, size);
		offset += size;
		return true;
	}
	template <typename T>
	bool Get(T& value) {
		return Raw(&value, sizeof(T));
	}
	template <typename T>
	bool GetVector(std::vector<T>& values) {
		uint32_t count = 0;
		if (!Get(count) || count > (data.size() - offset) / sizeof(T)) {
			ok = false;
			return false;
		}
		values.resize(count);
		return count == 0 || Raw(values.data(), count * sizeof(T));
	}
};

char InputKind(ShaderType stage) {
	switch (stage) {
		case ShaderType::Pixel: return 'P';
		case ShaderType::Compute: return 'C';
		default: return 'V';
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_WARMUP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

std::vector<uint8_t> Encode(const Record& record) {
	Writer w;
	w.Put(static_cast<uint32_t>(record.stage));
	w.Put(record.hash);
	w.Put(record.wave_size);
	w.Put(record.user_data_base);
	w.Put(record.push_cursor);
	w.Put<uint8_t>(record.bindless_images ? 1 : 0);
	w.Put<uint8_t>(record.non_fatal ? 1 : 0);
	w.PutVector(record.code);
	w.PutVector(record.back_code);
	w.PutVector(record.user_data);
	switch (InputKind(record.stage)) {
		case 'P': {
			auto copy  = record.pixel;
			copy.stage = {};
			w.Put(copy);
			break;
		}
		case 'C': {
			auto copy  = record.compute;
			copy.stage = {};
			w.Put(copy);
			break;
		}
		default: {
			auto copy  = record.vertex;
			copy.stage = {};
			w.Put(copy);
			break;
		}
	}
	w.PutVector(record.specialization.buffers);
	w.PutVector(record.specialization.images);
	w.PutVector(record.specialization.samplers);
	return std::move(w.bytes);
}

bool Decode(std::span<const uint8_t> bytes, Record& record) {
	Reader   r {bytes};
	uint32_t stage           = 0;
	uint8_t  bindless_images = 0;
	uint8_t  non_fatal       = 0;
	r.Get(stage);
	r.Get(record.hash);
	r.Get(record.wave_size);
	r.Get(record.user_data_base);
	r.Get(record.push_cursor);
	r.Get(bindless_images);
	r.Get(non_fatal);
	if (!r.ok || stage == 0 || stage > static_cast<uint32_t>(ShaderType::TessellationEvaluation)) {
		return false;
	}
	record.stage           = static_cast<ShaderType>(stage);
	record.bindless_images = bindless_images != 0;
	record.non_fatal       = non_fatal != 0;
	r.GetVector(record.code);
	r.GetVector(record.back_code);
	r.GetVector(record.user_data);
	switch (InputKind(record.stage)) {
		case 'P': r.Get(record.pixel); break;
		case 'C': r.Get(record.compute); break;
		default: r.Get(record.vertex); break;
	}
	r.GetVector(record.specialization.buffers);
	r.GetVector(record.specialization.images);
	r.GetVector(record.specialization.samplers);
	return r.ok && r.offset == bytes.size() && !record.code.empty();
}

bool Log::Open(const std::filesystem::path& path) {
	// The namespace's switch: a plain Enabled() here is the member (whether the file is open).
	if (!ShaderWarmup::Enabled()) {
		return false;
	}
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	const auto       path_text = Common::PathToString(path);
	const FileHeader expected {};
	bool             rewrite = true;
	if (std::FILE* in = std::fopen(path_text.c_str(), "rb"); in != nullptr) {
		FileHeader header {};
		if (std::fread(&header, sizeof(header), 1, in) == 1 && header == expected) {
			rewrite = false;
			for (;;) {
				uint32_t size     = 0;
				uint64_t identity = 0;
				if (std::fread(&size, sizeof(size), 1, in) != 1) {
					break; // the end of the file
				}
				std::vector<uint8_t> bytes;
				if (std::fread(&identity, sizeof(identity), 1, in) == 1 && size <= (64u << 20u)) {
					bytes.resize(size);
				}
				if (bytes.size() != size || (size != 0 && std::fread(bytes.data(), size, 1, in) != 1)) {
					rewrite = true; // a record cut short when a session ended mid-write
					break;
				}
				if (m_known.insert(identity).second) {
					m_loaded.emplace_back(identity, std::move(bytes));
				}
			}
		}
		// Another layout (another build's structures): the old records no longer decode.
		std::fclose(in);
	}
	if (rewrite) {
		// A new file, or the good records again without a damaged tail.
		std::FILE* out = std::fopen(path_text.c_str(), "wb");
		if (out == nullptr) {
			return false;
		}
		std::fwrite(&expected, sizeof(expected), 1, out);
		for (const auto& [identity, bytes]: m_loaded) {
			const uint32_t size = static_cast<uint32_t>(bytes.size());
			std::fwrite(&size, sizeof(size), 1, out);
			std::fwrite(&identity, sizeof(identity), 1, out);
			std::fwrite(bytes.data(), bytes.size(), 1, out);
		}
		std::fclose(out);
	}
	m_file = std::fopen(path_text.c_str(), "ab");
	if (m_file != nullptr) {
		LOGF("Shader warmup: %zu recorded shader permutations in %s\n", m_loaded.size(),
		     path_text.c_str());
	}
	return m_file != nullptr;
}

Log::~Log() {
	if (m_file != nullptr) {
		std::fclose(m_file);
	}
}

void Log::Add(const Record& record, uint64_t identity) {
	std::lock_guard lock(m_mutex);
	if (m_file == nullptr || !m_known.insert(identity).second) {
		return;
	}
	const auto     bytes = Encode(record);
	const uint32_t size  = static_cast<uint32_t>(bytes.size());
	std::fwrite(&size, sizeof(size), 1, m_file);
	std::fwrite(&identity, sizeof(identity), 1, m_file);
	std::fwrite(bytes.data(), bytes.size(), 1, m_file);
	std::fflush(m_file);
}

bool Log::Known(uint64_t identity) {
	std::lock_guard lock(m_mutex);
	return m_known.contains(identity);
}

std::vector<std::pair<uint64_t, std::vector<uint8_t>>> Log::TakeLoaded() {
	std::lock_guard lock(m_mutex);
	return std::move(m_loaded);
}

} // namespace Libs::Graphics::ShaderWarmup
