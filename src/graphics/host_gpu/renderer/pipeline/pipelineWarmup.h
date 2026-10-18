#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEWARMUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEWARMUP_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

// Pipeline warmup (KYTY_PIPELINE_WARMUP, default on): the shader warmup compiles every recorded
// shader permutation before play, but each new graphics or compute pipeline was still created on
// the GPU thread when its first draw came (~600 a session, up to 150 ms each), and every one
// stopped the frame. Here every pipeline the game creates is recorded as the Vulkan create info it
// was built from (SPIR-V, layout bindings and fixed-function state, never guest memory), and at the
// next start worker threads create them all, fully optimized, while the intro and menus play. A
// draw whose pipeline matches a prebuilt one byte for byte takes it instead of compiling.
//
// The records hold SPIR-V, so a recompiler change that alters a shader's SPIR-V makes its old
// recipes stop matching (they are dropped after a few sessions unused) and the session records the
// new ones.
namespace Libs::Graphics::PipelineWarmup {

[[nodiscard]] bool Enabled();

// The SPIR-V a shader module was created from (any thread; kept while warmup is on).
void RegisterModule(vk::ShaderModule module, std::span<const uint32_t> spirv);

class Store {
public:
	Store(GraphicContext& graphics, vk::PipelineCache driver_cache, std::filesystem::path path);
	~Store();
	Store(const Store&)            = delete;
	Store& operator=(const Store&) = delete;

	// Starts the workers once (the GPU thread, at its first pipeline: the device and the bindless
	// layout exist by then).
	void Start();

	// GPU thread. Records the pipeline the build describes if it is new; returns a prebuilt one
	// matching it (the caller owns it), or nullptr to build it as usual.
	vk::Pipeline TakeGraphics(const GraphicsPipelineBuild& build,
	                          const PipelineCache::Pipeline& pipeline);
	vk::Pipeline TakeCompute(const PipelineCache::Pipeline& pipeline, vk::ShaderModule module,
	                         uint32_t wave_size, uint64_t shader_hash);

	struct Module {
		uint64_t              hash = 0;
		std::vector<uint32_t> words;
	};

private:
	enum class State : uint8_t { Queued, Building, Built, Taken, Failed };
	struct Recipe {
		std::vector<uint8_t>  bytes;
		std::vector<uint64_t> modules;
		uint32_t              last_session = 0;
		State                 state        = State::Taken;
		vk::Pipeline          pipeline     = nullptr;
		bool                  used_logged  = false;
	};

	void                      Load();
	void                      WriteRecord(char type, uint64_t hash, std::span<const uint8_t> bytes);
	void                      WriteModule(const Module& module);
	vk::Pipeline              Take(uint64_t hash, std::vector<uint8_t>&& bytes,
	                               std::vector<std::shared_ptr<const Module>>&& modules);
	void                      Worker();
	vk::Pipeline              Build(uint64_t hash, const Recipe& recipe);
	vk::ShaderModule          ModuleFor(uint64_t hash);
	vk::PipelineLayout        LayoutFor(std::span<const uint8_t> layout_bytes, bool& ok);
	void                      LogProgress();

	GraphicContext&                                        m_graphics;
	vk::PipelineCache                                      m_driver_cache;
	std::filesystem::path                                  m_path;
	std::mutex                                             m_mutex;
	std::condition_variable                                m_changed;
	std::FILE*                                             m_file    = nullptr;
	uint32_t                                               m_session = 0;
	std::unordered_map<uint64_t, Recipe>                   m_recipes;
	std::unordered_map<uint64_t, std::vector<uint32_t>>    m_module_words; // loaded or written
	std::unordered_map<uint64_t, vk::ShaderModule>         m_modules;      // created for builds
	struct LayoutObjects {
		vk::DescriptorSetLayout set    = nullptr;
		vk::PipelineLayout      layout = nullptr;
	};
	std::unordered_map<uint64_t, LayoutObjects> m_layouts;
	std::vector<uint64_t>                       m_queue;
	size_t                                      m_next    = 0;
	std::atomic<bool>                           m_stop    = false;
	bool                                        m_started = false;
	std::vector<std::thread>                    m_threads;
	// Statistics: prebuilt pipelines built, failed, taken by draws, and pipelines recorded anew.
	uint64_t                                    m_built = 0, m_failed = 0, m_taken = 0, m_recorded = 0,
	                                            m_waited = 0;
	size_t                                      m_active = 0;
	std::chrono::steady_clock::time_point       m_begin;
};

} // namespace Libs::Graphics::PipelineWarmup

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEWARMUP_H_ */
