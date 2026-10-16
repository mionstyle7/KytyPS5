#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

// Shader warmup (the Demon's Souls fork's shader-warmup-cache): every shader permutation the game
// uses is recorded with the inputs of its compile (the guest code, the stage inputs and the
// resource specialization), never with Vulkan objects or guest memory. At the next start, worker
// threads compile every recorded permutation the shader disk cache lacks (after an emulator update,
// or under other switches) into the disk cache, so the game finds them there instead of translating
// on the GPU thread in the middle of play. The records do not depend on the emulator build: a new
// recompiler version compiles the same inputs again.
namespace Libs::Graphics::ShaderWarmup {

struct Record {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              wave_size       = 64;
	uint32_t              user_data_base  = 0;
	uint32_t              push_cursor     = 0;
	bool                  bindless_images = false;
	bool                  non_fatal       = false;
	std::vector<uint32_t> code;
	std::vector<uint32_t> back_code;
	std::vector<uint32_t> user_data;
	ShaderVertexInputInfo  vertex {};
	ShaderPixelInputInfo   pixel {};
	ShaderComputeInputInfo compute {};
	ShaderRecompiler::IR::ResourceSpecialization specialization;
};

[[nodiscard]] std::vector<uint8_t> Encode(const Record& record);
[[nodiscard]] bool                 Decode(std::span<const uint8_t> bytes, Record& record);

// The title's record file, appended as permutations appear (each record flushed at once, so a
// session that ends abruptly keeps what it saw). KYTY_SHADER_WARMUP=0 turns recording and the
// startup compiles off.
class Log {
public:
	// Reads the records already in the file; false when warmup is off or the file is unusable.
	bool Open(const std::filesystem::path& path);
	~Log();

	// Appends the record unless one with its identity (a hash of the permutation's disk cache key
	// without the session part: the same across emulator builds) is in the file.
	void Add(const Record& record, uint64_t identity);
	[[nodiscard]] bool Known(uint64_t identity);

	// The records read by Open, with their identities (moved out).
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> TakeLoaded();

	[[nodiscard]] bool Enabled() const { return m_file != nullptr; }

private:
	std::mutex                        m_mutex;
	std::FILE*                        m_file = nullptr;
	std::unordered_set<uint64_t>      m_known;
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> m_loaded;
};

[[nodiscard]] bool Enabled();

} // namespace Libs::Graphics::ShaderWarmup

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_ */
