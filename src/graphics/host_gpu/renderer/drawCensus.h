#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCENSUS_H_

#include <chrono>
#include <cstdint>
#include <span>

namespace Libs::Graphics {
struct ShaderStageRuntime;
} // namespace Libs::Graphics

// Draw records, step 1: how many draws and dispatches would a record serve (KYTY_DRAW_CENSUS=1).
//
// The native draw records of the Demon's Souls fork replay a draw from what its first use
// prepared (descriptor set, pipeline, render targets) while proofs show nothing it depends on
// changed: +24% there. Whether that pays here depends on how this title draws: a record serves a
// draw whose programs and resolved resources (buffer, image and sampler descriptors, render
// targets) were the same in an earlier frame. Constants (flattened SRT, user data) are copied
// anew by a record, so they may differ.
//
// This changes nothing: each draw and dispatch is keyed and timed, and every 10 s the log gives
//   repeat     the same key in an earlier frame (what a record would serve),
//   same-frame the same key earlier in this frame only (a record stored this frame),
//   new        first sighting,
//   exact      repeats whose constants were the same too (no relocation needed),
// with the average time the normal path took for each, so the time records could save is the
// repeat draws' time less a replay's (~1 us).
namespace Libs::Graphics::DrawCensus {

enum class Kind : uint8_t { Draw, Dispatch };

[[nodiscard]] bool Enabled();

// Times one draw or dispatch handler (GPU thread). Classify inside it names its class; one that
// returns before (an empty or skipped draw) counts as skipped.
class Scope {
public:
	explicit Scope(Kind kind);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

	// Called by ClassifyDraw/ClassifyDispatch for the innermost scope.
	void Set(uint8_t cls, bool exact, uint64_t shader) {
		m_class  = cls;
		m_exact  = exact;
		m_shader = shader;
	}

private:
	std::chrono::steady_clock::time_point m_start;
	Kind                                  m_kind;
	bool                                  m_active = false;
	uint8_t                               m_class  = 0; // 0: skipped
	bool                                  m_exact  = false;
	Scope*                                m_outer  = nullptr;
	uint64_t                              m_shader = 0; // the pixel or compute shader
};

// A draw about to be prepared: its stages (programs and resolved resources), its render
// targets (image slot ids; 0 for none) and topology, in frame `frame`.
void ClassifyDraw(std::span<const ShaderStageRuntime* const> stages,
                  std::span<const uint64_t> targets, uint32_t topology, uint64_t frame);

// A dispatch about to be prepared.
void ClassifyDispatch(const ShaderStageRuntime& stage, uint64_t frame);

} // namespace Libs::Graphics::DrawCensus

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWCENSUS_H_ */
