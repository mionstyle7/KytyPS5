#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_FRAMEBUDGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_FRAMEBUDGET_H_

#include <cstdint>

// Diagnostics: where each emulator thread spends a frame. Every 5 s the log gives, per thread and
// in milliseconds per presented frame, the time spent in each phase below, exclusive of the
// phases nested in it (a texture found while a draw is prepared counts as textures, not draw).
// Threads are named where they start (gpu, record, submit); scopes reached on any other thread
// (the game's threads waiting on the GPU thread, a readback) are summed under "others", without
// their time outside those scopes. KYTY_FRAME_BUDGET=0 turns it off.
namespace Libs::Graphics::FrameBudget {

enum class Phase : uint8_t {
	Other,      // in none of the phases below
	Idle,       // nothing to do (waiting for work)
	Blocked,    // work queued, but waiting for the game (a label, the constant engine)
	Requests,   // requests of the game's threads served by the GPU thread
	Pm4,        // reading the game's command buffers (packets, registers)
	Draw,       // draw state: render targets, descriptors, Vulkan commands
	Dispatch,   // compute dispatch state
	Shaders,    // shader and pipeline state of a draw outside the phases below
	ShaderGet,  // finding a draw's shader program and walking its resource table (SRT)
	ShaderCompile, // translating a new shader (or permutation) to SPIR-V
	PipelineBuild, // the driver building a pipeline (vkCreate*Pipelines)
	PipelineWait,  // waiting for a pipeline being built on a worker
	Buffers,    // finding and synchronizing buffers (memory checks, uploads)
	Textures,   // finding, uploading and validating images
	Readback,   // the game reading memory the GPU wrote (downloads)
	WaitGpu,    // waiting for the host GPU to finish submitted work
	WaitRecord, // waiting for the recording thread to catch up
	Collect,    // garbage collection of the caches
	Flip,       // preparing a flip
	Present,    // presenting to the window
	Record,     // recording thread: making the Vulkan calls
	Submit,     // submitting command buffers to the queue
	GuestWait,  // a game thread waiting for the GPU thread
	Count
};

[[nodiscard]] bool Enabled();

namespace Detail {
Phase Enter(Phase phase);
void  Leave(Phase previous);
} // namespace Detail

class Scope {
public:
	explicit Scope(Phase phase) {
		if (Enabled()) {
			m_active   = true;
			m_previous = Detail::Enter(phase);
		}
	}
	~Scope() {
		if (m_active) {
			Detail::Leave(m_previous);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Phase m_previous = Phase::Other;
	bool  m_active   = false;
};

// The calling thread's name in the report (a string literal).
void NameThread(const char* name);

// Called once per presented frame; logs every 5 s.
void Present();

} // namespace Libs::Graphics::FrameBudget

#define KYTY_FRAME_BUDGET_CONCAT2(a, b) a##b
#define KYTY_FRAME_BUDGET_CONCAT(a, b)  KYTY_FRAME_BUDGET_CONCAT2(a, b)
#define KYTY_FRAME_BUDGET(phase)                                                                   \
	const ::Libs::Graphics::FrameBudget::Scope KYTY_FRAME_BUDGET_CONCAT(frame_budget_, __LINE__)( \
	    ::Libs::Graphics::FrameBudget::Phase::phase)

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_FRAMEBUDGET_H_
