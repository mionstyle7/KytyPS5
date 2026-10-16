#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Whether the page holding vaddr is read-protected for tracking. Unlocked: a hint that can
	// be stale by the time the caller acts on it.
	[[nodiscard]] bool IsReadWatched(uint64_t vaddr) const noexcept;

	// The host protection of [vaddr, vaddr + size) was replaced behind the tracker (a view mapped
	// again with its own access by a partial unmap, a guest mprotect that allows writes): the
	// watched pages get the protection their counts ask for again, so CPU writes to them fault
	// (and reach the GPU) once more. Unwatched pages keep what they were given. Any thread,
	// outside the guest address-space lock.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size);

	// KYTY_ASYNC_REPROTECT (default on; =0 off), after the Demon's Souls fork: within this scope the
	// read protection of pages that GPU work is about to write (watcher edges to no-access) is
	// applied by a worker thread instead of the caller. The GPU writes those pages only once the
	// command buffer runs, later than the worker applies it; every other protection change waits
	// for the queued ones first, so their order holds. In the frames that write tracked buffers
	// those VirtualProtect calls took a few percent of the GPU thread.
	class DeferReadProtectScope {
	public:
		DeferReadProtectScope();
		~DeferReadProtectScope();
		KYTY_CLASS_NO_COPY(DeferReadProtectScope);

	private:
		bool m_previous;
	};
	// Waits until every queued protection is applied.
	static void DrainDeferredProtection();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
