#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Bumped whenever pages become CPU-dirty anywhere (a write fault, an untrack, an
// invalidation). A consumer that must see every CPU write, such as the pre-dispatch sync of
// all buffers for global-memory shaders, can skip its walk while this has not moved.
inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// Bumped by the GPU thread when it starts a guest submission (its first slice).
inline std::atomic<uint64_t> g_guest_submission_seq {1};

// Bumped by the GPU thread at every slice of a guest submission it processes (a submission's
// start, and its resumption after a WAIT_REG_MEM or another queue's turn): the unit within which
// a CPU-written page is uploaded once (RegionManager::ForEachUploadRangeOnce).
inline std::atomic<uint64_t> g_upload_epoch {1};

// KYTY_UPLOAD_ONCE=1 (default off): read-only uploads keep a page they upload CPU-dirty and
// writable for the rest of the slice instead of write-protecting it at once. Off by default: the
// title's write faults in the menu stayed at ~38,000 a second with it (each page is written about
// once per slice, so there was no fault ping-pong to remove) and the menu's frame rate did not move.
inline bool g_upload_once = [] {
	const char* value = std::getenv("KYTY_UPLOAD_ONCE");
	return value != nullptr && value[0] == '1';
}();

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	// Lock-free "any page in this region may be dirty" summaries. A binding that spans a
	// whole heap covers hundreds of regions; without these every draw locks and walks each
	// one. False is exact (set only after a walk saw no bits); true is conservative.
	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (source == DirtySource::Cpu) {
			// A page dirty again (or cleaned) by other means is no longer kept by an upload, and a
			// write reported here (a fault, a host write) is uploaded again in this slice.
			m_kept.UnsetRange(start, end);
			if constexpr (enable) {
				m_uploaded.UnsetRange(start, end);
				m_complete_epoch = 0;
			}
		}
		if constexpr (source == DirtySource::Gpu) {
			// A new GPU write, or the bytes current again: a stale read grant ends here.
			m_stale_readable.UnsetRange(start, end);
		}
		if constexpr (enable) {
			bits.SetRange(start, end);
			StoreSummary<source>(true);
			if constexpr (source == DirtySource::Cpu) {
				g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
			}
		} else {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			if constexpr (source == DirtySource::Cpu) {
				m_kept.UnsetRange(start, end);
			} else {
				m_stale_readable.UnsetRange(start, end);
			}
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// A read-only upload of the CPU-dirty pages of a range, for a game that rewrites the same pages
	// all the time. The usual upload write-protects each page it copies, and a heap-wide binding
	// syncs the whole heap at nearly every draw: a page the game wrote again right after was copied
	// and protected again, and the next write faulted again, over and over within one submission
	// (~40,000-70,000 write faults a second in the title, each an exception on a game thread plus a
	// VirtualProtect on the GPU thread). Here a page written since its last upload is copied and
	// left CPU-dirty and writable ("kept"); later syncs in the same slice (g_upload_epoch) skip it,
	// as the game wrote what this submission reads before submitting it. The first sync of a kept
	// page in a later slice copies it again and write-protects it, so a page the game stops writing
	// is soon clean. At most one fault per written page and two copies per slice remain.
	template <typename Func>
	void ForEachUploadRangeOnce(uint64_t vaddr, uint64_t size, uint64_t epoch, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (m_cpu_dirty.FirstRangeFrom(start).first >= end) {
			return;
		}
		if (m_uploaded_epoch != epoch) {
			m_uploaded.Clear();
			m_uploaded_epoch = epoch;
		} else if (m_complete_epoch == epoch) {
			return; // every dirty page of the region was uploaded in this slice
		}
		const RegionBits dirty(m_cpu_dirty, start, end);
		const RegionBits todo = dirty & ~m_uploaded;
		if (todo.None()) {
			return;
		}
		// Kept from an earlier slice: copied once more and protected, the usual upload.
		const RegionBits renew = todo & m_kept;
		if (renew.Any()) {
			m_cpu_dirty &= ~renew;
			m_kept &= ~renew;
			if (m_cpu_dirty.None()) {
				StoreSummary<DirtySource::Cpu>(false);
			}
			UpdateProtection<true, false>();
		}
		// Written in this slice and not uploaded yet: copied and kept writable.
		m_kept |= todo & ~renew;
		m_uploaded |= todo;
		if ((m_cpu_dirty & ~m_uploaded).None()) {
			m_complete_epoch = epoch;
		}
		for (const auto [first, last]: todo) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Relaxed readback (BufferCache): guest threads may read the GPU-dirty pages of the range with
	// the bytes guest memory holds now (a download of the new ones already recorded), until the
	// next GPU write to them or their unmark. Writes still fault. Caller holds lock.
	void GrantStaleRead(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits dirty(m_gpu_dirty, start, end);
		if (dirty.None()) {
			return;
		}
		m_stale_readable |= dirty;
		UpdateProtection<false, true>();
	}

	// The tracker mirrors the CPU summary in one bit per region (MemoryTracker::IsRangeCpuCleanHint).
	// Set before the region is published.
	void SetCpuSummaryBit(std::atomic<uint64_t>* word, uint64_t bit) noexcept {
		m_cpu_summary_word = word;
		m_cpu_summary_bit  = bit;
	}

	TrackingSpinLock lock;

private:
	// Every summary change goes through here; the CPU one is mirrored in the tracker's bitmap at
	// the same point, before any page protection changes.
	template <DirtySource source>
	void StoreSummary(bool value) {
		GetSummary<source>().store(value, std::memory_order_release);
		if constexpr (source == DirtySource::Cpu) {
			if (m_cpu_summary_word != nullptr) {
				if (value) {
					m_cpu_summary_word->fetch_or(m_cpu_summary_bit, std::memory_order_acq_rel);
				} else {
					m_cpu_summary_word->fetch_and(~m_cpu_summary_bit, std::memory_order_acq_rel);
				}
			}
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? (~m_gpu_dirty | m_stale_readable) : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	// The constructor fills m_cpu_dirty, so the CPU summary starts true.
	std::atomic<bool> m_cpu_maybe_dirty {true};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	std::atomic<uint64_t>* m_cpu_summary_word = nullptr;
	uint64_t               m_cpu_summary_bit  = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
	// GPU-dirty pages guest threads may read with their current guest bytes (GrantStaleRead).
	RegionBits   m_stale_readable;
	// ForEachUploadRangeOnce: pages left CPU-dirty and writable by an upload, and the pages
	// uploaded in slice m_uploaded_epoch.
	RegionBits   m_kept;
	RegionBits   m_uploaded;
	uint64_t     m_uploaded_epoch = 0;
	// The slice in which every CPU-dirty page of the region was found uploaded.
	uint64_t     m_complete_epoch = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
