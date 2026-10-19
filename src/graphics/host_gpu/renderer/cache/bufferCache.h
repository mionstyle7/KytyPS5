#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCachePages.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <map>
#include <mutex>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	// Diagnostics (readback hot regions): the guest shader whose bindings are being obtained, so
	// GPU writes can be attributed to it. Set by the draw and dispatch paths on the GPU thread.
	static void SetCurrentWriter(uint64_t shader_hash);
	void        NoteReadback(uint64_t vaddr, double ms, bool gpu_bytes, uint64_t age);
	void        NoteGuestReadbackTiming(double total_ms, double service_ms, uint32_t steps,
	                                    bool gpu_bytes, uint64_t age);
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static_assert(CACHING_PAGEBITS == BufferCachePages::kPageBits &&
	              CACHING_NUMPAGES == BufferCachePages::kNumPages &&
	              LOWER_ADDRESS_SIZE == BufferCachePages::kLowerAddressSize &&
	              LibKernel::Memory::kExtendedMemoryBase == BufferCachePages::kExtendedMemoryBase,
	              "bufferCachePages.h is out of date");
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Marks [vaddr, vaddr + size) CPU-modified ahead of writes the guest has not made yet, so they
	// do not fault one page at a time. A tracked chunk holding GPU-written bytes is left alone:
	// this never downloads.
	void                   InvalidateMemoryNoDownload(uint64_t vaddr, uint64_t size);
	// The same as InvalidateMemory for a range the caller is about to write whole (a file read
	// into guest memory, a DMA copy or fill done on the CPU): the GPU-written bytes of the pages
	// the write covers whole are dropped instead of read back, since they are about to be
	// replaced; only the pages it covers in part are read back (the Demon's Souls fork's
	// c4a7256f: sync downloads 0.71 -> 0 ms a frame at its busiest spot). KYTY_OVERWRITE_DROP=0
	// reads everything back as before.
	void                   InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	// The emulator wrote guest memory itself (a CP write, copy or fill done on the CPU): pages an
	// upload kept writable (KYTY_UPLOAD_ONCE) took it without a fault, so they are marked here.
	void                   NoteHostWrite(uint64_t vaddr, uint64_t size);
	// Invalidates guest memory the emulator is about to write on the CPU (a DMA copy or fill done
	// on the GPU thread), as a write fault would, so the write itself does not fault
	// (RenderContext::InvalidateMemory).
	std::function<void(uint64_t, uint64_t)> on_host_write;
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// needs_device_address: the caller reads the data through a buffer device address, which
	// the stream buffer used for small CPU-written reads does not have.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer      = false,
	                                                        BufferId id                   = {},
	                                                        bool     needs_device_address = false);
	// A written binding of which the shader's stores reach only [written_vaddr, written_vaddr +
	// written_size): the whole range is synchronized, only that part becomes GPU-written.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferWritten(uint64_t vaddr, uint64_t size,
	                                                               uint64_t written_vaddr,
	                                                               uint64_t written_size,
	                                                               BufferId id = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// GPU write bitmap (KYTY_WRITE_BITMAP=1, rework.md Phase 2 first step): one bit per 4 KiB
	// guest page below 1 TiB, set by shaders through bindings whose writes the host cannot bound.
	// Its device address, or 0 when it is off or unavailable.
	[[nodiscard]] vk::DeviceAddress WriteBitmapAddress();
	// ObtainBuffer for such a written binding: its pages become bitmap-tracked, so a guest read
	// of a page the GPU did not write needs no download (ReadMemoryStep).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferTracked(uint64_t vaddr, uint64_t size,
	                                                               BufferId id);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// The same, under the dirty-range lock: for a GPU-read delegate (Memory::SetGpuReadDelegate).
	[[nodiscard]] bool              HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	// Guest threads (KYTY_DIRECT_READBACK, default on): reads [vaddr, vaddr + size) of GPU-written
	// bytes whose writers were all submitted, without the GPU thread: the window around them is
	// copied on the side queue after a wait for the newest writer only, put in guest memory, and
	// read from there. The GPU thread lifts the pages' protection later (PublishDirectRead).
	// False to decline (the caller takes the ordinary readback).
	[[nodiscard]] bool              TryDirectRead(uint64_t vaddr, void* data, uint64_t size);
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void                            ProcessFaultBuffer();
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	[[nodiscard]] bool              WriteClean(uint64_t vaddr, const void* data, uint64_t size);
	// Any thread (RenderContext::TryWriteClean): none of the bytes is GPU-written or on its way
	// back, and their page still holds bytes that are, so a store to them would make the guest
	// wait for a download it does not need.
	[[nodiscard]] bool              CanWriteCleanConcurrently(uint64_t vaddr, uint64_t size) const;
	// Any thread: the bytes were just written to guest memory by an emulated store; the GPU
	// thread copies them to the GPU (ApplyCleanWrite) before it processes later submissions.
	void                            PostCleanWrite(uint64_t vaddr, uint64_t size);
	// Any thread: some of the bytes are GPU-written and none is on its way back from the GPU.
	[[nodiscard]] bool              IsGpuDirtyNotDownloading(uint64_t vaddr, uint64_t size) const;
	// Any thread: WriteClean on the GPU thread, waiting for the GPU thread but not for the GPU.
	[[nodiscard]] bool              WriteCleanSync(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	void                            SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	void                            RunGarbageCollector();
	// GPU thread, at the end of each guest submission (KYTY_READBACK_PREDICT=1 on; off by default): readback
	// windows the game threads read GPU-written bytes from again and again are downloaded ahead,
	// asynchronously, and their pages unprotected once the copy is in guest memory, so the next
	// read neither faults nor waits for the GPU thread and the GPU.
	void                            PrefetchReadbacks();
	// After the guest unmapped [vaddr, vaddr + size) (RenderContext::UnmapMemory), with `mapped` the
	// memory still mapped: the buffers the range holds whole go, and a buffer it holds in part,
	// mapped in no more than half of it now, is cut to its mapped runs (KYTY_UNMAP_SPLIT=0 off).
	void                            ReleaseUnmappedBuffers(uint64_t vaddr, uint64_t size,
	                                                       const RangeSet& mapped);
	// Diagnostics, GPU thread: makes guest memory current for [vaddr, vaddr + size) by
	// downloading what the GPU wrote there (drains the GPU).
	void                            DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size);

	// Diagnostics: the guest shader whose bindings are being prepared on this thread, if any.
	inline static thread_local uint64_t s_diag_shader_hash = 0;

private:
	friend struct BufferCacheTestAccess;

	void ReleaseBuffer(BufferId id);
	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// Records bytes a binding makes GPU-written (after SynchronizeBuffer marked their pages).
	void MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked = false);
	// ReadMemoryStep: drops the pages of the readback window the bitmap shows the GPU did not
	// write. True with wait_tick set when an asynchronous caller must first wait for that tick.
	bool RefineWithWriteBitmap(const Buffer& buffer, uint64_t vaddr, uint64_t size, bool async,
	                           uint64_t& wait_tick);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// One readback attempt on the GPU thread: 0 when done, else the tick to wait for.
	// Side readbacks (KYTY_SIDE_READBACK=1): a game thread's read of GPU-written bytes whose
	// writers were all submitted is copied on a second queue that waits only for the newest of
	// them, not behind everything queued since. Published by the faulting thread.
	struct SideTicket {
		uint32_t                                   slot  = 0;
		uint64_t                                   value = 0;
		uint64_t                                   begin = 0;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	struct SideState;
	static constexpr uint64_t SideRetry = UINT64_MAX;
	static constexpr uint64_t SideIssued = UINT64_MAX - 1;
	bool TrySideReadback(Buffer& buffer, uint64_t begin, uint64_t size, SideTicket& ticket);
	void FinishSideReadback(const SideTicket& ticket);
	void WaitSideRanges(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool SideBusy() const;
	[[nodiscard]] uint64_t NewestWriteTick(uint64_t begin, uint64_t size) const; // lock held
	std::unique_ptr<SideState> m_side;
	// Side command buffers, their pool and next_value: the GPU thread's side readbacks and the
	// guest threads' direct reads.
	std::mutex                 m_side_mutex;
	// Registered buffers by guest address, for direct reads off the GPU thread (a buffer found
	// here while a side slot is busy is not released until the slot is free: ReleaseBuffer).
	struct DirectBuffer {
		uint64_t   end    = 0;
		vk::Buffer handle = nullptr;
	};
	std::mutex                         m_direct_buffers_mutex;
	std::map<uint64_t, DirectBuffer>   m_direct_buffers;
	// Windows direct reads put in guest memory, with the newest writer tick they cover.
	struct DirectWindow {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::mutex                         m_direct_cache_mutex;
	std::array<DirectWindow, 64>       m_direct_cache {};
	size_t                             m_direct_next = 0;
	void PublishDirectRead(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t newest);
	RangeSet                   m_side_ranges; // guarded by m_dirty_ranges_mutex
	// The tick of the last GPU write per 1 MiB region, and writes over 64 regions apart
	// (guarded by m_dirty_ranges_mutex).
	struct BigWrite {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::unordered_map<uint64_t, uint64_t> m_region_write_ticks;
	std::vector<BigWrite>                  m_big_writes;
	SideTicket*                            m_step_ticket = nullptr; // GPU thread, set per step
	uint8_t                                m_step_reason = 0;       // GPU thread: why the step returned

	[[nodiscard]] uint64_t ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
	                                      bool from_gpu_thread, bool async);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	void DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                          uint64_t total_size);

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Every in-flight download's ranges, one entry each: m_downloading_ranges is their union. A
	// publication removes only its own entries (ISSUES #26: the first of two downloads of one
	// window cleared the range while the second was in flight; a game thread then took the
	// page as current, wrote it, and the second publication put older GPU bytes over that).
	std::vector<std::pair<uint64_t, uint64_t>>         m_downloading_list;
	// The asynchronous downloads among them, with the tick whose completion publishes each
	// (guarded by m_dirty_ranges_mutex). A reader of a page waits for the downloads covering it,
	// not for the newest download of any page (the Demon's Souls fork: readiness checked only
	// over the pages actually written back, +6.5%).
	struct DownloadTick {
		uint64_t start = 0;
		uint64_t bytes = 0;
		uint64_t tick  = 0;
	};
	std::vector<DownloadTick>                          m_download_ticks;
	// The newest tick among the asynchronous downloads covering [vaddr, vaddr + size), or 0 when
	// none is known (none in flight, or only side readbacks). Any thread.
	[[nodiscard]] uint64_t DownloadTickFor(uint64_t vaddr, uint64_t size) const;
	// Write bitmap (WriteBitmapAddress): the buffer, the whole pages whose GPU writes all came
	// through tracked bindings, and per 4 MiB region the last tick a tracked binding wrote it.
	// Guarded by m_dirty_ranges_mutex, as m_gpu_modified_ranges.
	std::unique_ptr<Buffer>                            m_write_bitmap;
	bool                                               m_write_bitmap_failed = false;
	RangeSet                                           m_bitmap_tracked;
	std::unordered_map<uint64_t, uint64_t>             m_bitmap_ticks;
	uint64_t                                           m_bitmap_dropped_pages = 0;
	uint64_t                                           m_bitmap_written_pages = 0;
	void ApplyCleanWrite(uint64_t vaddr, uint64_t size);
	// The tick of the latest asynchronous readback (KYTY_ASYNC_READBACK); GPU thread only.
	// Written on the GPU thread, always for a submitted tick; game threads read it to wait for
	// downloads in flight without asking the GPU thread (ReadMemory).
	std::atomic<uint64_t>                              m_last_async_download_tick {0};
	// Readback prediction (PrefetchReadbacks), GPU thread only: per readback window (its aligned
	// start) when game threads read it, and whether a prefetch landed since the last read.
	struct PredictWindow {
		uint64_t probe      = 0; // an address read in it, to find its buffer
		int64_t  first_ms   = 0;
		int64_t  last_ms    = 0;
		uint32_t gpu_reads  = 0;
		bool     learned    = false;
		bool     prefetched = false;
	};
	std::unordered_map<uint64_t, PredictWindow>        m_predict;
	struct PredictStats {
		uint64_t served = 0, downloaded = 0, after_prefetch = 0, prefetches = 0, bytes = 0,
		         lifted_pages = 0;
		int64_t  next_ms = 0;
	};
	PredictStats                                       m_predict_stats;
	void NotePredictRead(uint64_t vaddr, bool gpu_bytes);
	// Relaxed readback (KYTY_RELAXED_READBACK=1, off by default; the Astro Bot fork's d23ab526):
	// a game thread's read of a learned window whose bytes are all on their way back (a download
	// recorded) does not wait for it: the pages open read-only with the bytes guest memory holds,
	// as a CPU read on hardware sees memory before the GPU's write lands. The download publishes
	// as usual, the next GPU write closes the pages again, and writes still synchronize. GPU
	// thread; true when it granted the read.
	bool TryGrantStaleRead(uint64_t vaddr, uint64_t size);
	uint64_t m_stale_reads = 0;
	uint64_t m_waited_reads = 0;
	// Lifts the protection of the pages of [begin, end) protected as GPU-written whose bytes are
	// all in guest memory (none GPU-written or downloading), one change per run; the pages lifted.
	uint64_t LiftCurrentPages(uint64_t begin, uint64_t end);
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	// Guest reads UploadCopies leaves to the recording thread: staging destination, guest
	// address, bytes.
	struct UploadRead {
		uint8_t* destination;
		uint64_t address;
		uint64_t size;
	};
	std::vector<UploadRead> m_upload_reads;
	StreamBuffer                                       m_stream_buffer;
	StreamBuffer                                       m_download_buffer;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	uint64_t                                           m_total_used_memory = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// As configured; the ones above come down while memory is short (as in the texture cache).
	uint64_t m_base_trigger_gc_memory     = 0;
	uint64_t m_base_critical_gc_memory    = 0;
	uint64_t m_applied_trigger_gc_memory  = UINT64_MAX; // as in the texture cache
	uint64_t m_applied_critical_gc_memory = UINT64_MAX;
	uint64_t m_gc_tick            = 0;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
