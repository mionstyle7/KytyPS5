#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <chrono>

namespace Libs::Graphics {

namespace {
// Textures placed in system memory (CreateImage) that still exist.
std::atomic<uint64_t> g_host_placed_bytes {0};
std::atomic<uint64_t> g_host_placed_images {0};
std::atomic<uint64_t> g_host_placed_total {0};

// Texture defragmentation state (GraphicContext::BeginTextureDefragPass). The pool and the
// context are used from the GPU thread; g_defrag_open is read by DeleteImage on any thread.
VmaPool                        g_texture_pool       = nullptr;
uint32_t                       g_texture_pool_type  = UINT32_MAX;
VmaDefragmentationContext      g_defrag_context     = nullptr;
VmaDefragmentationPassMoveInfo g_defrag_pass {};
bool                           g_defrag_pass_open   = false;
uint32_t                       g_defrag_passes      = 0;
int64_t                        g_defrag_next_ms     = 0;
uint64_t                       g_defrag_moved_bytes  = 0; // whole run
uint64_t                       g_defrag_moved_images = 0;
uint64_t                       g_defrag_freed_bytes  = 0;
std::atomic<bool>              g_defrag_open {false};

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

bool DefragEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_DEFRAG");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	// Testing aid: KYTY_VRAM_LIMIT_MB=<n> caps every device-local heap at n MB, so a large card
	// behaves like a 4, 6 or 8 GB one: VMA reports the capped budget, allocations past it fail
	// or spill as they would there, and the caches' collection thresholds follow the budget.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* value = std::getenv("KYTY_VRAM_LIMIT_MB"); value != nullptr) {
		const auto  limit      = static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
		const auto& properties = GetPhysicalDeviceMemoryProperties();
		if (limit != 0) {
			for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
				const bool device_local = static_cast<bool>(properties.memoryHeaps[heap].flags &
				                                            vk::MemoryHeapFlagBits::eDeviceLocal);
				heap_limits[heap] = device_local ? std::min<VkDeviceSize>(
				                                       limit, properties.memoryHeaps[heap].size)
				                                 : VK_WHOLE_SIZE;
				if (device_local) {
					LOGF("KYTY_VRAM_LIMIT_MB: device heap %u capped at %" PRIu64 " MB (of %" PRIu64
					     " MB)\n",
					     heap, static_cast<uint64_t>(heap_limits[heap] >> 20u),
					     static_cast<uint64_t>(properties.memoryHeaps[heap].size >> 20u));
				}
			}
			info.pHeapSizeLimit = heap_limits.data();
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}

	// The texture pool: the video memory type a sampled texture takes, in 64 MB blocks (small
	// blocks empty, and are freed, sooner than VMA's 256 MB ones).
	if (DefragEnabled() &&
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
		VkImageCreateInfo sample {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		sample.imageType   = VK_IMAGE_TYPE_2D;
		sample.format      = VK_FORMAT_R8G8B8A8_UNORM;
		sample.extent      = {256, 256, 1};
		sample.mipLevels   = 1;
		sample.arrayLayers = 1;
		sample.samples     = VK_SAMPLE_COUNT_1_BIT;
		sample.tiling      = VK_IMAGE_TILING_OPTIMAL;
		sample.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo sample_alloc {};
		sample_alloc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		uint32_t type              = UINT32_MAX;
		if (vmaFindMemoryTypeIndexForImageInfo(allocator, &sample, &sample_alloc, &type) ==
		    VK_SUCCESS) {
			VmaPoolCreateInfo pool {};
			pool.memoryTypeIndex = type;
			pool.blockSize       = 64ull << 20u;
			if (vmaCreatePool(allocator, &pool, &g_texture_pool) == VK_SUCCESS) {
				g_texture_pool_type = type;
				vmaSetPoolName(allocator, g_texture_pool, "Kyty texture pool");
				LOGF("Texture pool: memory type %u, 64 MB blocks (defragmented when they hold much "
				     "more than is allocated)\n",
				     type);
			} else {
				g_texture_pool = nullptr;
			}
		}
	}

	// A system whose commit limit is nearly used up stops the session later with "Out of memory"
	// (ISSUES #21); say so at start. KYTY_COMMIT_WARN_MB sets the threshold (default 16 GB).
	uint64_t commit_limit = 0;
	uint64_t commit_free  = 0;
	if (Common::VirtualMemory::QueryCommit(commit_limit, commit_free)) {
		const char*    value     = std::getenv("KYTY_COMMIT_WARN_MB");
		const uint64_t warn_mb   = value != nullptr ? std::strtoull(value, nullptr, 10) : 16384u;
		LOGF("Commit limit (start): %" PRIu64 " MB free of %" PRIu64 " MB\n", commit_free >> 20u,
		     commit_limit >> 20u);
		if ((commit_free >> 20u) < warn_mb) {
			std::fprintf(stderr,
			             "Warning: only %" PRIu64 " MB of the system's commit limit (%" PRIu64
			             " MB, RAM plus page file) are free; a game can need 16-24 GB. If the "
			             "emulator stops with \"Out of memory\", close other programs or enlarge "
			             "the page file.\n",
			             commit_free >> 20u, commit_limit >> 20u);
			LOGF("Warning: only %" PRIu64 " MB of the commit limit are free (threshold %" PRIu64
			     " MB)\n",
			     commit_free >> 20u, warn_mb);
		}
	}
	return true;
}

static void DestroyImagePool(VmaAllocator allocator);
void        FlushBufferReclaimer(); // streamBuffer.cpp

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyImagePool(allocator);
	FlushBufferReclaimer();
	if (g_defrag_context != nullptr) {
		if (g_defrag_pass_open) {
			for (uint32_t i = 0; i < g_defrag_pass.moveCount; i++) {
				g_defrag_pass.pMoves[i].operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
			}
			(void)vmaEndDefragmentationPass(allocator, g_defrag_context, &g_defrag_pass);
			g_defrag_pass_open = false;
		}
		vmaEndDefragmentation(allocator, g_defrag_context, nullptr);
		g_defrag_context = nullptr;
		g_defrag_open    = false;
	}
	if (g_texture_pool != nullptr) {
		VmaStatistics stats {};
		vmaGetPoolStatistics(allocator, g_texture_pool, &stats);
		if (stats.allocationCount == 0) {
			vmaDestroyPool(allocator, g_texture_pool);
			g_texture_pool = nullptr;
		} else {
			LOGF("Texture pool: %u allocations still alive at shutdown; pool kept\n",
			     stats.allocationCount);
		}
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

// The system's commit limit (Windows: RAM + page file): an allocation fails when it is used up,
// whatever the VMA budget says (ISSUES #21: one Wolverine session commits ~23 GB).
static void LogCommit(const char* when) {
	uint64_t limit     = 0;
	uint64_t available = 0;
	if (Common::VirtualMemory::QueryCommit(limit, available)) {
		LOGF("Commit limit (%s): %" PRIu64 " MB free of %" PRIu64 " MB\n", when, available >> 20u,
		     limit >> 20u);
	}
}

// The video memory the emulator plans for, whatever the card has (KYTY_VRAM_TARGET_MB, default
// 14336 = what a 16 GB card leaves an application; 0 = the card's own budget). Senaxx, 2026-10-09:
// it must always fit on a 16 GB GPU. On cards with less the card's budget is the smaller one.
static uint64_t VramTargetBytes() {
	static const uint64_t target = [] {
		const char* value = std::getenv("KYTY_VRAM_TARGET_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {14336}) << 20u;
	}();
	return target;
}

uint64_t GraphicContext::MemoryShortfall() const {
	// KYTY_COMMIT_RESERVE_MB (default 0 = off): collecting video memory for a commit shortfall
	// was tried on 2026-10-09 and withdrawn. On a PC whose commit limit is short for other reasons
	// (other programs, guest memory, host heaps) it never catches up: the collectors wrote back
	// and freed ~100 images per interval from 7 GB of device use on, the frame rate fell to 7 fps
	// and glyph atlases lost letters. The limit is also dynamic on a system-managed page file. The
	// start-up warning and the log lines remain; the video memory target is the pressure that counts.
	static const uint64_t reserve = [] {
		const char* value = std::getenv("KYTY_COMMIT_RESERVE_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0}) << 20u;
	}();
	static std::atomic<int64_t>  next_ms {0};
	static std::atomic<uint64_t> shortfall {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto next = next_ms.load(std::memory_order_relaxed);
	if (now >= next && next_ms.compare_exchange_strong(next, now + 250, std::memory_order_relaxed)) {
		// The system's commit limit below the reserve.
		uint64_t limit     = 0;
		uint64_t available = 0;
		const uint64_t commit =
		    reserve != 0 && Common::VirtualMemory::QueryCommit(limit, available) &&
		            available < reserve
		        ? reserve - available
		        : 0;
		// Everything the device holds (images, buffers, the driver's own) past 90 % of the
		// budget, which the video memory target caps: the caches' own thresholds leave out the
		// bindless heap textures and render targets, so they alone cannot keep the total in it.
		uint64_t device = 0;
		if (CanReportMemoryUsage()) {
			const auto budget = GetTotalMemoryBudget();
			const auto usage  = GetDeviceMemoryUsage();
			const auto mark   = budget / 10 * 9;
			device            = budget != 0 && usage > mark ? usage - mark : 0;
		}
		const uint64_t value    = std::max(commit, device);
		const auto     previous = shortfall.exchange(value, std::memory_order_relaxed);
		// Every 10 s, where video memory is: per heap, the driver's count, VMA's blocks and what
		// is allocated in them (blocks much larger than their allocations: freed images whose
		// blocks other images keep alive), and the textures placed in system memory.
		static int64_t next_log_ms = 0;
		if (now >= next_log_ms && allocator != nullptr) {
			next_log_ms = now + 10000;
			const auto& properties = GetPhysicalDeviceMemoryProperties();
			VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(allocator, budgets);
			std::string heaps;
			for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
				const auto& b = budgets[heap];
				if (b.statistics.blockCount == 0 && b.usage == 0) {
					continue;
				}
				char line[256];
				std::snprintf(line, sizeof(line),
				              " heap%u(%s): usage=%" PRIu64 " budget=%" PRIu64 " blocks=%" PRIu64
				              "MB/%u allocated=%" PRIu64 "MB/%u;",
				              heap,
				              (properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
				                  ? "video"
				                  : "system",
				              static_cast<uint64_t>(b.usage >> 20u),
				              static_cast<uint64_t>(b.budget >> 20u),
				              static_cast<uint64_t>(b.statistics.blockBytes >> 20u),
				              b.statistics.blockCount,
				              static_cast<uint64_t>(b.statistics.allocationBytes >> 20u),
				              b.statistics.allocationCount);
				heaps += line;
			}
			if (g_texture_pool != nullptr) {
				VmaStatistics pool {};
				vmaGetPoolStatistics(allocator, g_texture_pool, &pool);
				char line[256];
				std::snprintf(line, sizeof(line),
				              " texture pool: blocks=%" PRIu64 "MB allocated=%" PRIu64
				              "MB/%u, defragmentation moved %" PRIu64 " MB (%" PRIu64
				              " images) and freed %" PRIu64 " MB so far;",
				              static_cast<uint64_t>(pool.blockBytes >> 20u),
				              static_cast<uint64_t>(pool.allocationBytes >> 20u),
				              pool.allocationCount, g_defrag_moved_bytes >> 20u,
				              g_defrag_moved_images, g_defrag_freed_bytes >> 20u);
				heaps += line;
			}
			LOGF("VRAM (10 s): device %" PRIu64 " MB of %" PRIu64 " MB budget (%" PRIu64
			     " %%);%s textures in system memory %" PRIu64 " MB (%" PRIu64 " images, %" PRIu64
			     " placed so far)\n",
			     GetDeviceMemoryUsage() >> 20u, GetTotalMemoryBudget() >> 20u,
			     GetTotalMemoryBudget() != 0 ? GetDeviceMemoryUsage() * 100 / GetTotalMemoryBudget()
			                                 : uint64_t {0},
			     heaps.c_str(), g_host_placed_bytes.load(std::memory_order_relaxed) >> 20u,
			     g_host_placed_images.load(std::memory_order_relaxed),
			     g_host_placed_total.load(std::memory_order_relaxed));
		}
		static std::atomic<uint32_t> reported {0};
		if ((previous == 0) != (value == 0) && reported.fetch_add(1) < 64) {
			LOGF("Memory: %s (commit %" PRIu64 " MB free of %" PRIu64 " MB, reserve %" PRIu64
			     " MB; device %" PRIu64 " MB of %" PRIu64 " MB budget)\n",
			     value != 0 ? "short, the caches collect" : "room again", available >> 20u,
			     limit >> 20u, reserve >> 20u, GetDeviceMemoryUsage() >> 20u,
			     GetTotalMemoryBudget() >> 20u);
			// Where the device memory is: per heap, what VMA allocated against what the driver
			// counts (the rest is the driver's own: pipelines, descriptor pools, ...).
			LogMemoryBudget();
		}
	}
	return shortfall.load(std::memory_order_relaxed);
}

void GraphicContext::LogMemoryBudget() const {
	LogCommit("now");
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
	LOGF("Video memory in use (system memory blocks counted there taken off): %" PRIu64 " MB\n",
	     GetDeviceMemoryUsage() >> 20u);
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage        = 0;
	uint64_t local_blocks = 0;
	uint64_t misplaced    = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		const uint64_t blocks = budgets[heap].statistics.blockBytes;
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
			local_blocks += blocks;
		} else if (blocks > budgets[heap].usage) {
			misplaced += blocks - budgets[heap].usage;
		}
	}
	// NVIDIA on Windows reports what this process holds in system memory (the host heap's
	// blocks: the 2.7 GB heap mirror, textures it placed there, staging) under the video memory
	// heap's usage, and the system heap's usage stays below those blocks. Counted as video memory
	// it read 9.4 GB of an 11.5 GB budget with 5 GB of video memory in use, declared the device
	// short and the caches shrank to 512 MB: 23 -> 3-8 fps at that moment in every run. A system
	// heap reporting less usage than its own blocks gives the amount to take off; video memory
	// use never reads below this process's own video memory blocks. KYTY_RAW_DEVICE_USAGE=1 off.
	static const bool raw = std::getenv("KYTY_RAW_DEVICE_USAGE") != nullptr;
	if (discrete && !raw && usage > local_blocks) {
		usage -= std::min(misplaced, usage - local_blocks);
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		const auto own = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		const auto target = VramTargetBytes();
		return target != 0 ? std::min(own, target) : own;
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// Recycling of freed images (KYTY_IMAGE_POOL, default on; =0 off), from the Demon's Souls fork. The
// game aliases transient render targets in one heap and the texture cache frees and re-creates
// images every frame (collector evictions revived the next frame). An image reaches DeleteImage
// only after the GPU is done with it and its views are gone, and a new image starts in the
// undefined layout, so reusing the VkImage and its memory for identical creation parameters is
// equivalent to a fresh allocation. At most budget/32 (512 MB) is kept, nothing while the device
// is short of memory, and the pool is emptied when an allocation fails.
namespace {
struct PooledImage {
	std::array<uint32_t, 10> key {};
	VkImage                  image      = VK_NULL_HANDLE;
	VmaAllocation            allocation = nullptr;
	uint64_t                 bytes      = 0;
	bool                     in_texture_pool = false;
};
std::mutex               g_image_pool_mutex;
std::vector<PooledImage> g_image_pool;
uint64_t                 g_image_pool_bytes = 0;
uint64_t                 g_image_pool_hits  = 0;

bool ImagePoolEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_POOL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

std::array<uint32_t, 10> ImagePoolKey(vk::ImageCreateFlags flags, vk::ImageType type, vk::Format format,
                                      vk::Extent3D extent, uint32_t levels, uint32_t layers,
                                      uint32_t samples, vk::ImageUsageFlags usage) {
	return {static_cast<uint32_t>(static_cast<VkImageCreateFlags>(flags)), static_cast<uint32_t>(type),
	        static_cast<uint32_t>(format), extent.width, extent.height, extent.depth, levels, layers, samples,
	        static_cast<uint32_t>(static_cast<VkImageUsageFlags>(usage))};
}

void TrimImagePoolLocked(VmaAllocator allocator, uint64_t limit) {
	while (g_image_pool_bytes > limit && !g_image_pool.empty()) {
		auto& oldest = g_image_pool.front();
		vmaDestroyImage(allocator, oldest.image, oldest.allocation);
		g_image_pool_bytes -= oldest.bytes;
		g_image_pool.erase(g_image_pool.begin());
	}
}
} // namespace

static void DestroyImagePool(VmaAllocator allocator) {
	std::scoped_lock lock(g_image_pool_mutex);
	TrimImagePoolLocked(allocator, 0);
}

// KYTY_TEXTURE_HOST_MARK_PCT (default 80): the share of the budget in use from which textures
// go to system memory. It was half the budget: on a 12 GB card (10.1 GB budget) every texture
// past 5 GB went to system memory while ~4.5 GB of video memory stayed unused, and an area
// sampling those textures over PCIe took ~350 ms a frame on the GPU (Wolverine, 15 -> 3 fps
// for over a minute, 2026-10-19). 80 % leaves ~2 GB for render targets and the driver; the
// caches start collecting at 90 % (MemoryShortfall).
uint32_t GraphicContext::TextureHostMarkPercent() {
	static const uint32_t pct = [] {
		const char* value = std::getenv("KYTY_TEXTURE_HOST_MARK_PCT");
		const auto  parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {80};
		return static_cast<uint32_t>(std::clamp<uint64_t>(parsed, 10, 100));
	}();
	return pct;
}

// 65 (80 before, 90 before that): under the placement mark (80), so unsampled heap textures leave
// video memory before new ones are sent to system memory. At 80 the probe never ran in a run
// where device use stayed at 55 % because textures were placed in system memory past 50 %.
uint32_t GraphicContext::BindlessEvictPercent() {
	static const uint32_t pct = [] {
		const char* value  = std::getenv("KYTY_BINDLESS_EVICT_PERCENT");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 65ul;
		return static_cast<uint32_t>(std::clamp<unsigned long>(parsed, 10ul, 100ul));
	}();
	return pct;
}

// A texture placed in system memory stays there for its life, however much video memory is
// freed later: after a return to the title menu (2026-10-19) the menu's textures were all
// placed there and it ran at 5 fps instead of 36. Once device use is 15 points under the mark the
// texture cache frees them, a few a frame, and they are created anew here, in video memory.
bool GraphicContext::RoomForHostPlacedImages() const {
	if (g_host_placed_images.load(std::memory_order_relaxed) == 0 || !CanReportMemoryUsage() ||
	    MemoryShortfall() != 0) {
		return false;
	}
	const auto mark   = TextureHostMarkPercent();
	const auto budget = GetTotalMemoryBudget();
	return mark > 15 && budget != 0 && GetDeviceMemoryInUse() < budget / 100 * (mark - 15);
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	if (ImagePoolEnabled() && image_info.pNext == nullptr && image_info.tiling == vk::ImageTiling::eOptimal &&
	    image_info.initialLayout == vk::ImageLayout::eUndefined) {
		const auto key = ImagePoolKey(image_info.flags, image_info.imageType, image_info.format, image_info.extent,
		                              image_info.mipLevels, image_info.arrayLayers,
		                              static_cast<uint32_t>(image_info.samples), image_info.usage);
		std::scoped_lock lock(g_image_pool_mutex);
		for (auto it = g_image_pool.begin(); it != g_image_pool.end(); ++it) {
			// Only a movable image may take memory of the texture pool, whose owners the
			// defragmentation must know.
			if (it->key != key || (it->in_texture_pool && !image.movable)) {
				continue;
			}
			image.image           = it->image;
			image.allocation      = it->allocation;
			image.host_placed     = false; // the pool keeps only video-memory images
			image.in_texture_pool = it->in_texture_pool;
			g_image_pool_bytes -= it->bytes;
			g_image_pool.erase(it);
			if (++g_image_pool_hits % 1024u == 1u) {
				LOGF("Image pool: %" PRIu64 " images reused, %zu kept (%" PRIu64 " MB)\n", g_image_pool_hits,
				     g_image_pool.size(), g_image_pool_bytes >> 20u);
			}
			image.format     = image_info.format;
			image.image_type = image_info.imageType;
			image.extent     = image_info.extent;
			image.layers     = image_info.arrayLayers;
			image.mip_levels = image_info.mipLevels;
			image.samples    = static_cast<uint32_t>(image_info.samples);
			image.usage      = image_info.usage;
			image.flags      = image_info.flags;
			image.state      = {.layout = image_info.initialLayout};
			image.subresource_states.clear();
			return true;
		}
	}

	// The image's memory requirements, for the dedicated and texture pool choices.
	uint64_t required_size = 0;
	uint32_t required_bits = 0;
	if (image_info.pNext == nullptr) {
		vk::DeviceImageMemoryRequirements requirements_info {};
		requirements_info.pCreateInfo = &image_info;
		const auto requirements = device.getImageMemoryRequirements(requirements_info);
		required_size = requirements.memoryRequirements.size;
		required_bits = requirements.memoryRequirements.memoryTypeBits;
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// From the Demon's Souls fork (small_video_memory): on a card whose budget cannot hold the
	// game's working set with the driver's own share (12 GB cards: ~11.5 GB budget, ~3.4 GB of it
	// the driver's), an image that is only sampled and copied -- a texture, not a render target,
	// depth or storage image -- goes to system memory once video memory use passes a mark
	// (TextureHostMarkPercent), so the targets the GPU writes every frame keep video memory.
	// KYTY_TEXTURE_HOST_PLACEMENT=0 keeps every image device-local.
	static const bool host_placement = [] {
		const char* value = std::getenv("KYTY_TEXTURE_HOST_PLACEMENT");
		return value == nullptr || value[0] != '0';
	}();
	// A texture is an image that can be neither a render target nor depth. Storage does not rule it
	// out: block-compressed images carry storage usage here (a texel view for guest writes to
	// them) and are the textures that fill video memory; requiring no storage usage left no image
	// at all to place (no "Textures placed" line in any run).
	const bool block = image_info.format >= vk::Format::eBc1RgbUnormBlock &&
	                   image_info.format <= vk::Format::eBc7SrgbBlock;
	const bool texture =
	    !(image_info.usage & (vk::ImageUsageFlagBits::eColorAttachment |
	                          vk::ImageUsageFlagBits::eDepthStencilAttachment)) &&
	    (block || !(image_info.usage & vk::ImageUsageFlagBits::eStorage));
	bool past_mark = false;
	if (host_placement && texture && CanReportMemoryUsage()) {
		const auto budget = GetTotalMemoryBudget();
		past_mark         = budget < (uint64_t {12} << 30u) &&
		            GetDeviceMemoryUsage() >= budget / 100 * TextureHostMarkPercent();
	}

	// KYTY_IMAGE_DEDICATED_MB (default 16; 0 = off): an image of at least this size gets its own
	// memory allocation instead of a share of a VMA block, so freeing it returns the memory.
	static const uint64_t dedicated_bytes = [] {
		const char* value = std::getenv("KYTY_IMAGE_DEDICATED_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {16}) << 20u;
	}();
	const bool dedicated = dedicated_bytes != 0 && required_size >= dedicated_bytes;
	const bool pooled    = g_texture_pool != nullptr && image.movable && !dedicated &&
	                    required_size != 0 && image_info.samples == vk::SampleCountFlagBits::e1 &&
	                    (required_bits & (1u << g_texture_pool_type)) != 0;

	const auto create = [&](const VmaAllocationCreateInfo& info) {
		vk::Image::CType native_image = VK_NULL_HANDLE;
		const auto       created      = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &info,
            &native_image, &image.allocation, nullptr));
		if (created == vk::Result::eSuccess) {
			image.image           = native_image;
			image.in_texture_pool = info.pool != nullptr && info.pool == g_texture_pool;
		}
		return created;
	};
	auto device_info = alloc_info;
	if (dedicated) {
		device_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
	}
	if (pooled) {
		device_info.pool = g_texture_pool;
	}
	auto result = vk::Result::eErrorOutOfDeviceMemory;
	if (past_mark) {
		// Past the mark a texture still goes to video memory when it fits in memory already
		// allocated (the space freed images left in blocks other images keep alive), without
		// allocating more; otherwise to system memory. Measured by the blocks alone, video memory
		// read 81 % with 2.2 GB of it free inside them, and the textures went to system memory.
		auto existing = device_info;
		existing.flags &= ~static_cast<VmaAllocationCreateFlags>(
		    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
		existing.flags |= VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
		result = create(existing);
		if (result != vk::Result::eSuccess && pooled) {
			existing.pool = nullptr;
			result        = create(existing);
		}
		if (result != vk::Result::eSuccess) {
			auto host          = alloc_info;
			host.requiredFlags = 0;
			host.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			result             = create(host);
			static std::atomic<uint64_t> placed {0};
			if (placed.fetch_add(1, std::memory_order_relaxed) % 1024u == 0u) {
				LOGF("Textures placed by the driver (video or system memory): %" PRIu64 "\n",
				     placed.load(std::memory_order_relaxed));
			}
		}
	} else {
		result = create(device_info);
		if (result != vk::Result::eSuccess && pooled) {
			device_info.pool = nullptr;
			result           = create(device_info);
		}
		if (result != vk::Result::eSuccess && ImagePoolEnabled()) {
			{
				std::scoped_lock lock(g_image_pool_mutex);
				TrimImagePoolLocked(allocator, 0);
			}
			result = create(device_info);
		}
	}
	if (result != vk::Result::eSuccess) {
		// The device heap is full. The guest has a unified 16 GB pool, so the image is
		// real; place it in host memory and pay in bandwidth rather than fail.
		auto host           = alloc_info;
		host.requiredFlags  = 0;
		host.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result              = create(host);
		static std::atomic<uint32_t> spill_count {0};
		if (const auto seen = spill_count.fetch_add(1, std::memory_order_relaxed); seen < 32) {
			LOGF("Image spilled to host memory (%u): %ux%ux%u layers=%u levels=%u format=%d -> %s\n",
			     seen + 1, image_info.extent.width, image_info.extent.height,
			     image_info.extent.depth, image_info.arrayLayers, image_info.mipLevels,
			     static_cast<int>(image_info.format), vk::to_string(result).c_str());
		}
	}
	if (result != vk::Result::eSuccess) {
		image.image      = nullptr;
		image.allocation = nullptr;
		LogMemoryBudget();
		return false;
	}
	{
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		const auto& properties = GetPhysicalDeviceMemoryProperties();
		image.host_placed =
		    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
		    !(properties.memoryTypes[allocation_info.memoryType].propertyFlags &
		      vk::MemoryPropertyFlagBits::eDeviceLocal);
		if (image.host_placed) {
			g_host_placed_bytes.fetch_add(allocation_info.size, std::memory_order_relaxed);
			g_host_placed_images.fetch_add(1, std::memory_order_relaxed);
			g_host_placed_total.fetch_add(1, std::memory_order_relaxed);
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

uint64_t GraphicContext::GetDeviceMemoryInUse() const {
	const auto usage = GetDeviceMemoryUsage();
	if (allocator == nullptr || usage == 0) {
		return usage;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	uint64_t free = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		if (physical_device_memory_properties.memoryHeaps[heap].flags &
		    vk::MemoryHeapFlagBits::eDeviceLocal) {
			const auto& stats = budgets[heap].statistics;
			free += stats.blockBytes > stats.allocationBytes ? stats.blockBytes - stats.allocationBytes
			                                                 : 0;
		}
	}
	return usage - std::min(usage, free);
}

bool GraphicContext::DeviceMemoryInUseAtLeast(uint32_t percent) const {
	if (MemoryShortfall() != 0) {
		return true;
	}
	if (!CanReportMemoryUsage()) {
		return false;
	}
	const auto budget = GetTotalMemoryBudget();
	return budget != 0 && GetDeviceMemoryInUse() / 100 >= budget / 100 * percent / 100;
}

bool GraphicContext::BindImageMemory(vk::Image image, VmaAllocation allocation) const {
	return vmaBindImageMemory(allocator, allocation, image) == VK_SUCCESS;
}

void GraphicContext::SetImageOwner(const VulkanImage& image, uint64_t owner) const {
	if (allocator != nullptr && image.allocation != nullptr && image.in_texture_pool) {
		vmaSetAllocationUserData(allocator, image.allocation,
		                         reinterpret_cast<void*>(static_cast<uintptr_t>(owner)));
	}
}

static void FinishTextureDefrag(VmaAllocator allocator) {
	VmaDefragmentationStats stats {};
	vmaEndDefragmentation(allocator, g_defrag_context, &stats);
	g_defrag_context   = nullptr;
	g_defrag_pass_open = false;
	g_defrag_open      = false;
	g_defrag_next_ms   = NowMs() + 5000;
	g_defrag_moved_bytes += stats.bytesMoved;
	g_defrag_moved_images += stats.allocationsMoved;
	g_defrag_freed_bytes += stats.bytesFreed;
	VmaStatistics pool {};
	vmaGetPoolStatistics(allocator, g_texture_pool, &pool);
	LOGF("Texture defragmentation: %" PRIu64 " MB moved (%u images) in %u passes, %" PRIu64
	     " MB of blocks freed; texture pool now %" PRIu64 " MB of blocks, %" PRIu64
	     " MB allocated\n",
	     static_cast<uint64_t>(stats.bytesMoved >> 20u), stats.allocationsMoved, g_defrag_passes,
	     static_cast<uint64_t>(stats.bytesFreed >> 20u), static_cast<uint64_t>(pool.blockBytes >> 20u),
	     static_cast<uint64_t>(pool.allocationBytes >> 20u));
}

bool GraphicContext::BeginTextureDefragPass(std::vector<TextureDefragMove>& moves) {
	moves.clear();
	if (g_texture_pool == nullptr || allocator == nullptr || g_defrag_pass_open) {
		return false;
	}
	if (g_defrag_context == nullptr) {
		const auto now = NowMs();
		if (now < g_defrag_next_ms) {
			return false;
		}
		g_defrag_next_ms = now + 1000;
		// KYTY_IMAGE_DEFRAG_MB (default 128): free space inside the texture pool's blocks, at least
		// 15 % of them, that starts a defragmentation.
		static const uint64_t min_free = [] {
			const char* value = std::getenv("KYTY_IMAGE_DEFRAG_MB");
			return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {128}) << 20u;
		}();
		VmaStatistics pool {};
		vmaGetPoolStatistics(allocator, g_texture_pool, &pool);
		const uint64_t free =
		    pool.blockBytes > pool.allocationBytes ? pool.blockBytes - pool.allocationBytes : 0;
		if (free < min_free || free * 100 < pool.blockBytes * 15) {
			return false;
		}
		VmaDefragmentationInfo info {};
		info.flags                 = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
		info.pool                  = g_texture_pool;
		info.maxBytesPerPass       = 128ull << 20u;
		info.maxAllocationsPerPass = 128;
		if (vmaBeginDefragmentation(allocator, &info, &g_defrag_context) != VK_SUCCESS) {
			g_defrag_context = nullptr;
			return false;
		}
		g_defrag_open   = true;
		g_defrag_passes = 0;
		{
			// Recycled images of the pool have no owner and could be in a pass's moves; while a
			// defragmentation runs none are kept (DeleteImage destroys them).
			std::scoped_lock lock(g_image_pool_mutex);
			for (auto it = g_image_pool.begin(); it != g_image_pool.end();) {
				if (it->in_texture_pool) {
					vmaDestroyImage(allocator, it->image, it->allocation);
					g_image_pool_bytes -= it->bytes;
					it = g_image_pool.erase(it);
				} else {
					++it;
				}
			}
		}
		LOGF("Texture defragmentation: begins, %" PRIu64 " MB free in %" PRIu64
		     " MB of texture pool blocks\n",
		     free >> 20u, static_cast<uint64_t>(pool.blockBytes >> 20u));
	}
	g_defrag_pass       = {};
	const auto begun    = vmaBeginDefragmentationPass(allocator, g_defrag_context, &g_defrag_pass);
	if (begun == VK_SUCCESS) {
		FinishTextureDefrag(allocator);
		return false;
	}
	if (begun != VK_INCOMPLETE || g_defrag_pass.moveCount == 0 || ++g_defrag_passes > 512) {
		if (begun == VK_INCOMPLETE) {
			for (uint32_t i = 0; i < g_defrag_pass.moveCount; i++) {
				g_defrag_pass.pMoves[i].operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
			}
			if (vmaEndDefragmentationPass(allocator, g_defrag_context, &g_defrag_pass) ==
			        VK_INCOMPLETE &&
			    g_defrag_passes <= 512) {
				return false; // the next frame asks for the next pass
			}
		}
		FinishTextureDefrag(allocator);
		return false;
	}
	g_defrag_pass_open = true;
	moves.resize(g_defrag_pass.moveCount);
	for (uint32_t i = 0; i < g_defrag_pass.moveCount; i++) {
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, g_defrag_pass.pMoves[i].srcAllocation, &info);
		moves[i].destination = g_defrag_pass.pMoves[i].dstTmpAllocation;
		moves[i].owner       = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info.pUserData));
		moves[i].copy        = false;
	}
	return true;
}

void GraphicContext::EndTextureDefragPass(const std::vector<TextureDefragMove>& moves) {
	if (g_defrag_context == nullptr || !g_defrag_pass_open) {
		return;
	}
	for (uint32_t i = 0; i < g_defrag_pass.moveCount; i++) {
		g_defrag_pass.pMoves[i].operation = i < moves.size() && moves[i].copy
		                                        ? VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY
		                                        : VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
	}
	g_defrag_pass_open = false;
	if (vmaEndDefragmentationPass(allocator, g_defrag_context, &g_defrag_pass) != VK_INCOMPLETE) {
		FinishTextureDefrag(allocator);
	}
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (image.host_placed) {
		// Never pooled: a new image of the same shape belongs in video memory if there is room.
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		g_host_placed_bytes.fetch_sub(std::min(allocation_info.size,
		                                       g_host_placed_bytes.load(std::memory_order_relaxed)),
		                              std::memory_order_relaxed);
		g_host_placed_images.fetch_sub(1, std::memory_order_relaxed);
		vmaDestroyImage(allocator, image.image, image.allocation);
		image.image       = nullptr;
		image.allocation  = nullptr;
		image.host_placed = false;
		return;
	}
	const bool in_texture_pool = image.in_texture_pool;
	image.in_texture_pool      = false;
	if (in_texture_pool) {
		// No owner any more: a defragmentation leaves it in place.
		vmaSetAllocationUserData(allocator, image.allocation, nullptr);
	}
	if (ImagePoolEnabled() && !(in_texture_pool && g_defrag_open.load())) {
		const uint64_t limit =
		    MemoryShortfall() != 0 ? 0 : std::min<uint64_t>(512ull << 20u, GetTotalMemoryBudget() / 32);
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		if (allocation_info.size <= limit) {
			std::scoped_lock lock(g_image_pool_mutex);
			g_image_pool.push_back({ImagePoolKey(image.flags, image.image_type, image.format, image.extent,
			                                     image.mip_levels, image.layers, image.samples, image.usage),
			                        image.image, image.allocation, allocation_info.size,
			                        in_texture_pool});
			g_image_pool_bytes += allocation_info.size;
			TrimImagePoolLocked(allocator, limit);
			image.image      = nullptr;
			image.allocation = nullptr;
			return;
		}
		std::scoped_lock lock(g_image_pool_mutex);
		TrimImagePoolLocked(allocator, limit);
	}
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
