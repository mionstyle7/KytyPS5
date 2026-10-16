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
#include <vector>
#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics {

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
			if (it->key != key) {
				continue;
			}
			image.image      = it->image;
			image.allocation = it->allocation;
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

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// From the Demon's Souls fork (small_video_memory): on a card whose budget cannot hold the
	// game's working set with the driver's own share (12 GB cards: ~11.5 GB budget, ~3.4 GB of it
	// the driver's), an image that is only sampled and copied -- a texture, not a render target,
	// depth or storage image -- goes to the memory type without flags once half the budget is in
	// use. The driver keeps it in video memory while there is room and in system memory after, so
	// the targets the GPU writes every frame keep video memory, and running short no longer makes
	// the caches evict and recreate the working set (the frame rate fell from 20 to 3-7 fps at
	// that point in every run). KYTY_TEXTURE_HOST_PLACEMENT=0 keeps every image device-local.
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
	if (host_placement && texture && CanReportMemoryUsage()) {
		const auto budget = GetTotalMemoryBudget();
		if (budget < (uint64_t {12} << 30u) && GetDeviceMemoryUsage() >= budget / 2) {
			alloc_info.requiredFlags = 0;
			alloc_info.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			static std::atomic<uint64_t> placed {0};
			if (placed.fetch_add(1, std::memory_order_relaxed) % 1024u == 0u) {
				LOGF("Textures placed by the driver (video or system memory): %" PRIu64 "\n",
				     placed.load(std::memory_order_relaxed));
			}
		}
	}

	vk::Image::CType native_image = VK_NULL_HANDLE;
	auto             result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	if (result != vk::Result::eSuccess && ImagePoolEnabled()) {
		{
			std::scoped_lock lock(g_image_pool_mutex);
			TrimImagePoolLocked(allocator, 0);
		}
		result = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
	}
	if (result != vk::Result::eSuccess) {
		// The device heap is full. The guest has a unified 16 GB pool, so the image is
		// real; place it in host memory and pay in bandwidth rather than fail.
		alloc_info.requiredFlags  = 0;
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result                    = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
            &alloc_info, &native_image, &image.allocation, nullptr));
		static std::atomic<uint32_t> spill_count {0};
		if (const auto seen = spill_count.fetch_add(1, std::memory_order_relaxed); seen < 32) {
			LOGF("Image spilled to host memory (%u): %ux%ux%u layers=%u levels=%u format=%d -> %s\n",
			     seen + 1, image_info.extent.width, image_info.extent.height,
			     image_info.extent.depth, image_info.arrayLayers, image_info.mipLevels,
			     static_cast<int>(image_info.format), vk::to_string(result).c_str());
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
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

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (ImagePoolEnabled()) {
		const uint64_t limit =
		    MemoryShortfall() != 0 ? 0 : std::min<uint64_t>(512ull << 20u, GetTotalMemoryBudget() / 32);
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		if (allocation_info.size <= limit) {
			std::scoped_lock lock(g_image_pool_mutex);
			g_image_pool.push_back({ImagePoolKey(image.flags, image.image_type, image.format, image.extent,
			                                     image.mip_levels, image.layers, image.samples, image.usage),
			                        image.image, image.allocation, allocation_info.size});
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
