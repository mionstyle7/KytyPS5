#include "common/assert.h"
#include "common/logging/log.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()), m_scheduler(scheduler) {}

bool CommandBuffer::IsInvalid() const {
	return !m_open;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	if (m_scheduler.Threaded()) {
		// Recording directly: the recording thread must have run everything queued first.
		return m_scheduler.DrainRecording();
	}
	return m_buffer;
}

CommandRecorder CommandBuffer::Recorder() const {
	EXIT_IF(IsInvalid());
	return m_scheduler.Threaded() ? CommandRecorder(m_scheduler, nullptr) : CommandRecorder(m_buffer);
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	if (m_scheduler.Threaded()) {
		// The recording thread ends it before submitting (CommandScheduler::Submit).
		return;
	}
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4,
                                 uint64_t arg5) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	if (m_graphics.diagnostic_checkpoints_enabled && !IsInvalid()) {
		const auto* marker = RecordDiagnosticCheckpoint({.op        = op,
		                                                 .submit_id = submit_id,
		                                                 .arg0      = arg0,
		                                                 .arg1      = arg1,
		                                                 .arg2      = arg2,
		                                                 .arg3      = arg3,
		                                                 .arg4      = arg4,
		                                                 .arg5      = arg5});
		if (marker != nullptr) {
			Recorder().setCheckpointNV(marker);
		}
	}
}

// Diagnostics: render passes really started, and which call sites ended them, every 10 s. A
// pass ended between draws of the same targets is begun again, and every restart stores and
// loads the attachments ("Render areas" counts draws per target size, not passes).
namespace {
struct RenderPassStats {
	std::mutex                                                   mutex;
	uint64_t                                                     started = 0;
	std::map<std::pair<const char*, uint32_t>, uint64_t>         ended;
	std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now() + std::chrono::seconds(10);
};
RenderPassStats g_render_pass_stats;

void LogRenderPassStatsLocked(RenderPassStats& stats) {
	const auto now = std::chrono::steady_clock::now();
	if (now < stats.next) {
		return;
	}
	stats.next = now + std::chrono::seconds(10);
	std::vector<std::pair<std::pair<const char*, uint32_t>, uint64_t>> top(stats.ended.begin(),
	                                                                       stats.ended.end());
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	uint64_t ended = 0;
	for (const auto& entry: top) {
		ended += entry.second;
	}
	std::string text;
	for (size_t i = 0; i < std::min<size_t>(top.size(), 12); i++) {
		const char* file  = top[i].first.first;
		const char* slash = file;
		for (const char* c = file; *c != 0; c++) {
			if (*c == '/' || *c == '\\') {
				slash = c + 1;
			}
		}
		text += fmt::format(" {}:{}={}", slash, top[i].first.second, top[i].second);
	}
	LOGF("Render passes (10 s): %" PRIu64 " started, %" PRIu64 " ended by:%s\n", stats.started, ended,
	     text.c_str());
	stats.started = 0;
	stats.ended.clear();
}
} // namespace

void CommandBuffer::NoteRenderPassStart() {
	std::lock_guard lock(g_render_pass_stats.mutex);
	g_render_pass_stats.started++;
	LogRenderPassStatsLocked(g_render_pass_stats);
}

void CommandBuffer::NoteRenderPassEnd(std::source_location loc) {
	std::lock_guard lock(g_render_pass_stats.mutex);
	g_render_pass_stats.ended[{loc.file_name(), loc.line()}]++;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();
	NoteRenderPassStart();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Recorder().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering(std::source_location loc) const {
	if (!m_rendering) {
		return;
	}
	NoteRenderPassEnd(loc);
	Recorder().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
