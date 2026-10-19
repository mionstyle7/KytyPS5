#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MESHARGSTASK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MESHARGSTASK_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// A task shader, the same for every mesh pipeline, that launches the mesh workgroups of a draw
// whose counts the GPU wrote (DRAW_INDIRECT): it reads the vertex and instance counts at the host
// address in push dwords 0-1 and emits (vertex count, instance count, 1) groups. Before it, each
// such draw ended the render pass to copy the counts into an indirect entry: ~350 of the ~560
// render pass restarts of a gameplay frame (2026-10-19). Assembled from (spirv-as, vulkan1.3):
//   OpEntryPoint TaskEXT "main", push block {uint lo; uint hi}, LocalSize 1 1 1;
//   addr = bitcast<u64>(uvec2(lo, hi)); x = *(uint*)addr; y = *(uint*)(addr + 4);
//   OpEmitMeshTasksEXT x y 1
// Created once; null when the device has no task shaders. KYTY_MESH_TASK_ARGS=0 turns it off.
[[nodiscard]] vk::ShaderModule MeshArgsTaskModule(GraphicContext& graphics);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MESHARGSTASK_H_ */
