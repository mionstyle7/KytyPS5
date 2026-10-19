#include "graphics/host_gpu/renderer/pipeline/meshArgsTask.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineWarmup.h"

#include <array>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics {

namespace {
constexpr std::array<uint32_t, 161> MeshArgsTaskSpirv {
0x07230203, 0x00010600, 0x00070000, 0x0000001b, 0x00000000, 0x00020011,
0x00000001, 0x00020011, 0x0000000b, 0x00020011, 0x000014a3, 0x00020011,
0x000014e3, 0x0006000a, 0x5f565053, 0x5f545845, 0x6873656d, 0x6168735f,
0x00726564, 0x0003000e, 0x000014e4, 0x00000001, 0x0006000f, 0x000014f4,
0x00000001, 0x6e69616d, 0x00000000, 0x00000002, 0x00060010, 0x00000001,
0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030047, 0x00000003,
0x00000002, 0x00050048, 0x00000003, 0x00000000, 0x00000023, 0x00000000,
0x00050048, 0x00000003, 0x00000001, 0x00000023, 0x00000004, 0x00020013,
0x00000004, 0x00030021, 0x00000005, 0x00000004, 0x00040015, 0x00000006,
0x00000020, 0x00000000, 0x00040015, 0x00000007, 0x00000040, 0x00000000,
0x00040017, 0x00000008, 0x00000006, 0x00000002, 0x0004001e, 0x00000003,
0x00000006, 0x00000006, 0x00040020, 0x00000009, 0x00000009, 0x00000003,
0x00040020, 0x0000000a, 0x00000009, 0x00000006, 0x00040020, 0x0000000b,
0x000014e5, 0x00000006, 0x0004003b, 0x00000009, 0x00000002, 0x00000009,
0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x0004002b, 0x00000006,
0x0000000d, 0x00000001, 0x0005002b, 0x00000007, 0x0000000e, 0x00000004,
0x00000000, 0x00050036, 0x00000004, 0x00000001, 0x00000000, 0x00000005,
0x000200f8, 0x0000000f, 0x00050041, 0x0000000a, 0x00000010, 0x00000002,
0x0000000c, 0x0004003d, 0x00000006, 0x00000011, 0x00000010, 0x00050041,
0x0000000a, 0x00000012, 0x00000002, 0x0000000d, 0x0004003d, 0x00000006,
0x00000013, 0x00000012, 0x00050050, 0x00000008, 0x00000014, 0x00000011,
0x00000013, 0x0004007c, 0x00000007, 0x00000015, 0x00000014, 0x00040078,
0x0000000b, 0x00000016, 0x00000015, 0x0006003d, 0x00000006, 0x00000017,
0x00000016, 0x00000002, 0x00000004, 0x00050080, 0x00000007, 0x00000018,
0x00000015, 0x0000000e, 0x00040078, 0x0000000b, 0x00000019, 0x00000018,
0x0006003d, 0x00000006, 0x0000001a, 0x00000019, 0x00000002, 0x00000004,
0x000414ae, 0x00000017, 0x0000001a, 0x0000000d, 0x00010038

};
} // namespace

vk::ShaderModule MeshArgsTaskModule(GraphicContext& graphics) {
	static std::once_flag   once;
	static vk::ShaderModule module = nullptr;
	std::call_once(once, [&graphics] {
		const char* value = std::getenv("KYTY_MESH_TASK_ARGS");
		if (!graphics.task_shader_enabled || (value != nullptr && value[0] == '0')) {
			LOGF("Mesh draws with GPU counts: indirect entry copies (task shaders %s)\n",
			     graphics.task_shader_enabled ? "off" : "unsupported");
			return;
		}
		vk::ShaderModuleCreateInfo info {};
		info.codeSize = MeshArgsTaskSpirv.size() * sizeof(uint32_t);
		info.pCode    = MeshArgsTaskSpirv.data();
		if (graphics.device.createShaderModule(&info, nullptr, &module) != vk::Result::eSuccess) {
			module = nullptr;
			LOGF("Mesh draws with GPU counts: task shader module failed; indirect entry copies\n");
			return;
		}
		PipelineWarmup::RegisterModule(module, MeshArgsTaskSpirv);
		LOGF("Mesh draws with GPU counts: launched by a task shader, inside the render pass\n");
	});
	return module;
}

} // namespace Libs::Graphics
