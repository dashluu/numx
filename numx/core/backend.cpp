#include "backend.h"

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../runtime/metal/mtl_runner.h"
#endif

namespace nx::core {
    Backend &Backend::get_instance() {
        static Backend instance;
        instance.init();
        return instance;
    }

    void Backend::init() {
        if (count_devices() > 0) {
            // This ensures backend is initialized once
            return;
        }

#ifdef __APPLE__
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        // Get all available Metal devices
        NS::Array *mtl_devices = MTL::CopyAllDevices();

        if (!mtl_devices) {
            return;
        }

        MTL::Device *mtl_device;
        DevicePtr device;
        nx::runtime::metal::MTLContextPtr runtime_ctx;
        MemoryProfilerPtr memory_profiler;

#ifdef PROJECT_ROOT
        const std::string project_root = PROJECT_ROOT;
#else
        const std::string project_root = ".";
#endif
        const std::string lib_path = project_root + "/numx/build/runtime/metal/kernels/kernels.metallib";

        for (NS::UInteger i = 0; i < mtl_devices->count(); ++i) {
            mtl_device = mtl_devices->object<MTL::Device>(i);
            device = std::make_shared<Device>(DeviceType::MPS, i);
            memory_profiler = std::make_shared<MemoryProfiler>(device);
            runtime_ctx = std::make_shared<nx::runtime::metal::MTLContext>(mtl_device, lib_path, memory_profiler);
            runtime_ctx->init_kernels();

            auto runner_builder = [](GraphPtr graph, RuntimeContextPtr runtime_ctx) -> RunnerPtr {
                return std::make_shared<nx::runtime::metal::MTLRunner>(graph, runtime_ctx);
            };

            auto graph_builder = [](OpPtr op) -> GraphPtr {
                return std::make_shared<nx::graph::Graph>(op);
            };

            auto rand_key_gen = std::make_shared<RandomKeyGenerator>(seed());
            auto device_ctx = std::make_shared<DeviceContext>(device, runtime_ctx, runner_builder, graph_builder, rand_key_gen);
            m_device_ctx_by_name.emplace(device->get_name(), device_ctx);
            std::println("Initialized device {}...", *device);
        }

        pool->release();
#endif
    }

    DeviceContextPtr Backend::get_device_context(const std::string &device_name) const {
        if (m_device_ctx_by_name.find(device_name) == m_device_ctx_by_name.end()) {
            throw std::invalid_argument(std::format("No context found for device named {}.", device_name));
        }

        return m_device_ctx_by_name.at(device_name);
    }
} // namespace nx::core