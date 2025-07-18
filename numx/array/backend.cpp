#include "backend.h"

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../runtime/metal/mtl_runner.h"
#endif

namespace nx::array {
    Backend &Backend::get_instance() {
        static Backend instance;
        instance.init();
        return instance;
    }

    void Backend::use_profiler(ProfilerPtr profiler) {
        get_instance().m_profiler = profiler;
    }

    void Backend::init() {
        if (m_device_by_name.size() > 0) {
            // This ensures backend is initialized once
            return;
        }

        // TODO: assume there is a CPU for now
        auto cpu = std::make_shared<Device>(DeviceType::CPU, 0);
        m_device_by_name.emplace("cpu", cpu);
        m_device_by_name.emplace(cpu->get_name(), cpu);

#ifdef __APPLE__
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        // Get all available Metal devices
        NS::Array *mtl_devices = MTL::CopyAllDevices();

        if (!mtl_devices) {
            return;
        }

        MTL::Device *mtl_device;
        DevicePtr device;
        nx::runtime::metal::MTLContextPtr ctx;

#ifdef PROJECT_ROOT
        const std::string project_root = PROJECT_ROOT;
#else
        const std::string project_root = ".";
#endif
        const std::string lib_path = project_root + "/numx/build/runtime/metal/kernels/kernels.metallib";

        for (NS::UInteger i = 0; i < mtl_devices->count(); ++i) {
            mtl_device = mtl_devices->object<MTL::Device>(i);
            device = std::make_shared<Device>(DeviceType::MPS, i);
            m_device_by_name.emplace(device->get_name(), device);
            ctx = std::make_shared<nx::runtime::metal::MTLContext>(mtl_device, lib_path);
            ctx->init_kernels();
            m_ctx_by_device_name.emplace(device->get_name(), ctx);

            m_runner_factory_by_device_name.emplace(device->get_name(), [](GraphPtr graph, RuntimeContextPtr ctx, ProfilerPtr profiler) -> RunnerPtr {
                return std::make_shared<nx::runtime::metal::MTLRunner>(graph, ctx, profiler);
            });

            m_graph_factory_by_device_name.emplace(device->get_name(), [](OpPtr op) -> GraphPtr {
                return std::make_shared<nx::graph::Graph>(op);
            });

            std::println("Initialized device {}...", *device);
        }

        pool->release();
#endif
    }

    DevicePtr Backend::get_device_by_name(const std::string &name) const {
        if (m_device_by_name.find(name) == m_device_by_name.end()) {
            throw std::invalid_argument(std::format("No device named {}.", name));
        }

        return m_device_by_name.at(name);
    }

    RuntimeContextPtr Backend::get_context_by_device_name(const std::string &device_name) const {
        if (m_ctx_by_device_name.find(device_name) == m_ctx_by_device_name.end()) {
            throw std::invalid_argument(std::format("No context found for device named {}.", device_name));
        }

        return m_ctx_by_device_name.at(device_name);
    }

    RunnerFactory Backend::get_runner_factory_by_device_name(const std::string &device_name) const {
        if (m_runner_factory_by_device_name.find(device_name) == m_runner_factory_by_device_name.end()) {
            throw std::invalid_argument(std::format("No runner factory found for device named {}.", device_name));
        }

        return m_runner_factory_by_device_name.at(device_name);
    }

    GraphFactory Backend::get_graph_factory_by_device_name(const std::string &device_name) const {
        if (m_graph_factory_by_device_name.find(device_name) == m_graph_factory_by_device_name.end()) {
            throw std::invalid_argument(std::format("No graph factory found for device named {}.", device_name));
        }

        return m_graph_factory_by_device_name.at(device_name);
    }
} // namespace nx::array