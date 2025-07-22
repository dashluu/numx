#pragma once

#include "../instrument/profiler.h"
#include "../runtime/runner.h"
#include "../runtime/runtime_context.h"

namespace nx::core {
    using namespace nx::utils;
    using namespace nx::primitive;
    using namespace nx::graph;
    using namespace nx::runtime;
    using namespace nx::instrument;
    using RunnerFactory = std::function<RunnerPtr(GraphPtr, RuntimeContextPtr, ProfilerPtr)>;
    using GraphFactory = std::function<GraphPtr(OpPtr)>;

    class Backend {
    private:
        ProfilerPtr m_profiler = nullptr;
        std::unordered_map<std::string, DevicePtr> m_device_by_name;
        std::unordered_map<std::string, RuntimeContextPtr> m_ctx_by_device_name;
        std::unordered_map<std::string, RunnerFactory> m_runner_factory_by_device_name;
        std::unordered_map<std::string, GraphFactory> m_graph_factory_by_device_name;

        void init();

    public:
        Backend() = default;
        Backend(const Backend &) = delete;
        ~Backend() = default;
        Backend &operator=(const Backend &) = delete;
        ProfilerPtr get_memory_profiler() const { return m_profiler; }
        DevicePtr get_device_by_name(const std::string &name) const;
        RuntimeContextPtr get_context_by_device_name(const std::string &device_name) const;
        RunnerFactory get_runner_factory_by_device_name(const std::string &device_name) const;
        GraphFactory get_graph_factory_by_device_name(const std::string &device_name) const;
        size_t count_devices() const { return m_device_by_name.size(); }
        static Backend &get_instance();
        static void use_profiler(ProfilerPtr profiler);
    };
} // namespace nx::core