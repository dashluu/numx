#pragma once

#include "../runtime/runner.h"
#include "../runtime/runtime_context.h"

namespace nx::array {
    using namespace nx::utils;
    using namespace nx::core;
    using namespace nx::graph;
    using namespace nx::runtime;
    using RunnerFactory = std::function<RunnerPtr(GraphPtr, RuntimeContextPtr)>;
    using GraphFactory = std::function<GraphPtr(OpPtr)>;

    class Backend {
    private:
        std::unordered_map<std::string, DevicePtr> m_device_by_name;
        std::unordered_map<std::string, RuntimeContextPtr> m_ctx_by_device_name;
        std::unordered_map<std::string, RunnerFactory> m_runner_factory_by_device_name;
        std::unordered_map<std::string, GraphFactory> m_graph_factory_by_device_name;

    public:
        Backend() = default;
        Backend(const Backend &) = delete;
        Backend &operator=(const Backend &) = delete;
        DevicePtr get_device_by_name(const std::string &name) const;
        RuntimeContextPtr get_context_by_device_name(const std::string &device_name) const;
        RunnerFactory get_runner_factory_by_device_name(const std::string &device_name) const;
        GraphFactory get_graph_factory_by_device_name(const std::string &device_name) const;
        size_t count_devices() const { return m_device_by_name.size(); }
        void init();
    };
} // namespace nx::array