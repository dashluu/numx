#pragma once

#include "../primitive/random.h"
#include "../runtime/runner.h"
#include "../runtime/runtime_context.h"

namespace nx::core {
    using namespace nx::utils;
    using namespace nx::primitive;
    using namespace nx::graph;
    using namespace nx::runtime;
    using RunnerBuilder = std::function<RunnerPtr(GraphPtr, RuntimeContextPtr)>;
    using GraphBuilder = std::function<GraphPtr(OpPtr)>;

    struct DeviceContext : public std::enable_shared_from_this<DeviceContext> {
    private:
        DevicePtr m_device;
        RuntimeContextPtr m_runtime_ctx;
        RandomKeyGeneratorPtr m_rand_key_gen;
        RunnerBuilder m_runner_builder;
        GraphBuilder m_graph_builder;

    public:
        DeviceContext(DevicePtr device, RuntimeContextPtr runtime_ctx, RunnerBuilder runner_builder, GraphBuilder graph_builder, RandomKeyGeneratorPtr rand_key_gen) : m_device(device), m_runtime_ctx(runtime_ctx), m_runner_builder(runner_builder), m_graph_builder(graph_builder), m_rand_key_gen(rand_key_gen) {}
        DeviceContext(const DeviceContext &) = delete;
        DeviceContext(DeviceContext &&) noexcept = delete;
        ~DeviceContext() = default;
        DeviceContext &operator=(const DeviceContext &) = delete;
        DeviceContext &operator=(DeviceContext &&) noexcept = delete;
        DevicePtr get_device() const { return m_device; }
        RuntimeContextPtr get_runtime_context() const { return m_runtime_ctx; }
        RandomKeyGeneratorPtr get_random_key_generator() const { return m_rand_key_gen; }
        RunnerBuilder get_runner_builder() const { return m_runner_builder; }
        GraphBuilder get_graph_builder() const { return m_graph_builder; }
    };

    using DeviceContextPtr = std::shared_ptr<DeviceContext>;

    class Backend {
    private:
        std::unordered_map<std::string, DeviceContextPtr> m_device_ctx_by_name;

        void init();

    public:
        Backend() = default;
        Backend(const Backend &) = delete;
        Backend(Backend &&) noexcept = delete;
        ~Backend() = default;
        Backend &operator=(const Backend &) = delete;
        Backend &operator=(Backend &&) noexcept = delete;
        size_t count_devices() const { return m_device_ctx_by_name.size(); }
        DeviceContextPtr get_device_context(const std::string &device_name) const;
        static Backend &get_instance();
        std::unordered_map<std::string, DeviceContextPtr>::const_iterator begin() const { return m_device_ctx_by_name.cbegin(); }
        std::unordered_map<std::string, DeviceContextPtr>::const_iterator end() const { return m_device_ctx_by_name.cend(); }
    };
} // namespace nx::core