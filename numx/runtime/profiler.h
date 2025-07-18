#pragma once

#include "../graph/graph.h"

namespace nx::runtime {
    using namespace nx::core;
    using namespace nx::graph;

    class Profiler : public std::enable_shared_from_this<Profiler> {
    private:
        std::set<ArrayData> m_arrays;

        void log_node(OpPtr op, std::ostream &stream);
        bool log_edge(OpPtr op, std::ostream &stream);

    public:
        Profiler() {}
        Profiler(const Profiler &) = delete;
        ~Profiler() = default;
        Profiler &operator=(const Profiler &) = delete;
        void record_alloc(const ArrayData &data) { m_arrays.emplace(data); }
        void record_free(const ArrayData &data) { m_arrays.erase(data); }
        bool has_memory_leaks() const { return !m_arrays.empty(); }
        void log_memory(std::ostream &stream);
        void log_graph(GraphPtr graph, std::ostream &stream);
    };

    using ProfilerPtr = std::shared_ptr<Profiler>;
} // namespace nx::runtime