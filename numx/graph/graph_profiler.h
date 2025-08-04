#pragma once

#include "graph.h"

namespace nx::graph {
    class GraphProfiler : public std::enable_shared_from_this<GraphProfiler> {
    private:
        void stream_node(std::ostream &stream, OpPtr op);
        bool stream_edge(std::ostream &stream, OpPtr op);

    public:
        GraphProfiler() = default;
        GraphProfiler(const GraphProfiler &) = delete;
        ~GraphProfiler() = default;
        GraphProfiler &operator=(const GraphProfiler &) = delete;
        void print_graph_profile(GraphPtr graph) { stream_graph_profile(graph, std::cout); }
        void save_graph_profile(GraphPtr graph, const std::string &file_name);
        void stream_graph_profile(GraphPtr graph, std::ostream &stream);
    };

    using GraphProfilerPtr = std::shared_ptr<GraphProfiler>;
} // namespace nx::graph