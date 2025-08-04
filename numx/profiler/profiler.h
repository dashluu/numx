#pragma once

#include "../core/array.h"
#include "../graph/graph_profiler.h"

namespace nx::profiler {
    using namespace nx::graph;
    using namespace nx::runtime;
    using namespace nx::core;

    inline MemoryProfilerPtr get_memory_profiler(const std::string &device_name) { return nx::core::get_device_context(device_name)->get_runtime_context()->get_memory_profiler(); }
    void enable_memory_profile();
    void disable_memory_profile();
    void stream_memory_profile(std::ostream &stream);
    void stream_graph_profile(GraphPtr graph, std::ostream &stream);
    void save_memory_profile(const std::string &file_name);
    void save_graph_profile(GraphPtr graph, const std::string &file_name);
    inline void print_memory_profile() { stream_memory_profile(std::cout); }
    inline void print_graph_profile(GraphPtr graph) { stream_graph_profile(graph, std::cout); }
} // namespace nx::profiler