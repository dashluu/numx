#pragma once

#include "../core/array.h"
#include "../graph/graph_profiler.h"

namespace nx::profiler {
    using namespace nx::graph;
    using namespace nx::runtime;
    using namespace nx::core;

    void enable_memory_profile();
    void disable_memory_profile();
    void stream_memory_profile(std::ostream &stream);
    void stream_graph_profile(const Array &array, std::ostream &stream);
    void save_memory_profile(const std::string &file_name);
    void save_graph_profile(const Array &array, const std::string &file_name);
    inline void print_memory_profile() { stream_memory_profile(std::cout); }
    inline void print_graph_profile(const Array &array) { stream_graph_profile(array, std::cout); }
    inline MemoryProfilerPtr get_memory_profiler(const std::string &device_name) { return nx::core::get_device_context(device_name)->get_runtime_context()->get_memory_profiler(); }
    inline void enable_device_memory_profile(const std::string &device_name) { get_memory_profiler(device_name)->enable(); }
    inline void disable_device_memory_profile(const std::string &device_name) { get_memory_profiler(device_name)->disable(); }
    inline isize get_device_peak_memory(const std::string &device_name) { return get_memory_profiler(device_name)->get_peak_memory(); }
    inline isize get_device_pool_memory(const std::string &device_name) { return get_memory_profiler(device_name)->get_pool_memory(); }
    inline bool stream_device_memory_profile(const std::string &device_name, std::ostream &stream) { return get_memory_profiler(device_name)->stream_profile(stream); }
    inline bool save_device_memory_profile(const std::string &device_name, const std::string &file_name) { return get_memory_profiler(device_name)->save_profile(file_name); }
    inline bool print_device_memory_profile(const std::string &device_name) { return get_memory_profiler(device_name)->print_profile(); }
} // namespace nx::profiler