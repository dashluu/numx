#pragma once

#include "memory_snapshot_info.h"

namespace nx::instrument {
    class Profiler : public std::enable_shared_from_this<Profiler> {
    private:
        size_t peak_memory = 0;
        size_t memory_usage = 0;
        std::unordered_map<ArrayId, MemorySnapshotInfo> m_memory_snapshot;

        void stream_node(std::ostream &stream, OpPtr op);
        bool stream_edge(std::ostream &stream, OpPtr op);
        void stream_memory_snapshot_info(std::ostream &stream, const MemorySnapshotInfo &snapshot_info);

    public:
        Profiler() {}
        Profiler(const Profiler &) = delete;
        ~Profiler() = default;
        Profiler &operator=(const Profiler &) = delete;
        size_t get_peak_memory() const { return peak_memory; }
        void record_alloc(const ArrayData &data);
        void record_free(const ArrayData &data);
        void print_memory_profile() { stream_memory_profile(std::cout); }
        void print_graph_profile(GraphPtr graph) { stream_graph_profile(graph, std::cout); }
        void write_memory_profile(const std::string &file_name);
        void write_graph_profile(GraphPtr graph, const std::string &file_name);
        void stream_memory_profile(std::ostream &stream);
        void stream_graph_profile(GraphPtr graph, std::ostream &stream);
    };

    using ProfilerPtr = std::shared_ptr<Profiler>;
} // namespace nx::instrument