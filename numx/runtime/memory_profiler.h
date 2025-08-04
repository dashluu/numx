#pragma once

#include "memory_snapshot.h"

namespace nx::runtime {
    class MemoryProfiler : public std::enable_shared_from_this<MemoryProfiler> {
    private:
        DevicePtr m_device;
        isize m_peak_memory = 0;
        isize m_block_memory = 0;
        isize m_pool_memory = 0;
        std::unordered_map<ArrayId, MemorySnapshot> m_snapshot_by_id;
        bool m_enabled = false;

        void stream_memory_snapshot(std::ostream &stream, const MemorySnapshot &snapshot);

    public:
        MemoryProfiler(DevicePtr device) : m_device(device) {}
        MemoryProfiler(const MemoryProfiler &) = delete;
        ~MemoryProfiler() = default;
        MemoryProfiler &operator=(const MemoryProfiler &) = delete;
        isize get_peak_memory() const { return m_peak_memory; }
        isize get_pool_memory() const { return m_pool_memory; }
        bool is_enabled() const { return m_enabled; }
        void enable() { m_enabled = true; }
        void disable() { m_enabled = false; }
        void trace_alloc_block(const ArrayData &data);
        void trace_free_block(const ArrayData &data);
        void trace_alloc_pool(isize capacity);
        void trace_free_pool(isize capacity);
        void print_memory_profile() { stream_memory_profile(std::cout); }
        void save_memory_profile(const std::string &file_name);
        void stream_memory_profile(std::ostream &stream);
    };

    using MemoryProfilerPtr = std::shared_ptr<MemoryProfiler>;
} // namespace nx::runtime