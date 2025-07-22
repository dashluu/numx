#pragma once

#include "../graph/graph.h"
#include "../memory/block.h"

namespace nx::instrument {
    using namespace nx::primitive;
    using namespace nx::graph;
    using namespace nx::memory;

    struct MemorySnapshotInfo {
    private:
        Block m_block;
        std::chrono::system_clock::time_point m_alloc_time;
        std::chrono::system_clock::time_point m_free_time;
        bool m_alive = true;

    public:
        MemorySnapshotInfo(const Block &block) : m_block(block) { m_alloc_time = std::chrono::system_clock::now(); }

        MemorySnapshotInfo(const MemorySnapshotInfo &snapshot_info) {
            m_block = snapshot_info.m_block;
            m_alloc_time = snapshot_info.m_alloc_time;
            m_free_time = snapshot_info.m_free_time;
            m_alive = snapshot_info.m_alive;
        }

        ~MemorySnapshotInfo() = default;

        MemorySnapshotInfo &operator=(const MemorySnapshotInfo &snapshot_info) {
            m_block = snapshot_info.m_block;
            m_alloc_time = snapshot_info.m_alloc_time;
            m_free_time = snapshot_info.m_free_time;
            m_alive = snapshot_info.m_alive;
            return *this;
        }

        const Block &get_block() const { return m_block; }
        const std::chrono::system_clock::time_point &get_alloc_time() const { return m_alloc_time; }
        const std::chrono::system_clock::time_point &get_free_time() const { return m_free_time; }
        bool is_alive() const { return m_alive; }
        auto elapsed() const { return std::chrono::duration_cast<std::chrono::microseconds>(m_free_time - m_alloc_time).count(); }

        void tock() {
            m_free_time = std::chrono::system_clock::now();
            m_alive = false;
        }
    };
} // namespace nx::instrument