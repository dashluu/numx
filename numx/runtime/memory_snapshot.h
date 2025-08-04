#pragma once

#include "../graph/graph.h"

namespace nx::runtime {
    using namespace nx::primitive;
    using namespace nx::graph;

    struct MemorySnapshot {
    private:
        BufferBlock *m_block;
        std::chrono::system_clock::time_point m_alloc_time;
        std::chrono::system_clock::time_point m_free_time;
        bool m_alive = true;

    public:
        MemorySnapshot(BufferBlock *block) : m_block(block) { m_alloc_time = std::chrono::system_clock::now(); }

        MemorySnapshot(const MemorySnapshot &snapshot) {
            m_block = snapshot.m_block;
            m_alloc_time = snapshot.m_alloc_time;
            m_free_time = snapshot.m_free_time;
            m_alive = snapshot.m_alive;
        }

        ~MemorySnapshot() = default;

        MemorySnapshot &operator=(const MemorySnapshot &snapshot) {
            m_block = snapshot.m_block;
            m_alloc_time = snapshot.m_alloc_time;
            m_free_time = snapshot.m_free_time;
            m_alive = snapshot.m_alive;
            return *this;
        }

        BufferBlock *get_block() const { return m_block; }
        const std::chrono::system_clock::time_point &get_alloc_time() const { return m_alloc_time; }
        const std::chrono::system_clock::time_point &get_free_time() const { return m_free_time; }
        bool is_alive() const { return m_alive; }
        auto elapsed() const { return std::chrono::duration_cast<std::chrono::microseconds>(m_free_time - m_alloc_time).count(); }

        void stop() {
            m_free_time = std::chrono::system_clock::now();
            m_alive = false;
        }
    };
} // namespace nx::runtime