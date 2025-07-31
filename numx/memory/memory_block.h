#pragma once

#include "resource_list.h"

namespace nx::memory {
    struct MemoryPool;

    struct MemoryBlock : public ResourceNode {
    private:
        uint8_t *m_ptr = nullptr;
        isize m_size = 0;
        MemoryPool *m_pool = nullptr;

    public:
        MemoryBlock() = default;
        MemoryBlock(uint8_t *ptr, isize size, MemoryPool *pool) : m_ptr(ptr), m_size(size), m_pool(pool) {}
        MemoryBlock(uint8_t *ptr, isize size) : m_ptr(ptr), m_size(size) {}
        ~MemoryBlock() = default;
        uint8_t *get_ptr() const { return m_ptr; }
        isize get_size() const { return m_size; }
        MemoryPool *get_pool() const { return m_pool; }
        bool is_valid() const { return m_ptr && m_size > 0 && m_pool; }
    };
} // namespace nx::memory