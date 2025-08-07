#pragma once

#include "resource_list.h"

namespace nx::primitive {
    struct BufferPool;

    struct BufferBlock : public Resource {
    private:
        uint8_t *m_ptr = nullptr;
        isize m_size = 0;
        BufferPool *m_pool = nullptr;

    public:
        BufferBlock() = default;
        BufferBlock(uint8_t *ptr, isize size, BufferPool *pool) : m_ptr(ptr), m_size(size), m_pool(pool) {}
        BufferBlock(uint8_t *ptr, isize size) : m_ptr(ptr), m_size(size) {}
        ~BufferBlock() = default;
        uint8_t *get_ptr() const { return m_ptr; }
        isize get_size() const { return m_size; }
        BufferPool *get_pool() const { return m_pool; }
    };
} // namespace nx::primitive