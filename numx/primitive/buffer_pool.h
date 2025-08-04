#pragma once

#include "../allocator/allocator.h"
#include "buffer_block.h"

namespace nx::primitive {
    using namespace nx::allocator;

    struct BufferPool : public Resource {
    private:
        uint8_t *m_pool;
        isize m_capacity;
        isize m_block_size;
        ResourceList m_free_blocks;
        AllocatorPtr m_allocator;

    public:
        BufferPool(isize capacity, isize block_size, AllocatorPtr allocator) : m_capacity(capacity), m_block_size(block_size), m_allocator(allocator) {
            m_pool = m_allocator->alloc_bytes(capacity);
            std::memset(m_pool, 0, capacity);

            for (isize i = 0; i < capacity; i += block_size) {
                BufferBlock *block = new BufferBlock(m_pool + i, block_size, this);
                m_free_blocks.push(block);
            }
        }

        ~BufferPool() {
            for (Resource *resource : m_free_blocks) {
                delete static_cast<BufferBlock *>(resource);
            }

            m_allocator->free_bytes(m_pool);
        }

        isize get_capacity() const { return m_capacity; }
        isize get_block_size() const { return m_block_size; }
        bool empty() const { return m_free_blocks.empty(); }
        BufferBlock *alloc_block() { return static_cast<BufferBlock *>(m_free_blocks.pop()); }
        void free_block(BufferBlock *block) { m_free_blocks.push(block); }
    };
} // namespace nx::primitive