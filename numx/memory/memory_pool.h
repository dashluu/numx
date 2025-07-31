#pragma once

#include "memory_block.h"
#include "runtime_allocator.h"

namespace nx::memory {
    class CacheManager;

    struct MemoryPool : public ResourceNode {
    private:
        uint8_t *m_pool;
        isize m_capacity;
        isize m_block_size;
        ResourceList m_free_blocks;
        RuntimeAllocatorPtr m_runtime_allocator;

    public:
        friend class CacheManager;

        MemoryPool(isize capacity, isize block_size, RuntimeAllocatorPtr runtime_allocator) : m_capacity(capacity), m_block_size(block_size), m_runtime_allocator(runtime_allocator) {
            m_pool = m_runtime_allocator->alloc(capacity);
            std::memset(m_pool, 0, capacity);

            for (isize i = 0; i < capacity; i += block_size) {
                MemoryBlock *block = new MemoryBlock(m_pool + i, block_size, this);
                m_free_blocks.push(block);
            }
        }

        ~MemoryPool() {
            for (ResourceNode *node : m_free_blocks) {
                delete static_cast<MemoryBlock *>(node);
            }

            m_runtime_allocator->free(m_pool);
        }

        isize get_capacity() const { return m_capacity; }
        isize get_block_size() const { return m_block_size; }
        bool empty() const { return m_free_blocks.empty(); }
        MemoryBlock *alloc() { return static_cast<MemoryBlock *>(m_free_blocks.pop()); }
        void free(MemoryBlock *block) { m_free_blocks.push(block); }
    };
} // namespace nx::memory