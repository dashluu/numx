#include "cache.h"

namespace nx::runtime {
    std::pair<isize, isize> Cache::get_pool_and_block_size(isize size) {
        // Round up to the nearest power of 2
        isize block_size = std::bit_ceil(static_cast<size_t>(size));
        isize pool_capacity = block_size * s_blocks_per_pool;
        pool_capacity = align_to(pool_capacity, s_alignment);
        return {pool_capacity, block_size};
    }

    Cache::~Cache() {
        BufferPool *pool;

        for (Resource *resource : *m_used_pools) {
            pool = static_cast<BufferPool *>(resource);

            if (m_memory_profiler->is_enabled()) {
                m_memory_profiler->trace_free_pool(pool->get_capacity());
            }

            delete pool;
        }

        for (const auto &[bucket, free_pools] : m_free_pools_by_size) {
            for (Resource *resource : *free_pools) {
                pool = static_cast<BufferPool *>(resource);

                if (m_memory_profiler->is_enabled()) {
                    m_memory_profiler->trace_free_pool(pool->get_capacity());
                }

                delete pool;
            }
        }

        std::println("Leaked {}B from buffer pools...", m_memory_profiler->get_pool_memory());
        m_free_pools_by_size.clear();
    }

    BufferBlock *Cache::alloc_block(isize size) {
        if (size <= 0) {
            throw std::invalid_argument(std::format("Attemped to allocate a block of non-positive size {}B.", size));
        }

        auto [pool_capacity, block_size] = get_pool_and_block_size(size);
        BufferPool *pool;

        if (!m_free_pools_by_size.contains(block_size)) {
            auto [iter, inserted] = m_free_pools_by_size.emplace(block_size, std::make_shared<ResourceList>());
            pool = new BufferPool(pool_capacity, block_size, m_allocator);

            if (m_memory_profiler->is_enabled()) {
                m_memory_profiler->trace_alloc_pool(pool_capacity);
            }

            iter->second->push(pool);
            // Note: no need to check if the pool is empty since each pool is initialized with s_blocks_per_pool blocks
            return pool->alloc_block();
        }

        ResourceListPtr free_pools = m_free_pools_by_size.at(block_size);

        if (free_pools->empty()) {
            pool = new BufferPool(pool_capacity, block_size, m_allocator);

            if (m_memory_profiler->is_enabled()) {
                m_memory_profiler->trace_alloc_pool(pool_capacity);
            }

            free_pools->push(pool);
            return pool->alloc_block();
        }

        pool = static_cast<BufferPool *>(free_pools->peek());
        BufferBlock *block = pool->alloc_block();

        if (pool->empty()) {
            free_pools->pop();
            m_used_pools->push(pool);
        }

        return block;
    }

    void Cache::free_block(BufferBlock *block) {
        if (!block) {
            throw std::invalid_argument("Attempted to free a null block.");
        }

        BufferPool *pool = block->get_pool();

        if (!pool) {
            throw std::invalid_argument("Attempted to free a block with no pool.");
        }

        ResourceListPtr free_pools = m_free_pools_by_size.at(pool->get_block_size());
        bool is_empty = pool->empty();
        pool->free_block(block);

        if (is_empty) {
            m_used_pools->unlink(pool);
            free_pools->push(pool);
        }
    }
} // namespace nx::runtime
