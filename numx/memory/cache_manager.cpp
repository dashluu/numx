#include "cache_manager.h"

namespace nx::memory {
    CacheBucket CacheBucket::from_size(isize size) {
        // Round up to the nearest power of 2
        isize block_size = std::bit_ceil(static_cast<size_t>(size));
        isize pool_capacity = block_size * BLOCKS_PER_POOL;
        pool_capacity = align_to(pool_capacity, ALIGNMENT);
        return CacheBucket(pool_capacity, block_size);
    }

    CacheManager::~CacheManager() {
        for (ResourceNode *node : *m_used_pools) {
            delete static_cast<MemoryPool *>(node);
        }

        for (const auto &[bucket, free_pools] : m_free_pools_by_size) {
            for (ResourceNode *node : *free_pools) {
                delete static_cast<MemoryPool *>(node);
                std::println("Freed pool of capacity {}B and block size {}B...", bucket.get_pool_capacity(), bucket.get_block_size());
            }
        }

        m_free_pools_by_size.clear();
    }

    MemoryBlock *CacheManager::alloc(isize size) {
        if (size <= 0) {
            throw std::invalid_argument(std::format("Attemped to allocate a block of non-positive size {}B.", size));
        }

        // Bucket is always valid for valid size
        CacheBucket bucket = CacheBucket::from_size(size);
        MemoryPool *pool;

        if (!m_free_pools_by_size.contains(bucket)) {
            auto [iter, inserted] = m_free_pools_by_size.emplace(bucket, std::make_shared<ResourceList>());
            isize pool_capacity = bucket.get_pool_capacity();
            isize block_size = bucket.get_block_size();
            pool = new MemoryPool(pool_capacity, block_size, m_runtime_allocator);
            iter->second->push(pool);
            // Note: no need to check if the pool is empty since each pool is initialized with BLOCKS_PER_POOL blocks
            return pool->alloc();
        }

        ResourceListPtr free_pools = m_free_pools_by_size.at(bucket);
        pool = static_cast<MemoryPool *>(free_pools->peek());
        MemoryBlock *block = pool->alloc();

        if (pool->empty()) {
            free_pools->pop();
            m_used_pools->push(pool);
        }

        return block;
    }

    void CacheManager::free(MemoryBlock *block) {
        if (!block) {
            throw std::invalid_argument("Attempted to free a null block.");
        }

        MemoryPool *pool = block->get_pool();

        if (!pool) {
            throw std::invalid_argument("Attempted to free a block with no pool.");
        }

        CacheBucket bucket(pool->get_capacity(), pool->get_block_size());
        ResourceListPtr free_pools = m_free_pools_by_size.at(bucket);
        bool is_empty = pool->empty();
        pool->free(block);

        if (is_empty) {
            m_used_pools->unlink(pool);
            free_pools->push(pool);
        }
    }
} // namespace nx::memory
