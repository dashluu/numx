#pragma once

#include "memory_manager.h"
#include "memory_pool.h"

namespace nx::memory {
    struct CacheBucket {
    private:
        static constexpr isize s_blocks_per_pool = 4;
        // Cache line size
        static constexpr isize s_alignment = 128;
        isize m_pool_capacity;
        isize m_block_size;

    public:
        CacheBucket(isize pool_capacity, isize block_size) : m_pool_capacity(pool_capacity), m_block_size(block_size) {}
        CacheBucket(const CacheBucket &) = default;
        ~CacheBucket() = default;
        CacheBucket &operator=(const CacheBucket &) = default;
        bool operator==(const CacheBucket &bucket) const { return m_pool_capacity == bucket.m_pool_capacity && m_block_size == bucket.m_block_size; }
        isize get_pool_capacity() const { return m_pool_capacity; }
        isize get_block_size() const { return m_block_size; }
        static CacheBucket from_size(isize size);
    };
} // namespace nx::memory

namespace std {
    template <>
    struct hash<nx::memory::CacheBucket> {
        size_t operator()(const nx::memory::CacheBucket &bucket) const {
            return hash<nx::memory::isize>{}(bucket.get_block_size());
        }
    };
} // namespace std

namespace nx::memory {
    class CacheManager : public MemoryManager {
    private:
        std::unordered_map<CacheBucket, ResourceListPtr> m_free_pools_by_size;
        ResourceListPtr m_used_pools;

    public:
        CacheManager(RuntimeAllocatorPtr runtime_allocator) : MemoryManager(runtime_allocator) { m_used_pools = std::make_shared<ResourceList>(); }
        ~CacheManager();
        MemoryBlock *alloc(isize size) override;
        void free(MemoryBlock *block) override;
    };

    using CacheManagerPtr = std::shared_ptr<CacheManager>;
} // namespace nx::memory