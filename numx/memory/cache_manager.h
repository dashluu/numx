#pragma once

#include "memory_manager.h"
#include "memory_pool.h"

namespace nx::memory {
    struct CacheBucket {
    private:
        static constexpr isize BLOCKS_PER_POOL = 4;
        // Cache line size
        static constexpr isize ALIGNMENT = 128;
        isize pool_capacity;
        isize block_size;

    public:
        CacheBucket(isize pool_capacity, isize block_size) : pool_capacity(pool_capacity), block_size(block_size) {}
        CacheBucket(const CacheBucket &) = default;
        ~CacheBucket() = default;
        CacheBucket &operator=(const CacheBucket &) = default;
        bool operator==(const CacheBucket &bucket) const { return pool_capacity == bucket.pool_capacity && block_size == bucket.block_size; }
        isize get_pool_capacity() const { return pool_capacity; }
        isize get_block_size() const { return block_size; }
        static CacheBucket from_size(isize size);
    };
} // namespace nx::memory

namespace std {
    template <>
    struct hash<nx::memory::CacheBucket> {
        size_t operator()(const nx::memory::CacheBucket &bucket) const {
            size_t h1 = hash<nx::memory::isize>{}(bucket.get_pool_capacity());
            size_t h2 = hash<nx::memory::isize>{}(bucket.get_block_size());
            return h1 ^ (h2 + 0x9e3779b97f4a7c15 + (h1 << 6) + (h1 >> 2));
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