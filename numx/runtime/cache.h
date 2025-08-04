#pragma once

#include "../primitive/buffer_pool.h"
#include "memory.h"

namespace nx::runtime {
    class Cache : public Memory {
    private:
        static constexpr isize s_blocks_per_pool = 4;
        // Cache line size
        static constexpr isize s_alignment = 128;
        std::unordered_map<isize, ResourceListPtr> m_free_pools_by_size;
        ResourceListPtr m_used_pools;

        static std::pair<isize, isize> get_pool_and_block_size(isize size);

    public:
        Cache(AllocatorPtr allocator, MemoryProfilerPtr memory_profiler) : Memory(allocator, memory_profiler) {
            m_used_pools = std::make_shared<ResourceList>();
        }

        ~Cache();
        BufferBlock *alloc_block(isize size);
        void free_block(BufferBlock *block);
    };

    using CachePtr = std::shared_ptr<Cache>;
} // namespace nx::runtime