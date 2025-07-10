#pragma once

#include "../pool_allocator.h"

namespace nx::runtime::metal {
    struct MTLAllocator : public PoolAllocator {
    public:
        StoragePool alloc(isize capacity) override {
            auto base = new uint8_t[capacity];
            std::memset(base, 0, capacity);
            return StoragePool(base, capacity);
        }

        void free(const StoragePool &pool) override {
            delete[] pool.get_base();
        }
    };
} // namespace nx::runtime::metal