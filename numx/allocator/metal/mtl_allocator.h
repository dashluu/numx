#pragma once

#include "../allocator.h"

namespace nx::allocator::metal {
    class MTLAllocator : public Allocator {
    public:
        MTLAllocator() = default;
        ~MTLAllocator() = default;
        uint8_t *alloc_bytes(isize size) override { return new uint8_t[size]; }
        void free_bytes(uint8_t *ptr) override { delete[] ptr; }
    };

    using MTLAllocatorPtr = std::shared_ptr<MTLAllocator>;
} // namespace nx::allocator::metal