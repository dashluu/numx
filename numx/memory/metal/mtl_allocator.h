#pragma once

#include "../runtime_allocator.h"

namespace nx::memory::metal {
    class MTLAllocator : public RuntimeAllocator {
    public:
        MTLAllocator() = default;
        ~MTLAllocator() = default;
        uint8_t *alloc(isize size) override { return new uint8_t[size]; }
        void free(uint8_t *ptr) override { delete[] ptr; }
    };

    using MTLAllocatorPtr = std::shared_ptr<MTLAllocator>;
} // namespace nx::memory::metal