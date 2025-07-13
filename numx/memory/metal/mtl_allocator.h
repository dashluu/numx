#pragma once

#include "../allocator.h"

namespace nx::memory::metal {
    class MTLAllocator : public Allocator {
    public:
        Block alloc(isize size) override {
            auto ptr = new uint8_t[size];
            std::memset(ptr, 0, size);
            return Block(ptr, size);
        }

        void free(const Block &block) override {
            delete[] block.get_ptr();
        }
    };

    using MTLAllocatorPtr = std::shared_ptr<MTLAllocator>;
} // namespace nx::memory::metal