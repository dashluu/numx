#pragma once

#include "../primitive/block.h"

namespace nx::memory {
    using namespace nx::primitive;

    class Allocator : public std::enable_shared_from_this<Allocator> {
    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        virtual ~Allocator() = default;
        Allocator &operator=(const Allocator &) = delete;
        void alloc_pool(isize size);
        void free_pool();
        virtual Block alloc(isize size) = 0;
        virtual void free(const Block &block) = 0;
    };

    using AllocatorPtr = std::shared_ptr<Allocator>;
} // namespace nx::memory