#pragma once

#include "../utils.h"

namespace nx::allocator {
    using namespace nx::utils;

    class Allocator : public std::enable_shared_from_this<Allocator> {
    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        Allocator(Allocator &&) noexcept = delete;
        virtual ~Allocator() = default;
        Allocator &operator=(const Allocator &) = delete;
        Allocator &operator=(Allocator &&) noexcept = delete;
        virtual uint8_t *alloc_bytes(isize size) = 0;
        virtual void free_bytes(uint8_t *ptr) = 0;
    };

    using AllocatorPtr = std::shared_ptr<Allocator>;
} // namespace nx::allocator