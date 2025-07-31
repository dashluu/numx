#pragma once

#include "../utils.h"

namespace nx::memory {
    using namespace nx::utils;

    class RuntimeAllocator : public std::enable_shared_from_this<RuntimeAllocator> {
    public:
        RuntimeAllocator() = default;
        RuntimeAllocator(const RuntimeAllocator &) = delete;
        virtual ~RuntimeAllocator() = default;
        RuntimeAllocator &operator=(const RuntimeAllocator &) = delete;
        virtual uint8_t *alloc(isize size) = 0;
        virtual void free(uint8_t *ptr) = 0;
    };

    using RuntimeAllocatorPtr = std::shared_ptr<RuntimeAllocator>;
} // namespace nx::memory