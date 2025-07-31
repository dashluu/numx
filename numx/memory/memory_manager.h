#pragma once

#include "memory_block.h"
#include "runtime_allocator.h"

namespace nx::memory {
    class MemoryManager : public std::enable_shared_from_this<MemoryManager> {
    protected:
        RuntimeAllocatorPtr m_runtime_allocator;

    public:
        MemoryManager(RuntimeAllocatorPtr runtime_allocator) : m_runtime_allocator(runtime_allocator) {}
        MemoryManager(const MemoryManager &) = delete;
        virtual ~MemoryManager() = default;
        MemoryManager &operator=(const MemoryManager &) = delete;
        virtual MemoryBlock *alloc(isize size) = 0;
        virtual void free(MemoryBlock *block) = 0;
    };

    using MemoryManagerPtr = std::shared_ptr<MemoryManager>;
} // namespace nx::memory