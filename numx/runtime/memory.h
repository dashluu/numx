#pragma once

#include "../allocator/allocator.h"
#include "../primitive/buffer_block.h"
#include "memory_profiler.h"

namespace nx::runtime {
    using namespace nx::allocator;

    class Memory : public std::enable_shared_from_this<Memory> {
    protected:
        AllocatorPtr m_allocator;
        MemoryProfilerPtr m_memory_profiler;

    public:
        Memory(AllocatorPtr allocator, MemoryProfilerPtr memory_profiler) : m_allocator(allocator), m_memory_profiler(memory_profiler) {}
        Memory(const Memory &) = delete;
        Memory(Memory &&) noexcept = delete;
        virtual ~Memory() = default;
        Memory &operator=(const Memory &) = delete;
        Memory &operator=(Memory &&) noexcept = delete;
        virtual BufferBlock *alloc_block(isize size) = 0;
        virtual void free_block(BufferBlock *block) = 0;
    };

    using MemoryPtr = std::shared_ptr<Memory>;
} // namespace nx::runtime