#pragma once

#include "../utils.h"

namespace nx::memory {
    using namespace nx::utils;

    struct MemoryBlock {
    private:
        uint8_t *m_ptr = nullptr;
        isize m_size = 0;

    public:
        MemoryBlock() = default;
        MemoryBlock(uint8_t *ptr, isize size) : m_ptr(ptr), m_size(size) {}
        MemoryBlock(const MemoryBlock &block) : m_ptr(block.m_ptr), m_size(block.m_size) {}
        ~MemoryBlock() = default;

        MemoryBlock &operator=(const MemoryBlock &block) {
            m_ptr = block.m_ptr;
            m_size = block.m_size;
            return *this;
        }

        uint8_t *get_ptr() const { return m_ptr; }
        isize get_size() const { return m_size; }
    };

    class Allocator : public std::enable_shared_from_this<Allocator> {
    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        virtual ~Allocator() = default;
        Allocator &operator=(const Allocator &) = delete;
        virtual MemoryBlock alloc(isize size) = 0;
        virtual void free(const MemoryBlock &block) = 0;
    };

    using AllocatorPtr = std::shared_ptr<Allocator>;
} // namespace nx::memory