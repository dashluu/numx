#pragma once

#include "../memory/allocator.h"

namespace nx::core {
    using namespace nx::utils;
    using namespace nx::memory;

    struct ArrayBuffer {
    private:
        MemoryBlock m_block;
        AllocatorPtr m_allocator = nullptr;

    public:
        ArrayBuffer() = default;
        ArrayBuffer(AllocatorPtr allocator, isize size) : m_allocator(allocator) { m_block = allocator->alloc(size); }
        ArrayBuffer(uint8_t *ptr, isize size) : m_block(ptr, size) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_block(buffer.m_block) {}

        ~ArrayBuffer() {
            if (m_allocator) {
                m_allocator->free(m_block);
            }
        }

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            m_block = buffer.m_block;
            return *this;
        }

        MemoryBlock get_block() const { return m_block; }
        uint8_t *get_ptr() const { return m_block.get_ptr(); }
        isize get_size() const { return m_block.get_size(); }
    };
} // namespace nx::core