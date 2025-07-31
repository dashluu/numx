#pragma once

#include "../memory/memory_block.h"

namespace nx::primitive {
    using namespace nx::utils;
    using namespace nx::memory;

    struct ArrayBuffer {
    private:
        MemoryBlock *m_block = nullptr;
        // Being persistent means that memory block is only freed when the array is freed
        // and is not freed when passed around using operator= or going out of scope
        bool m_is_persistent = false;

    public:
        ArrayBuffer() = default;
        ArrayBuffer(MemoryBlock *block, bool is_persistent) : m_block(block), m_is_persistent(is_persistent) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_block(buffer.m_block), m_is_persistent(buffer.m_is_persistent) {}

        ~ArrayBuffer() {
            if (m_is_persistent) {
                delete m_block;
            }
        }

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            if (m_is_persistent) {
                delete m_block;
            }

            m_block = buffer.m_block;
            m_is_persistent = buffer.m_is_persistent;
            return *this;
        }

        MemoryBlock *get_block() const { return m_block; }
        uint8_t *get_ptr() const { return m_block->get_ptr(); }
        isize get_size() const { return m_block->get_size(); }
        bool is_valid() const { return m_block && m_block->is_valid(); }
        bool is_persistent() const { return m_is_persistent && is_valid(); }
    };
} // namespace nx::primitive