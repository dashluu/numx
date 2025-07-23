#pragma once

#include "block.h"

namespace nx::primitive {
    using namespace nx::utils;

    struct ArrayBuffer {
    private:
        Block m_block;
        // A buffer is primary when it is an original buffer and not just a view
        bool m_primary;

    public:
        ArrayBuffer() = default;
        ArrayBuffer(uint8_t *ptr, isize size, bool is_primary) : m_block(ptr, size), m_primary(is_primary) {}
        ArrayBuffer(const Block &block, bool is_primary) : m_block(block), m_primary(is_primary) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_block(buffer.m_block), m_primary(buffer.m_primary) {}
        ~ArrayBuffer() = default;

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            m_block = buffer.m_block;
            m_primary = buffer.m_primary;
            return *this;
        }

        const Block &get_block() const { return m_block; }
        uint8_t *get_ptr() const { return m_block.get_ptr(); }
        isize get_size() const { return m_block.get_size(); }
        bool is_valid() const { return m_block.get_ptr() != nullptr; }
        bool is_primary() const { return m_primary && is_valid(); }

        void invalidate() {
            m_block = Block();
            m_primary = false;
        }
    };
} // namespace nx::primitive