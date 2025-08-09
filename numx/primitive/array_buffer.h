#pragma once

#include "buffer_block.h"

namespace nx::primitive {
    struct ArrayBuffer {
    private:
        BufferBlock *m_block = nullptr;
        bool m_is_view;

    public:
        ArrayBuffer(BufferBlock *block, bool is_view) : m_block(block), m_is_view(is_view) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_is_view(true) { m_block = new BufferBlock(buffer.m_block->get_ptr(), buffer.m_block->get_size()); }

        ~ArrayBuffer() {
            if (m_is_view) {
                delete m_block;
            }
        }

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            if (m_is_view) {
                delete m_block;
            }

            // Note: assigning another buffer to a primary buffer, or owning buffer, causes a memory leak
            m_block = new BufferBlock(buffer.m_block->get_ptr(), buffer.m_block->get_size());
            m_is_view = true;
            return *this;
        }

        BufferBlock *get_block() const { return m_block; }
        bool is_view() const { return m_is_view; }
        uint8_t *get_ptr() const { return m_block->get_ptr(); }
        isize get_size() const { return m_block->get_size(); }
    };
} // namespace nx::primitive