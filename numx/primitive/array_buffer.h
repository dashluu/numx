#pragma once

#include "buffer_block.h"

namespace nx::primitive {
    struct ArrayBuffer {
    private:
        BufferBlock *m_block;
        bool m_is_view;

        bool is_primary() const { return m_block != nullptr && !m_is_view; }

    public:
        ArrayBuffer(BufferBlock *block, bool is_view) : m_block(block), m_is_view(is_view) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_is_view(true) { m_block = new BufferBlock(buffer.m_block->get_ptr(), buffer.m_block->get_size()); }

        ArrayBuffer(ArrayBuffer &&buffer) noexcept {
            m_block = buffer.m_block;
            m_is_view = buffer.m_is_view;
            buffer.m_block = nullptr;
            buffer.m_is_view = false;
        }

        ~ArrayBuffer() {
            if (!is_primary()) {
                delete m_block;
            }
        }

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            if (buffer == *this) {
                return *this;
            }

            if (is_primary()) {
                throw std::runtime_error("Cannot copy assign to a primary buffer.");
            } else if (buffer.m_block == nullptr) {
                delete m_block;
                m_block = nullptr;
                m_is_view = false;
            } else {
                delete m_block;
                m_block = new BufferBlock(buffer.get_ptr(), buffer.get_size());
                m_is_view = true;
            }

            return *this;
        }

        ArrayBuffer &operator=(ArrayBuffer &&buffer) {
            if (buffer == *this) {
                return *this;
            }

            if (is_primary()) {
                throw std::runtime_error("Cannot move assign to a primary buffer.");
            } else {
                delete m_block;
            }

            m_block = buffer.m_block;
            m_is_view = buffer.m_is_view;
            buffer.m_block = nullptr;
            buffer.m_is_view = false;
            return *this;
        }

        BufferBlock *get_block() const { return m_block; }
        bool is_view() const { return m_is_view; }
        uint8_t *get_ptr() const { return m_block->get_ptr(); }
        isize get_size() const { return m_block->get_size(); }
        bool operator==(const ArrayBuffer &buffer) const { return m_block == buffer.m_block && m_is_view == buffer.m_is_view; }
    };
} // namespace nx::primitive