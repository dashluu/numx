#pragma once

#include "buffer_block.h"

namespace nx::primitive {
    enum struct ArrayBufferType {
        External,
        Managed,
        View
    };

    struct ArrayBuffer {
    private:
        BufferBlock *m_block = nullptr;
        ArrayBufferType m_type;

    public:
        ArrayBuffer() = default;
        ArrayBuffer(BufferBlock *block, ArrayBufferType type) : m_block(block), m_type(type) {}
        ArrayBuffer(const ArrayBuffer &buffer) : m_block(buffer.m_block), m_type(buffer.m_type) {}

        ~ArrayBuffer() {
            if (m_type == ArrayBufferType::External) {
                delete m_block;
            }
        }

        ArrayBuffer &operator=(const ArrayBuffer &buffer) {
            if (m_type == ArrayBufferType::External) {
                delete m_block;
            }

            m_block = buffer.m_block;
            m_type = buffer.m_type;
            return *this;
        }

        BufferBlock *get_block() const { return m_block; }
        ArrayBufferType get_type() const { return m_type; }
        uint8_t *get_ptr() const { return m_block->get_ptr(); }
        isize get_size() const { return m_block->get_size(); }
        bool is_valid() const { return m_block && m_block->is_valid(); }
    };
} // namespace nx::primitive