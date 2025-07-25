#pragma once

#include "../utils.h"

namespace nx::primitive {
    using namespace nx::utils;

    struct Block {
    private:
        uint8_t *m_ptr = nullptr;
        isize m_size = 0;

    public:
        Block() = default;
        Block(uint8_t *ptr, isize size) : m_ptr(ptr), m_size(size) {}
        Block(const Block &block) : m_ptr(block.m_ptr), m_size(block.m_size) {}
        ~Block() = default;

        Block &operator=(const Block &block) {
            m_ptr = block.m_ptr;
            m_size = block.m_size;
            return *this;
        }

        uint8_t *get_ptr() const { return m_ptr; }
        isize get_size() const { return m_size; }
        bool is_valid() const { return m_ptr != nullptr && m_size > 0; }
    };
} // namespace nx::primitive