#pragma once

#include "array_data.h"

namespace nx::core {
    struct ArrayIterator {
    private:
        ArrayData m_data;
        uint8_t *m_ptr;
        isize m_counter;

    public:
        ArrayIterator(const ArrayData &data) : m_data(data) {}
        ArrayIterator(const ArrayIterator &) = delete;
        ~ArrayIterator() = default;
        ArrayIterator &operator=(const ArrayIterator &) = delete;
        bool has_next() const { return m_counter < m_data.get_shape().get_numel(); }
        isize count() const { return m_counter; }
        void start() { m_counter = 0; }

        uint8_t *next() {
            m_ptr = m_data.is_contiguous() ? m_data.get_ptr() + m_counter * m_data.get_itemsize() : m_data.get_elm_ptr(m_counter);
            m_counter++;
            return m_ptr;
        }
    };
} // namespace nx::core