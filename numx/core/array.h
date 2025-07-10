#pragma once

#include "op_impl.h"

namespace nx::core {
    struct Array {
    private:
        OpPtr m_op = nullptr;

    public:
        Array(OpPtr op) : m_op(op) {}
        Array(const Array &array) : m_op(array.m_op) {}

        Array &operator=(const Array &array) {
            m_op = array.m_op;
            return *this;
        }

        const OpPtr &get_op() const { return m_op; }
    };
} // namespace nx::core