#include "array_iter.h"

namespace nx::primitive {
    uint8_t *ArrayData::get_elm_ptr(isize index) const {
        if (is_contiguous()) {
            return get_ptr() + index * get_itemsize();
        }

        const ShapeStride &stride = get_stride();
        uint8_t *ptr = get_ptr();
        isize carry = index;

        for (isize i = get_ndim() - 1; i >= 0; i--) {
            ptr += (carry % m_shape[i]) * stride[i] * get_itemsize();
            carry /= m_shape[i];
        }

        return ptr;
    }

    const std::string ArrayData::str() const {
        ArrayIterator iter(*this);
        iter.start();
        bool next_elm_available = iter.has_next();

        if (!next_elm_available) {
            return "[]";
        }

        std::string s = "";

        for (isize i = 0; i < get_ndim(); i++) {
            s += "[";
        }

        ShapeView elms_per_dim = m_shape.get_elms_per_dim();
        size_t close = 0;

        while (next_elm_available) {
            close = 0;
            uint8_t *ptr = iter.next();
            s += m_dtype->value_str(ptr);

            for (ssize_t i = elms_per_dim.size() - 1; i >= 0; i--) {
                if (iter.count() % elms_per_dim[i] == 0) {
                    s += "]";
                    close += 1;
                }
            }

            next_elm_available = iter.has_next();

            if (next_elm_available) {
                if (close > 0) {
                    s += ", \n";
                } else {
                    s += ", ";
                }

                for (size_t i = 0; i < close; i++) {
                    s += "[";
                }
            }
        }

        return s;
    }
} // namespace nx::primitive