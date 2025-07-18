#pragma once

#include "array_buffer.h"
#include "array_id.h"
#include "device.h"
#include "dtype.h"
#include "exceptions.h"
#include "range.h"
#include "shape.h"

namespace nx::core {
    struct ArrayData {
    private:
        static ArrayIdGenerator s_id_gen;
        ArrayId m_id;
        Shape m_shape;
        DtypePtr m_dtype;
        DevicePtr m_device;

    public:
        ArrayBuffer m_buffer;

        ArrayData(const Shape &shape, DtypePtr dtype, DevicePtr device) : m_id(s_id_gen.generate()), m_shape(shape), m_dtype(dtype), m_device(device) {}
        ArrayData(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype, DevicePtr device) : m_id(s_id_gen.generate()), m_shape(shape), m_dtype(dtype), m_device(device) { m_buffer = ArrayBuffer(ptr, size, false); }
        ArrayData(const ArrayData &data) : m_id(data.m_id), m_shape(data.m_shape), m_dtype(data.m_dtype), m_device(data.m_device), m_buffer(data.m_buffer) {}
        ~ArrayData() = default;

        ArrayData &operator=(const ArrayData &data) {
            m_id = data.m_id;
            m_shape = data.m_shape;
            m_dtype = data.m_dtype;
            m_device = data.m_device;
            m_buffer = data.m_buffer;
            return *this;
        }

        static ArrayData from_buffer(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype, DevicePtr device) {
            return ArrayData(ptr, size, shape, dtype, device);
        }

        const ArrayId &get_id() const { return m_id; }
        const Shape &get_shape() const { return m_shape; }
        isize get_offset() const { return m_shape.get_offset(); }
        const ShapeView &get_view() const { return m_shape.get_view(); }
        const ShapeStride &get_stride() const { return m_shape.get_stride(); }
        DtypePtr get_dtype() const { return m_dtype; }
        DevicePtr get_device() const { return m_device; }
        const std::string &get_device_name() const { return m_device->get_name(); }
        uint8_t *get_ptr() const { return m_buffer.get_ptr() + get_offset() * get_itemsize(); }
        isize get_numel() const { return m_shape.get_numel(); }
        isize get_ndim() const { return m_shape.get_ndim(); }
        isize get_itemsize() const { return m_dtype->get_size(); }
        isize get_nbytes() const { return get_numel() * get_itemsize(); }
        bool is_contiguous() const { return m_shape.is_contiguous(); }
        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) const { return !is_contiguous(); }
        uint8_t *get_elm_ptr(isize index) const;
        bool operator==(const ArrayData &data) const { return m_id == data.m_id; }
        auto operator<=>(const ArrayData &data) const { return m_id <=> data.m_id; }
        const std::string str() const;
        friend std::ostream &operator<<(std::ostream &os, const ArrayData &data) { return os << data.str(); }
    };

    inline ArrayIdGenerator ArrayData::s_id_gen = ArrayIdGenerator();
}; // namespace nx::core

namespace std {
    template <>
    struct formatter<nx::core::ArrayData> : formatter<string> {
        auto format(const nx::core::ArrayData &data, format_context &ctx) const {
            return formatter<string>::format(data.str(), ctx);
        }
    };
} // namespace std