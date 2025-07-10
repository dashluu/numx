#pragma once

#include "array_id.h"
#include "device.h"
#include "dtype.h"
#include "exceptions.h"
#include "range.h"
#include "shape.h"

namespace nx::core {
    class ArrayDescriptor {
    private:
        static ArrayIdGenerator s_id_gen;
        ArrayId m_id;
        Shape m_shape;
        DtypePtr m_dtype;
        DevicePtr m_device;

    public:
        ArrayDescriptor(const Shape &shape, DtypePtr dtype, DevicePtr device) : m_id(s_id_gen.generate()), m_shape(shape), m_dtype(dtype), m_device(device) {}
        ArrayDescriptor(const ArrayDescriptor &descriptor) : m_id(s_id_gen.generate()), m_shape(descriptor.m_shape), m_dtype(descriptor.m_dtype), m_device(descriptor.m_device) {}
        ~ArrayDescriptor() = default;

        ArrayDescriptor &operator=(const ArrayDescriptor &descriptor) {
            m_shape = descriptor.m_shape;
            m_dtype = descriptor.m_dtype;
            m_device = descriptor.m_device;
            return *this;
        }

        const ArrayId &get_id() const { return m_id; }
        const Shape &get_shape() const { return m_shape; }
        isize get_offset() const { return m_shape.get_offset(); }
        const ShapeView &get_view() const { return m_shape.get_view(); }
        const ShapeStride &get_stride() const { return m_shape.get_stride(); }
        DtypePtr get_dtype() const { return m_dtype; }
        DevicePtr get_device() const { return m_device; }
        const std::string &get_device_name() const { return m_device->get_name(); }
        isize get_numel() const { return m_shape.get_numel(); }
        isize get_ndim() const { return m_shape.get_ndim(); }
        isize get_itemsize() const { return m_dtype->get_size(); }
        isize get_nbytes() const { return get_numel() * get_itemsize(); }
        bool is_contiguous() const { return m_shape.is_contiguous(); }
        // TODO: handle more cases to reduce copying?
        bool copy_when_reshape(const ShapeView &view) { return !is_contiguous(); }
        const std::string str() const;
    };

    inline ArrayIdGenerator ArrayDescriptor::s_id_gen = ArrayIdGenerator();
}; // namespace nx::core