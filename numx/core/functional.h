#pragma once

#include "array.h"

namespace nx::core {
    template <NumericType T>
    Array operator+(T constant, const Array &array) { return array + constant; }

    template <NumericType T>
    Array operator-(T constant, const Array &array) { return array - constant; }

    template <NumericType T>
    Array operator*(T constant, const Array &array) { return array * constant; }

    template <NumericType T>
    Array operator/(T constant, const Array &array) { return array.recip() * constant; }

    inline Array from_buffer(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::from_buffer(ptr, size, shape, dtype, device));
    }

    template <NumericOrBoolType T>
    Array full(const ShapeView &view, T constant, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::full(view, constant, dtype, device));
    }

    template <NumericOrBoolType T>
    Array full_like(const Array &array, T constant, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::full_like(array.get_op(), constant, dtype, device));
    }

    inline Array arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::arange(view, start, step, dtype, device));
    }

    inline Array zeros(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::zeros(view, dtype, device));
    }

    inline Array ones(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::ones(view, dtype, device));
    }

    inline Array zeros_like(const Array &array, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::zeros_like(array.get_op(), dtype, device));
    }

    inline Array ones_like(const Array &array, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::ones_like(array.get_op(), dtype, device));
    }

    inline Array empty(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::empty(view, dtype, device));
    }

    inline Array empty_like(const Array &array, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        return Array(nx::graph::empty_like(array.get_op(), dtype, device));
    }

    inline Array empty_like(const Array &array) { return Array(nx::graph::empty_like(array.get_op())); }
    std::pair<isize, isize> compute_fan_in_and_fan_out(const ShapeView &view);
    std::pair<isize, isize> compute_fan_in_and_fan_out(const Array &array);
} // namespace nx::core