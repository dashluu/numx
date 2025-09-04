#pragma once

#include "bind.h"

namespace nx::bind {
    template <class T>
    nb::ndarray<nb::numpy> array_to_numpy_impl(nxc::Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);
        std::vector<size_t> view(array.get_shape().begin(), array.get_shape().end());

        return nb::ndarray<nb::numpy>(
            array.get_ptr(),
            array.get_ndim(),
            view.data(),
            pyarr.ptr(),
            array.get_stride().data(),
            nb::dtype<T>(),
            // Numpy can only run on the cpu
            nb::device::cpu::value,
            'C');
    }

    template <class T>
    nb::ndarray<nb::pytorch> array_to_torch_impl(nxc::Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);
        std::vector<size_t> view(array.get_shape().begin(), array.get_shape().end());
        int device;

        switch (array.get_device()->get_type()) {
        case nxp::DeviceType::CPU:
            device = nb::device::cpu::value;
            break;
        default:
            // Try CPU for now
            // TODO: change to metal later
            device = nb::device::cpu::value;
            break;
        }

        return nb::ndarray<nb::pytorch>(
            array.get_ptr(),
            array.get_ndim(),
            view.data(),
            pyarr.ptr(),
            array.get_stride().data(),
            nb::dtype<T>(),
            device,
            'C');
    }

    inline std::string get_class_name(const nb::object &py_obj) {
        auto cls = py_obj.attr("__class__");
        auto name = cls.attr("__name__");
        return nb::cast<std::string>(name);
    }

    template <class F>
    nxc::Array binary(const nxc::Array &array, const nb::object &rhs, F &&f) {
        if (nb::isinstance<nxc::Array>(rhs)) {
            return f(array, nb::cast<nxc::Array>(rhs));
        } else if (nb::isinstance<nb::float_>(rhs)) {
            return f(array, nb::cast<float>(rhs));
        } else if (nb::isinstance<nb::int_>(rhs)) {
            return f(array, nb::cast<int>(rhs));
        } else if (nb::isinstance<nb::bool_>(rhs)) {
            return f(array, nb::cast<bool>(rhs));
        }

        throw nxp::NanobindInvalidArgumentType("float, int, bool, Array", get_class_name(rhs));
    }

    template <class F>
    nxc::Array in_place_binary(nxc::Array &array, const nb::object &rhs, F &&f) {
        if (nb::isinstance<nxc::Array>(rhs)) {
            return f(array, nb::cast<nxc::Array>(rhs));
        } else if (nb::isinstance<nb::float_>(rhs)) {
            return f(array, nb::cast<float>(rhs));
        } else if (nb::isinstance<nb::int_>(rhs)) {
            return f(array, nb::cast<int>(rhs));
        } else if (nb::isinstance<nb::bool_>(rhs)) {
            return f(array, nb::cast<bool>(rhs));
        }

        throw nxp::NanobindInvalidArgumentType("float, int, bool, Array", get_class_name(rhs));
    }

    nxp::isize get_index(nxp::isize len, nxp::isize index);
    nxp::ShapeDims get_indices(nxp::isize len, nxp::ShapeDims &dims);
    nxp::Range slice_to_range(nxp::isize len, const nb::object &slice);
    std::vector<nxp::Range> selector_to_ranges(const nxc::Array &array, const nb::object &selector);
    nxp::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
    const std::string device_from_nb_device(int nb_device_id, int nb_device_type);
    nb::ndarray<nb::numpy> array_to_numpy(nxc::Array &array);
    nxc::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
    nb::ndarray<nb::pytorch> array_to_torch(nxc::Array &array);
    nb::object item(nxc::Array &array);
    nxc::Array full(const nxp::ShapeView &view, const nb::object &constant, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
    nxc::Array full_like(const nxc::Array &array, const nb::object &constant, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
    nxc::Array neg(const nxc::Array &array);
    nxc::Array add(const nxc::Array &array, const nb::object &rhs);
    nxc::Array iadd(nxc::Array &array, const nb::object &rhs);
    nxc::Array sub(const nxc::Array &array, const nb::object &rhs);
    nxc::Array isub(nxc::Array &array, const nb::object &rhs);
    nxc::Array mul(const nxc::Array &array, const nb::object &rhs);
    nxc::Array imul(nxc::Array &array, const nb::object &rhs);
    nxc::Array div(const nxc::Array &array, const nb::object &rhs);
    nxc::Array idiv(nxc::Array &array, const nb::object &rhs);
    nxc::Array eq(const nxc::Array &array, const nb::object &rhs);
    nxc::Array neq(const nxc::Array &array, const nb::object &rhs);
    nxc::Array lt(const nxc::Array &array, const nb::object &rhs);
    nxc::Array gt(const nxc::Array &array, const nb::object &rhs);
    nxc::Array leq(const nxc::Array &array, const nb::object &rhs);
    nxc::Array geq(const nxc::Array &array, const nb::object &rhs);
    nxc::Array minimum(const nxc::Array &array, const nb::object &rhs);
    nxc::Array maximum(const nxc::Array &array, const nb::object &rhs);
    nxc::Array slice(const nxc::Array &array, const nb::object &selector);
    nxc::Array permute(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array transpose(const nxc::Array &array, nxp::isize start_dim, nxp::isize end_dim);
    nxc::Array flatten(const nxc::Array &array, nxp::isize start_dim, nxp::isize end_dim);
    nxc::Array squeeze(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array unsqueeze(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array sum(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array mean(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array max(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array min(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array argmax(const nxc::Array &array, nxp::ShapeDims &dims);
    nxc::Array argmin(const nxc::Array &array, nxp::ShapeDims &dims);
} // namespace nx::bind