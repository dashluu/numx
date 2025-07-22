#pragma once

#include "bind.h"

namespace nx::bind {
    template <class T>
    nb::ndarray<nb::numpy> array_to_numpy_impl(nxc::Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);
        std::vector<size_t> view(array.get_shape().cbegin(), array.get_shape().cend());

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
        std::vector<size_t> view(array.get_shape().cbegin(), array.get_shape().cend());
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

    inline std::string get_pyclass(const nb::object &obj) {
        auto cls = obj.attr("__class__");
        auto name = cls.attr("__name__");
        return nb::cast<std::string>(name);
    }

    template <class F>
    nxc::Array binary(const nxc::Array &array, const nb::object &obj, F &&f) {
        if (nb::isinstance<nxc::Array>(obj)) {
            return f(array, nb::cast<nxc::Array>(obj));
        } else if (nb::isinstance<nb::float_>(obj)) {
            return f(array, nb::cast<float>(obj));
        } else if (nb::isinstance<nb::int_>(obj)) {
            return f(array, nb::cast<int>(obj));
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return f(array, nb::cast<bool>(obj));
        }

        throw nxp::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    template <class F>
    nxc::Array in_place_binary(nxc::Array &array, const nb::object &obj, F &&f) {
        if (nb::isinstance<nxc::Array>(obj)) {
            return f(array, nb::cast<nxc::Array>(obj));
        } else if (nb::isinstance<nb::float_>(obj)) {
            return f(array, nb::cast<float>(obj));
        } else if (nb::isinstance<nb::int_>(obj)) {
            return f(array, nb::cast<int>(obj));
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return f(array, nb::cast<bool>(obj));
        }

        throw nxp::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    nxp::isize get_pyindex(nxp::isize len, nxp::isize idx);
    nxp::ShapeDims get_pyindices(nxp::isize len, nxp::ShapeDims &dims);
    nxp::Range pyslice_to_range(nxp::isize len, const nb::object &obj);
    std::vector<nxp::Range> pyslices_to_ranges(const nxc::Array &array, const nb::object &obj);
    nxp::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
    const std::string device_from_nb_device(int nb_device_id, int nb_device_type);
    nb::ndarray<nb::numpy> array_to_numpy(nxc::Array &array);
    nxc::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
    nb::ndarray<nb::pytorch> array_to_torch(nxc::Array &array);
    nb::object item(nxc::Array &array);
    nxc::Array full(const nxp::ShapeView &view, const nb::object &obj, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
    nxc::Array full_like(const nxc::Array &other, const nb::object &obj, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
    nxc::Array neg(const nxc::Array &array);
    nxc::Array add(const nxc::Array &array, const nb::object &obj);
    nxc::Array iadd(nxc::Array &array, const nb::object &obj);
    nxc::Array sub(const nxc::Array &array, const nb::object &obj);
    nxc::Array isub(nxc::Array &array, const nb::object &obj);
    nxc::Array mul(const nxc::Array &array, const nb::object &obj);
    nxc::Array imul(nxc::Array &array, const nb::object &obj);
    nxc::Array div(const nxc::Array &array, const nb::object &obj);
    nxc::Array idiv(nxc::Array &array, const nb::object &obj);
    nxc::Array eq(const nxc::Array &array, const nb::object &obj);
    nxc::Array neq(const nxc::Array &array, const nb::object &obj);
    nxc::Array lt(const nxc::Array &array, const nb::object &obj);
    nxc::Array gt(const nxc::Array &array, const nb::object &obj);
    nxc::Array leq(const nxc::Array &array, const nb::object &obj);
    nxc::Array geq(const nxc::Array &array, const nb::object &obj);
    nxc::Array minimum(const nxc::Array &array, const nb::object &obj);
    nxc::Array maximum(const nxc::Array &array, const nb::object &obj);
    nxc::Array slice(const nxc::Array &array, const nb::object &obj);
    nxc::Array permute(const nxc::Array &array, nx::primitive::ShapeDims &dims);
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