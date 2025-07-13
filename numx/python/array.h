#pragma once

#include "bind.h"

namespace nx::bind {
    template <class T>
    nb::ndarray<nb::numpy> array_to_numpy_impl(nxa::Array &array) {
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
    nb::ndarray<nb::pytorch> array_to_torch_impl(nxa::Array &array) {
        array.eval();
        nb::object pyarr = nb::find(array);
        std::vector<size_t> view(array.get_shape().cbegin(), array.get_shape().cend());
        int device;

        switch (array.get_device()->get_type()) {
        case nxc::DeviceType::CPU:
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
    nxa::Array binary(const nxa::Array &array, const nb::object &obj, F &&f) {
        if (nb::isinstance<nxa::Array>(obj)) {
            return f(array, nb::cast<nxa::Array>(obj));
        } else if (nb::isinstance<nb::float_>(obj)) {
            return f(array, nb::cast<float>(obj));
        } else if (nb::isinstance<nb::int_>(obj)) {
            return f(array, nb::cast<int>(obj));
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return f(array, nb::cast<bool>(obj));
        }

        throw nxc::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    template <class F>
    nxa::Array in_place_binary(nxa::Array &array, const nb::object &obj, F &&f) {
        if (nb::isinstance<nxa::Array>(obj)) {
            return f(array, nb::cast<nxa::Array>(obj));
        } else if (nb::isinstance<nb::float_>(obj)) {
            return f(array, nb::cast<float>(obj));
        } else if (nb::isinstance<nb::int_>(obj)) {
            return f(array, nb::cast<int>(obj));
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return f(array, nb::cast<bool>(obj));
        }

        throw nxc::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    nxc::isize get_pyindex(nxc::isize len, nxc::isize idx);
    nxc::ShapeDims get_pyindices(nxc::isize len, nxc::ShapeDims &dims);
    nxc::Range pyslice_to_range(nxc::isize len, const nb::object &obj);
    std::vector<nxc::Range> pyslices_to_ranges(const nxa::Array &array, const nb::object &obj);
    nxc::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
    const std::string device_from_nb_device(int nb_device_id, int nb_device_type);
    nb::ndarray<nb::numpy> array_to_numpy(nxa::Array &array);
    nxa::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
    nb::ndarray<nb::pytorch> array_to_torch(nxa::Array &array);
    nb::object item(nxa::Array &array);
    nxa::Array full(const nxc::ShapeView &view, const nb::object &obj, nxc::DtypePtr dtype, const std::string &device_name = nxc::default_device_name);
    nxa::Array full_like(const nxa::Array &other, const nb::object &obj, nxc::DtypePtr dtype, const std::string &device_name = nxc::default_device_name);
    nxa::Array neg(const nxa::Array &array);
    nxa::Array add(const nxa::Array &array, const nb::object &obj);
    nxa::Array iadd(nxa::Array &array, const nb::object &obj);
    nxa::Array sub(const nxa::Array &array, const nb::object &obj);
    nxa::Array isub(nxa::Array &array, const nb::object &obj);
    nxa::Array mul(const nxa::Array &array, const nb::object &obj);
    nxa::Array imul(nxa::Array &array, const nb::object &obj);
    nxa::Array div(const nxa::Array &array, const nb::object &obj);
    nxa::Array idiv(nxa::Array &array, const nb::object &obj);
    nxa::Array eq(const nxa::Array &array, const nb::object &obj);
    nxa::Array neq(const nxa::Array &array, const nb::object &obj);
    nxa::Array lt(const nxa::Array &array, const nb::object &obj);
    nxa::Array gt(const nxa::Array &array, const nb::object &obj);
    nxa::Array leq(const nxa::Array &array, const nb::object &obj);
    nxa::Array geq(const nxa::Array &array, const nb::object &obj);
    nxa::Array minimum(const nxa::Array &array, const nb::object &obj);
    nxa::Array maximum(const nxa::Array &array, const nb::object &obj);
    nxa::Array slice(const nxa::Array &array, const nb::object &obj);
    nxa::Array permute(const nxa::Array &array, nx::core::ShapeDims &dims);
    nxa::Array transpose(const nxa::Array &array, nxc::isize start_dim, nxc::isize end_dim);
    nxa::Array flatten(const nxa::Array &array, nxc::isize start_dim, nxc::isize end_dim);
    nxa::Array squeeze(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array unsqueeze(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array sum(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array mean(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array max(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array min(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array argmax(const nxa::Array &array, nxc::ShapeDims &dims);
    nxa::Array argmin(const nxa::Array &array, nxc::ShapeDims &dims);
} // namespace nx::bind