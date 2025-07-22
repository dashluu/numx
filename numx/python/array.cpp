#include "array.h"

namespace nx::bind {
    nxp::isize get_pyindex(nxp::isize len, nxp::isize idx) {
        if (idx < -len || idx >= len) {
            throw nxp::OutOfRange(idx, -len, len);
        }
        return idx < 0 ? idx + len : idx;
    }

    nxp::ShapeDims get_pyindices(nxp::isize len, nxp::ShapeDims &dims) {
        if (!dims.empty()) {
            std::transform(dims.begin(), dims.end(), dims.begin(), [len](auto dim) { return get_pyindex(len, dim); });
        }
        return dims;
    }

    nxp::Range pyslice_to_range(nxp::isize len, const nb::object &obj) {
        // Note: no need to check for out-of-bounds indices when converting to range
        // Shape does the checking eventually
        if (!nb::isinstance<nb::slice>(obj)) {
            throw nxp::NanobindInvalidArgumentType(get_pyclass(obj), "slice");
        }

        auto slice = nb::cast<nb::slice>(obj);
        bool start_none = slice.attr("start").is_none();
        bool stop_none = slice.attr("stop").is_none();
        bool step_none = slice.attr("step").is_none();
        nxp::isize start, stop, step;

        if (step_none) {
            start = start_none ? 0 : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("start")));
            stop = stop_none ? len : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("stop")));
            return nxp::Range(start, stop, 1);
        }

        step = nb::cast<nxp::isize>(slice.attr("step"));

        if (step > 0) {
            start = start_none ? 0 : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("start")));
            stop = stop_none ? len : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("stop")));
        } else {
            start = start_none ? len - 1 : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("start")));
            stop = stop_none ? -1 : get_pyindex(len, nb::cast<nxp::isize>(slice.attr("stop")));
        }

        return nxp::Range(start, stop, step);
    }

    std::vector<nxp::Range> pyslices_to_ranges(const nxc::Array &array, const nb::object &obj) {
        std::vector<nxp::Range> ranges;
        const nxp::Shape &shape = array.get_shape();

        // obj can be an int, a slice, or a sequence of ints or slices
        if (nb::isinstance<nb::int_>(obj)) {
            nxp::isize idx = get_pyindex(shape[0], nb::cast<nxp::isize>(obj));
            ranges.emplace_back(idx, idx + 1, 1);

            for (nxp::isize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::slice>(obj)) {
            ranges.push_back(pyslice_to_range(shape[0], obj));

            for (nxp::isize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::sequence>(obj) && !nb::isinstance<nb::str>(obj)) {
            // Object is a sequence but not a string
            auto sequence = nb::cast<nb::sequence>(obj);
            size_t seq_len = nb::len(sequence);

            if (seq_len > shape.get_ndim()) {
                throw nxp::OutOfRange(seq_len, 1, shape.get_ndim() + 1);
            }

            for (size_t i = 0; i < seq_len; i++) {
                auto elm = sequence[i];
                // elm must be a sequence of ints or slices
                if (nb::isinstance<nb::int_>(elm)) {
                    nxp::isize idx = get_pyindex(shape[i], nb::cast<nxp::isize>(elm));
                    ranges.emplace_back(idx, idx + 1, 1);
                } else if (nb::isinstance<nb::slice>(elm)) {
                    ranges.push_back(pyslice_to_range(shape[i], elm));
                } else {
                    throw nxp::NanobindInvalidArgumentType(get_pyclass(elm), "int, slice");
                }
            }

            for (nxp::isize i = seq_len; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        }

        throw nxp::NanobindInvalidArgumentType(get_pyclass(obj), "int, slice, sequence");
    }

    nxp::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype) {
        if (nb_dtype == nb::dtype<float>()) {
            return &nxp::f32;
        } else if (nb_dtype == nb::dtype<int>()) {
            return &nxp::i32;
        } else if (nb_dtype == nb::dtype<bool>()) {
            return &nxp::b8;
        }
        throw nb::type_error("Nanobind data type cannot be converted to arrayx data type.");
    }

    const std::string device_from_nb_device(int nb_device_id, int nb_device_type) {
        if (nb_device_type == nb::device::cpu::value) {
            return std::format("cpu:{}", nb_device_id);
        } else if (nb_device_type == nb::device::metal::value) {
            return std::format("mps:{}", nb_device_id);
        }
        throw std::invalid_argument("Nanobind device is not supported.");
    }

    nb::ndarray<nb::numpy> array_to_numpy(nxc::Array &array) {
        switch (array.get_dtype()->get_name()) {
        case nxp::DtypeName::F32:
            return array_to_numpy_impl<float>(array);
        case nxp::DtypeName::I32:
            return array_to_numpy_impl<int>(array);
        default:
            return array_to_numpy_impl<bool>(array);
        }
    }

    nxc::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr) {
        nxp::ShapeView view;
        nxp::ShapeStride stride;

        for (size_t i = 0; i < ndarr.ndim(); ++i) {
            view.push_back(ndarr.shape(i));
            stride.push_back(ndarr.stride(i));
        }

        nxp::Shape shape(0, view, stride);
        uint8_t *ptr = reinterpret_cast<uint8_t *>(ndarr.data());
        nxp::DtypePtr dtype = dtype_from_nb_dtype(ndarr.dtype());
        return nxc::Array::from_buffer(ptr, ndarr.nbytes(), shape, dtype, nxp::default_device_name);
    }

    nb::ndarray<nb::pytorch> array_to_torch(nxc::Array &array) {
        switch (array.get_dtype()->get_name()) {
        case nxp::DtypeName::F32:
            return array_to_torch_impl<float>(array);
        case nxp::DtypeName::I32:
            return array_to_torch_impl<int>(array);
        default:
            return array_to_torch_impl<bool>(array);
        }
    }

    nb::object item(nxc::Array &array) {
        nxp::isize value = array.item();
        nxp::DtypePtr dtype = array.get_dtype();

        switch (dtype->get_name()) {
        case nxp::DtypeName::F32:
            return nb::cast<float>(std::bit_cast<float>(static_cast<int32_t>(value)));
        case nxp::DtypeName::I32:
            return nb::cast<int>(value);
        default:
            return nb::cast<bool>(value);
        }
    }

    nxc::Array full(const nxp::ShapeView &view, const nb::object &obj, nxp::DtypePtr dtype, const std::string &device_name) {
        if (nb::isinstance<nb::float_>(obj)) {
            return nxc::Array::full(view, nb::cast<float>(obj), dtype, device_name);
        } else if (nb::isinstance<nb::int_>(obj)) {
            return nxc::Array::full(view, nb::cast<int>(obj), dtype, device_name);
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return nxc::Array::full(view, nb::cast<bool>(obj), dtype, device_name);
        }
        throw nxp::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    nxc::Array full_like(const nxc::Array &other, const nb::object &obj, nxp::DtypePtr dtype, const std::string &device_name) {
        return full(other.get_view(), obj, dtype, device_name);
    }

    nxc::Array neg(const nxc::Array &array) {
        return array.neg();
    }

    nxc::Array add(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a + b; });
    }

    nxc::Array iadd(nxc::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a += b; });
    }

    nxc::Array sub(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a - b; });
    }

    nxc::Array isub(nxc::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a -= b; });
    }

    nxc::Array mul(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a * b; });
    }

    nxc::Array imul(nxc::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a *= b; });
    }

    nxc::Array div(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a / b; });
    }

    nxc::Array idiv(nxc::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a /= b; });
    }

    nxc::Array eq(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a == b; });
    }

    nxc::Array neq(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a != b; });
    }

    nxc::Array lt(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a < b; });
    }

    nxc::Array gt(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a > b; });
    }

    nxc::Array leq(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a <= b; });
    }

    nxc::Array geq(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a >= b; });
    }

    nxc::Array minimum(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a.minimum(b); });
    }

    nxc::Array maximum(const nxc::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a.maximum(b); });
    }

    nxc::Array slice(const nxc::Array &array, const nb::object &obj) {
        return array.slice(nxb::pyslices_to_ranges(array, obj));
    }

    nxc::Array permute(const nxc::Array &array, nx::primitive::ShapeDims &dims) {
        return array.permute(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array transpose(const nxc::Array &array, nxp::isize start_dim, nxp::isize end_dim) {
        return array.transpose(get_pyindex(array.get_shape().get_ndim(), start_dim), get_pyindex(array.get_shape().get_ndim(), end_dim));
    }

    nxc::Array flatten(const nxc::Array &array, nxp::isize start_dim, nxp::isize end_dim) {
        return array.flatten(get_pyindex(array.get_shape().get_ndim(), start_dim), get_pyindex(array.get_shape().get_ndim(), end_dim));
    }

    nxc::Array squeeze(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.squeeze(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array unsqueeze(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.unsqueeze(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array sum(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.sum(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array mean(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.mean(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array max(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.max(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array min(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.min(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array argmax(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.argmax(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxc::Array argmin(const nxc::Array &array, nxp::ShapeDims &dims) {
        return array.argmin(get_pyindices(array.get_shape().get_ndim(), dims));
    }
} // namespace nx::bind