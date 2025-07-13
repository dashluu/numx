#include "array.h"

namespace nx::bind {
    nxc::isize get_pyindex(nxc::isize len, nxc::isize idx) {
        if (idx < -len || idx >= len) {
            throw nxc::OutOfRange(idx, -len, len);
        }
        return idx < 0 ? idx + len : idx;
    }

    nxc::ShapeDims get_pyindices(nxc::isize len, nxc::ShapeDims &dims) {
        if (!dims.empty()) {
            std::transform(dims.begin(), dims.end(), dims.begin(), [len](auto dim) { return get_pyindex(len, dim); });
        }
        return dims;
    }

    nxc::Range pyslice_to_range(nxc::isize len, const nb::object &obj) {
        // Note: no need to check for out-of-bounds indices when converting to range
        // Shape does the checking eventually
        if (!nb::isinstance<nb::slice>(obj)) {
            throw nxc::NanobindInvalidArgumentType(get_pyclass(obj), "slice");
        }

        auto slice = nb::cast<nb::slice>(obj);
        bool start_none = slice.attr("start").is_none();
        bool stop_none = slice.attr("stop").is_none();
        bool step_none = slice.attr("step").is_none();
        nxc::isize start, stop, step;

        if (step_none) {
            start = start_none ? 0 : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("start")));
            stop = stop_none ? len : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("stop")));
            return nxc::Range(start, stop, 1);
        }

        step = nb::cast<nxc::isize>(slice.attr("step"));

        if (step > 0) {
            start = start_none ? 0 : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("start")));
            stop = stop_none ? len : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("stop")));
        } else {
            start = start_none ? len - 1 : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("start")));
            stop = stop_none ? -1 : get_pyindex(len, nb::cast<nxc::isize>(slice.attr("stop")));
        }

        return nxc::Range(start, stop, step);
    }

    std::vector<nxc::Range> pyslices_to_ranges(const nxa::Array &array, const nb::object &obj) {
        std::vector<nxc::Range> ranges;
        const nxc::Shape &shape = array.get_shape();

        // obj can be an int, a slice, or a sequence of ints or slices
        if (nb::isinstance<nb::int_>(obj)) {
            nxc::isize idx = get_pyindex(shape[0], nb::cast<nxc::isize>(obj));
            ranges.emplace_back(idx, idx + 1, 1);

            for (nxc::isize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::slice>(obj)) {
            ranges.push_back(pyslice_to_range(shape[0], obj));

            for (nxc::isize i = 1; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        } else if (nb::isinstance<nb::sequence>(obj) && !nb::isinstance<nb::str>(obj)) {
            // Object is a sequence but not a string
            auto sequence = nb::cast<nb::sequence>(obj);
            size_t seq_len = nb::len(sequence);

            if (seq_len > shape.get_ndim()) {
                throw nxc::OutOfRange(seq_len, 1, shape.get_ndim() + 1);
            }

            for (size_t i = 0; i < seq_len; i++) {
                auto elm = sequence[i];
                // elm must be a sequence of ints or slices
                if (nb::isinstance<nb::int_>(elm)) {
                    nxc::isize idx = get_pyindex(shape[i], nb::cast<nxc::isize>(elm));
                    ranges.emplace_back(idx, idx + 1, 1);
                } else if (nb::isinstance<nb::slice>(elm)) {
                    ranges.push_back(pyslice_to_range(shape[i], elm));
                } else {
                    throw nxc::NanobindInvalidArgumentType(get_pyclass(elm), "int, slice");
                }
            }

            for (nxc::isize i = seq_len; i < shape.get_ndim(); i++) {
                ranges.emplace_back(0, shape[i], 1);
            }

            return ranges;
        }

        throw nxc::NanobindInvalidArgumentType(get_pyclass(obj), "int, slice, sequence");
    }

    nxc::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype) {
        if (nb_dtype == nb::dtype<float>()) {
            return &nxc::f32;
        } else if (nb_dtype == nb::dtype<int>()) {
            return &nxc::i32;
        } else if (nb_dtype == nb::dtype<bool>()) {
            return &nxc::b8;
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

    nb::ndarray<nb::numpy> array_to_numpy(nxa::Array &array) {
        switch (array.get_dtype()->get_name()) {
        case nxc::DtypeName::F32:
            return array_to_numpy_impl<float>(array);
        case nxc::DtypeName::I32:
            return array_to_numpy_impl<int>(array);
        default:
            return array_to_numpy_impl<bool>(array);
        }
    }

    nxa::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr) {
        nxc::ShapeView view;
        nxc::ShapeStride stride;

        for (size_t i = 0; i < ndarr.ndim(); ++i) {
            view.push_back(ndarr.shape(i));
            stride.push_back(ndarr.stride(i));
        }

        nxc::Shape shape(0, view, stride);
        uint8_t *ptr = reinterpret_cast<uint8_t *>(ndarr.data());
        nxc::DtypePtr dtype = dtype_from_nb_dtype(ndarr.dtype());
        return nxa::Array::from_buffer(ptr, ndarr.nbytes(), shape, dtype, nxc::default_device_name);
    }

    nb::ndarray<nb::pytorch> array_to_torch(nxa::Array &array) {
        switch (array.get_dtype()->get_name()) {
        case nxc::DtypeName::F32:
            return array_to_torch_impl<float>(array);
        case nxc::DtypeName::I32:
            return array_to_torch_impl<int>(array);
        default:
            return array_to_torch_impl<bool>(array);
        }
    }

    nb::object item(nxa::Array &array) {
        nxc::isize value = array.item();
        nxc::DtypePtr dtype = array.get_dtype();

        switch (dtype->get_name()) {
        case nxc::DtypeName::F32:
            return nb::cast<float>(std::bit_cast<float>(static_cast<int32_t>(value)));
        case nxc::DtypeName::I32:
            return nb::cast<int>(value);
        default:
            return nb::cast<bool>(value);
        }
    }

    nxa::Array full(const nxc::ShapeView &view, const nb::object &obj, nxc::DtypePtr dtype, const std::string &device_name) {
        if (nb::isinstance<nb::float_>(obj)) {
            return nxa::Array::full(view, nb::cast<float>(obj), dtype, device_name);
        } else if (nb::isinstance<nb::int_>(obj)) {
            return nxa::Array::full(view, nb::cast<int>(obj), dtype, device_name);
        } else if (nb::isinstance<nb::bool_>(obj)) {
            return nxa::Array::full(view, nb::cast<bool>(obj), dtype, device_name);
        }
        throw nxc::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
    }

    nxa::Array full_like(const nxa::Array &other, const nb::object &obj, nxc::DtypePtr dtype, const std::string &device_name) {
        return full(other.get_view(), obj, dtype, device_name);
    }

    nxa::Array neg(const nxa::Array &array) {
        return array.neg();
    }

    nxa::Array add(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a + b; });
    }

    nxa::Array iadd(nxa::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a += b; });
    }

    nxa::Array sub(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a - b; });
    }

    nxa::Array isub(nxa::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a -= b; });
    }

    nxa::Array mul(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a * b; });
    }

    nxa::Array imul(nxa::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a *= b; });
    }

    nxa::Array div(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a / b; });
    }

    nxa::Array idiv(nxa::Array &array, const nb::object &obj) {
        return in_place_binary(array, obj, [](auto &a, const auto &b) { return a /= b; });
    }

    nxa::Array eq(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a == b; });
    }

    nxa::Array neq(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a != b; });
    }

    nxa::Array lt(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a < b; });
    }

    nxa::Array gt(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a > b; });
    }

    nxa::Array leq(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a <= b; });
    }

    nxa::Array geq(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a >= b; });
    }

    nxa::Array minimum(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a.minimum(b); });
    }

    nxa::Array maximum(const nxa::Array &array, const nb::object &obj) {
        return binary(array, obj, [](const auto &a, const auto &b) { return a.maximum(b); });
    }

    nxa::Array slice(const nxa::Array &array, const nb::object &obj) {
        return array.slice(nxb::pyslices_to_ranges(array, obj));
    }

    nxa::Array permute(const nxa::Array &array, nx::core::ShapeDims &dims) {
        return array.permute(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array transpose(const nxa::Array &array, nxc::isize start_dim, nxc::isize end_dim) {
        return array.transpose(get_pyindex(array.get_shape().get_ndim(), start_dim), get_pyindex(array.get_shape().get_ndim(), end_dim));
    }

    nxa::Array flatten(const nxa::Array &array, nxc::isize start_dim, nxc::isize end_dim) {
        return array.flatten(get_pyindex(array.get_shape().get_ndim(), start_dim), get_pyindex(array.get_shape().get_ndim(), end_dim));
    }

    nxa::Array squeeze(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.squeeze(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array unsqueeze(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.unsqueeze(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array sum(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.sum(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array mean(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.mean(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array max(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.max(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array min(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.min(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array argmax(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.argmax(get_pyindices(array.get_shape().get_ndim(), dims));
    }

    nxa::Array argmin(const nxa::Array &array, nxc::ShapeDims &dims) {
        return array.argmin(get_pyindices(array.get_shape().get_ndim(), dims));
    }
} // namespace nx::bind