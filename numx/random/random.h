#pragma once

#include "../core/functional.h"

namespace nx::random {
    using namespace nx::primitive;
    using namespace nx::core;

    inline RandomKeyGeneratorPtr get_random_key_generator(const std::string &device_name) { return get_device_context(device_name)->get_random_key_generator(); }
    inline void seed(uint64_t value) { s_seed = value; }

    template <NumericType T>
    Array uniform(const ShapeView &view, T low = T(0), T high = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        if (!dtype->is_float()) {
            throw IncompatDtypeForRandomFunction("uniform", "float", dtype->str());
        }

        DevicePtr device = get_device(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator(device_name);
        return Array(nx::graph::uniform(view, rand_key_gen, low, high, dtype, device));
    }

    template <NumericType T>
    Array normal(const ShapeView &view, T mean = T(0), T std = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        if (!dtype->is_float()) {
            throw IncompatDtypeForRandomFunction("normal", "float", dtype->str());
        }

        DevicePtr device = get_device(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator(device_name);
        return Array(nx::graph::normal(view, rand_key_gen, mean, std, dtype, device));
    }

    inline Array kaiming_uniform(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        if (!dtype->is_float()) {
            throw IncompatDtypeForRandomFunction("kaiming_uniform", "float", dtype->str());
        }

        auto [fan_in, fan_out] = compute_fan_in_and_fan_out(view);
        float low = -std::sqrt(6.0f / fan_in);
        float high = std::sqrt(6.0f / fan_in);
        return uniform(view, low, high, dtype, device_name);
    }

    template <IntegerType T>
    Array randint(const ShapeView &view, T low = T(0), T high = T(10), DtypePtr dtype = &i32, const std::string &device_name = default_device_name) {
        if (!dtype->is_int()) {
            throw IncompatDtypeForRandomFunction("randint", "int", dtype->str());
        }

        DtypePtr float_dtype = float_dtype_by_dtype(dtype);
        return uniform(view, low, high, float_dtype, device_name).astype(dtype);
    }

    inline Array randbool(const ShapeView &view, const std::string &device_name = default_device_name) {
        // TODO: change i32 to something else?
        return randint<int>(view, 0, 2, &i32, device_name).astype(&b8);
    }
} // namespace nx::random