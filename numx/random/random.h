#pragma once

#include "../core/functional.h"

namespace nx::random {
    using namespace nx::primitive;
    using namespace nx::core;

    inline RandomKeyGeneratorPtr get_random_key_generator(const std::string &device_name) { return get_device_context(device_name)->get_random_key_generator(); }
    inline void seed(uint64_t value) { s_seed = value; }

    template <NumericType T>
    Array uniform(const ShapeView &view, T low = T(0), T high = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator(device_name);
        return Array(nx::graph::uniform(view, rand_key_gen, low, high, dtype, device));
    }

    template <NumericType T>
    Array normal(const ShapeView &view, T mean = T(0), T std = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator(device_name);
        return Array(nx::graph::normal(view, rand_key_gen, mean, std, dtype, device));
    }

    inline Array kaiming_uniform(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        auto [fan_in, fan_out] = compute_fan_in_and_fan_out(view);
        float low = -std::sqrt(6.0f / fan_in);
        float high = std::sqrt(6.0f / fan_in);
        return uniform(view, low, high, dtype, device_name);
    }
} // namespace nx::random