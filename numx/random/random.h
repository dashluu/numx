#pragma once

#include "../core/array.h"

namespace nx::random {
    using namespace nx::primitive;
    using namespace nx::core;

    inline void seed(uint64_t value) { s_seed = value; }

    template <NumericType T>
    Array uniform(const ShapeView &view, T low = T(0), T high = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device_by_name(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator_by_device_name(device_name);
        return Array(nx::graph::uniform(view, rand_key_gen, low, high, dtype, device));
    }

    template <NumericType T>
    Array normal(const ShapeView &view, T mean = T(0), T std = T(1), DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
        DevicePtr device = get_device_by_name(device_name);
        RandomKeyGeneratorPtr rand_key_gen = get_random_key_generator_by_device_name(device_name);
        return Array(nx::graph::normal(view, rand_key_gen, mean, std, dtype, device));
    }
} // namespace nx::random