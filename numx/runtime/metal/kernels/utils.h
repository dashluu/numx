#pragma once

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

typedef int64_t isize;
typedef uint8_t ubyte;

constexpr constant uint simd_size = 32;

inline isize get_elm_loc(const uint id, const isize ndim, const constant isize *shape, const constant isize *stride) {
    isize carry = id;
    isize loc = 0;

    for (isize i = ndim - 1; i >= 0; i--) {
        loc += (carry % shape[i]) * stride[i];
        carry /= shape[i];
    }

    return loc;
}

template <class T>
struct Limits {
    static T finite_min() { return metal::numeric_limits<T>::min(); }
    static T finite_max() { return metal::numeric_limits<T>::max(); }
    static T min() { return metal::numeric_limits<T>::has_infinity ? -metal::numeric_limits<T>::infinity() : finite_min(); }
    static T max() { return metal::numeric_limits<T>::has_infinity ? metal::numeric_limits<T>::infinity() : finite_max(); }
};