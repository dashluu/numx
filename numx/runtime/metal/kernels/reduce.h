#include "utils.h"

template <class T>
struct IndexValPair {
    T val;
    // metal::simd_shuffle_down does not support isize so uint is used
    uint idx;
};

template <class Op, class T>
T simd_reduce(thread Op &op, T val) {
    for (uint lanes = simd_size / 2; lanes > 0; lanes /= 2) {
        // Shuffle down does not affect the upper lanes
        // Since we start from simd_size / 2, the upper half of simd_size / 2 elements are not affected
        val = op(val, metal::simd_shuffle_down(val, lanes));
    }
    return val;
}

template <class Op, class T>
IndexValPair<T> arg_simd_reduce(thread Op &op, IndexValPair<T> pair) {
    for (uint lanes = simd_size / 2; lanes > 0; lanes /= 2) {
        pair = op(pair, IndexValPair<T>{metal::simd_shuffle_down(pair.val, lanes), metal::simd_shuffle_down(pair.idx, lanes)});
    }
    return pair;
}

struct Sum {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }

    template <class T>
    T get_default() { return 0; }
};

struct Max {
    template <class T>
    T operator()(T lhs, T rhs) { return metal::max(lhs, rhs); }

    template <class T>
    T get_default() { return Limits<T>::min(); }
};

struct Min {
    template <class T>
    T operator()(T lhs, T rhs) { return metal::min(lhs, rhs); }

    template <class T>
    T get_default() { return Limits<T>::max(); }
};

struct AtomicSum {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val) {
        // memory_order_relaxed guarantees atomicity without ordering or proper synchronization
        // since we're doing addition, this is somewhat similar to a counter
        // atomic_fetch_add_explicit runs output += val but atomically
        metal::atomic_fetch_add_explicit(output, val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxInt {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val) {
        metal::atomic_fetch_max_explicit(output, new_val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxFloat {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val) {
        // CAS algorithm
        // output = max(output, val)
        R old_val = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        do {
            if (old_val >= new_val) {
                break;
            }
            // old_val gets updated by metal::atomic_compare_exchange_weak_explicit if the operation fails
            // No need for old_val to be in the while loop
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};

struct AtomicMinInt {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val) {
        metal::atomic_fetch_min_explicit(output, new_val, metal::memory_order_relaxed);
    }
};

struct AtomicMinFloat {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val) {
        // CAS algorithm
        // output = min(output, val)
        R old_val = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        do {
            if (old_val <= new_val) {
                break;
            }
            // old_val gets updated by metal::atomic_compare_exchange_weak_explicit if the operation fails
            // No need for old_val to be in the while loop
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};

struct Argmax {
    template <class T>
    IndexValPair<T> operator()(IndexValPair<T> lhs, IndexValPair<T> rhs) {
        IndexValPair<T> res;
        if (lhs.val >= rhs.val) {
            res.val = lhs.val;
            res.idx = lhs.idx;
        } else {
            res.val = rhs.val;
            res.idx = rhs.idx;
        }
        return res;
    }
};

struct Argmin {
    template <class T>
    IndexValPair<T> operator()(IndexValPair<T> lhs, IndexValPair<T> rhs) {
        IndexValPair<T> res;
        if (lhs.val <= rhs.val) {
            res.val = lhs.val;
            res.idx = lhs.idx;
        } else {
            res.val = rhs.val;
            res.idx = rhs.idx;
        }
        return res;
    }
};

struct AtomicArgmax {
    template <class T>
    void operator()(const device T *input, volatile device metal::_atomic<uint> *output, thread IndexValPair<T> &new_pair) {
        uint old_idx = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        T old_val;

        do {
            old_val = input[old_idx];
            if (old_val >= new_pair.val) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_idx, new_pair.idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }

    template <class T>
    void operator()(const device T *input, const isize row_idx, const isize ndim, const constant isize *shape, const constant isize *stride, volatile device metal::_atomic<uint> *output, thread IndexValPair<T> &new_pair) {
        uint col_idx = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        isize old_loc;
        T old_val;

        do {
            old_loc = get_elm_loc(row_idx + col_idx, ndim, shape, stride);
            old_val = input[old_loc];
            if (old_val >= new_pair.val) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &col_idx, new_pair.idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};

struct AtomicArgmin {
    template <class T>
    void operator()(const device T *input, volatile device metal::_atomic<uint> *output, thread IndexValPair<T> &new_pair) {
        uint old_idx = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        T old_val;

        do {
            old_val = input[old_idx];
            if (old_val <= new_pair.val) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_idx, new_pair.idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }

    template <class T>
    void operator()(const device T *input, const isize row_idx, const isize ndim, const constant isize *shape, const constant isize *stride, volatile device metal::_atomic<uint> *output, thread IndexValPair<T> &new_pair) {
        uint col_idx = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        isize old_loc;
        T old_val;

        do {
            old_loc = get_elm_loc(row_idx + col_idx, ndim, shape, stride);
            old_val = input[old_loc];
            if (old_val <= new_pair.val) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &col_idx, new_pair.idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};