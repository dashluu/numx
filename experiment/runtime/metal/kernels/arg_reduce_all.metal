#include "reduce.h"

template <class Op, class AtomicOp, class T>
kernel void arg_reduce_all(
    const constant isize &numel [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<uint> *output [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    Op op;
    T default_val = input[offset[0]];
    IndexValPair<T> best_pair = gid < numel ? IndexValPair<T>{input[offset[0] + gid], gid} : IndexValPair<T>{default_val, 0};
    threadgroup IndexValPair<T> ldata[simd_size];
    // Perform per-SIMD partial reduction -> shuffling within SIMD group.
    // Each thread gets the value from another thread offset lanes above it.
    best_pair = arg_simd_reduce(op, best_pair);
    
    if (simd_per_group > 1) {
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = best_pair;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        best_pair = (lid < simd_per_group) ? ldata[lid] : IndexValPair<T>{default_val, 0};
        // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
        best_pair = arg_simd_reduce(op, best_pair);
    }
    
    if (lid == 0) {
        AtomicOp()(input + offset[0], output + offset[1], best_pair);
    }
}

template <class Op, class AtomicOp, class T>
kernel void strided_arg_reduce_all(
    const constant isize &numel [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize &ndim [[buffer(2)]],
    const constant isize *shape [[buffer(3)]],
    const constant isize *stride [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<uint> *output [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    Op op;
    T default_val = input[offset[0]];
    IndexValPair<T> best_pair = gid < numel ? IndexValPair<T>{input[offset[0] + get_elm_loc(gid, ndim, shape, stride)], gid} : IndexValPair<T>{default_val, 0};
    threadgroup IndexValPair<T> ldata[simd_size];
    // Perform per-SIMD partial reduction -> shuffling within SIMD group.
    // Each thread gets the value from another thread offset lanes above it.
    best_pair = arg_simd_reduce(op, best_pair);
    
    if (simd_per_group > 1) {
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = best_pair;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        best_pair = (lid < simd_per_group) ? ldata[lid] : IndexValPair<T>{default_val, 0};
        // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
        best_pair = arg_simd_reduce(op, best_pair);
    }
    
    if (lid == 0) {
        AtomicOp()(input + offset[0], output + offset[1], best_pair);
    }
}

#define def_arg_reduce_all_kernels(opname, op, atomic_op, dtype, T) \
template [[host_name(#opname "_all_" #dtype)]] [[kernel]] decltype(arg_reduce_all<op, atomic_op, T>) arg_reduce_all<op, atomic_op, T>;                              \
template [[host_name("strided_" #opname "_all_" #dtype)]] [[kernel]] decltype(strided_arg_reduce_all<op, atomic_op, T>) strided_arg_reduce_all<op, atomic_op, T>;

#define def_arg_reduce_all(opname, op, atomic_op)                   \
def_arg_reduce_all_kernels(opname, op, atomic_op, f32, float);      \
def_arg_reduce_all_kernels(opname, op, atomic_op, i32, int);

def_arg_reduce_all(argmax, Argmax, AtomicArgmax);
def_arg_reduce_all(argmin, Argmin, AtomicArgmin);