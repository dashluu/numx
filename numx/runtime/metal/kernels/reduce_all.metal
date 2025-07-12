#include "reduce.h"

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_all(
    const constant isize &numel [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<R> *output [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    Op op;
    T default_val = op.template get_default<T>();
    T best_val = gid < numel ? input[offset[0] + gid] : default_val;
    threadgroup T ldata[simd_size];
    // Perform per-SIMD partial reduction -> shuffling within SIMD group.
    // Each thread gets the value from another thread offset lanes above it.
    best_val = simd_reduce(op, best_val);
    
    if (simd_per_group > 1) {
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = best_val;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        best_val = (lid < simd_per_group) ? ldata[lid] : default_val;
        // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
        best_val = simd_reduce(op, best_val);
    }
    
    if (lid == 0) {
        AtomicOp()(output + offset[1], best_val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void strided_reduce_all(
    const constant isize &numel [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize &ndim [[buffer(2)]],
    const constant isize *shape [[buffer(3)]],
    const constant isize *stride [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<R> *output [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    Op op;
    T default_val = op.template get_default<T>();
    T best_val = gid < numel ? input[offset[0] + get_elm_loc(gid, ndim, shape, stride)] : default_val;
    threadgroup T ldata[simd_size];
    // Perform per-SIMD partial reduction -> shuffling within SIMD group.
    // Each thread gets the value from another thread offset lanes above it.
    best_val = simd_reduce(op, best_val);
    
    if (simd_per_group > 1) {
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = best_val;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        best_val = (lid < simd_per_group) ? ldata[lid] : default_val;
        // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
        best_val = simd_reduce(op, best_val);
    }
    
    // Atomically update the reduction result.
    if (lid == 0) {
        AtomicOp()(output + offset[1], best_val);
    }
}

#define def_reduce_all_kernels(opname, op, atomic_op, dtype, T, R)      \
template [[host_name(#opname "_all_" #dtype)]] [[kernel]] decltype(reduce_all<op, atomic_op, T, R>) reduce_all<op, atomic_op, T, R>;                            \
template [[host_name("strided_" #opname "_all_" #dtype)]] [[kernel]] decltype(strided_reduce_all<op, atomic_op, T, R>) strided_reduce_all<op, atomic_op, T, R>;

#define def_reduce_all(opname, op, atomic_op_float, atomic_op_int)      \
def_reduce_all_kernels(opname, op, atomic_op_float, f32, float, float); \
def_reduce_all_kernels(opname, op, atomic_op_int, i32, int, int);

def_reduce_all(sum, Sum, AtomicSum, AtomicSum);
def_reduce_all(max, Max, AtomicMaxFloat, AtomicMaxInt);
def_reduce_all(min, Min, AtomicMinFloat, AtomicMinInt);