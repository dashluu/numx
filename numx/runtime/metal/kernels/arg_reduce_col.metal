#include "reduce.h"

template <class Op, class AtomicOp, class T, uint tgrow, uint tgcol>
kernel void arg_reduce_col(
    const constant isize &ncol [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<uint> *output [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint grow = gid.y;
    const uint gcol = gid.x;
    const uint lrow = lid.y;
    const uint lcol = lid.x;
    const uint lwidth = lsize.x;
    Op op;
    T default_val = input[offset[0] + grow * ncol];
	IndexValPair<T> best_pair = gcol < ncol ? IndexValPair<T>{input[offset[0] + grow * ncol + gcol], gcol} : IndexValPair<T>{default_val, 0};
	threadgroup IndexValPair<T> ldata[tgrow][tgcol];
	best_pair = arg_simd_reduce(op, best_pair);
	uint simd_per_row_group = lwidth / simd_size;
	
	if (simd_per_row_group > 1) {
		// Write per-SIMD partial reduction value to threadgroup memory.
		if (simd_lane_id == 0) {
			ldata[lrow][lcol / simd_size] = best_pair;
		}
		// Wait for all partial reductions to complete.
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		best_pair = (lcol < simd_per_row_group) ? ldata[lrow][lcol] : IndexValPair<T>{default_val, 0};
		// Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
		best_pair = arg_simd_reduce(op, best_pair);
	}
    
    if (lcol == 0) {
    	AtomicOp()(input + offset[0] + grow * ncol, output + offset[1] + grow, best_pair);
    }
}

template <class Op, class AtomicOp, class T, uint tgrow, uint tgcol>
kernel void strided_arg_reduce_col(
    const constant isize &ncol [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
	const constant isize &ndim [[buffer(2)]],
    const constant isize *shape [[buffer(3)]],
	const constant isize *stride [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<uint> *output [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint grow = gid.y;
    const uint gcol = gid.x;
    const uint lrow = lid.y;
    const uint lcol = lid.x;
    const uint lwidth = lsize.x;
    Op op;
    T default_val = input[offset[0] + get_elm_loc(grow * ncol, ndim, shape, stride)];
	IndexValPair<T> best_pair = gcol < ncol ? IndexValPair<T>{input[offset[0] + get_elm_loc(grow * ncol + gcol, ndim, shape, stride)], gcol} : IndexValPair<T>{default_val, 0};
	threadgroup IndexValPair<T> ldata[tgrow][tgcol];
	best_pair = arg_simd_reduce(op, best_pair);
	uint simd_per_row_group = lwidth / simd_size;
	
	if (simd_per_row_group > 1) {
		// Write per-SIMD partial reduction value to threadgroup memory.
		if (simd_lane_id == 0) {
			ldata[lrow][lcol / simd_size] = best_pair;
		}
		// Wait for all partial reductions to complete.
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		best_pair = (lcol < simd_per_row_group) ? ldata[lrow][lcol] : IndexValPair<T>{default_val, 0};
		// Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
		best_pair = arg_simd_reduce(op, best_pair);
	}
    
    if (lcol == 0) {
    	AtomicOp()(input + offset[0], grow * ncol, ndim, shape, stride, output + offset[1] + grow, best_pair);
    }
}

#define def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, tgrow, tgcol)   \
template [[host_name(#opname "_col_" #tgrow "x" #tgcol "_" #dtype)]] [[kernel]] decltype(arg_reduce_col<op, atomic_op, T, tgrow, tgcol>) arg_reduce_col<op, atomic_op, T, tgrow, tgcol>;                            \
template [[host_name("strided_" #opname "_col_" #tgrow "x" #tgcol "_" #dtype)]] [[kernel]] decltype(strided_arg_reduce_col<op, atomic_op, T, tgrow, tgcol>) strided_arg_reduce_col<op, atomic_op, T, tgrow, tgcol>;

#define config_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T)              \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 1);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 2);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 4);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 8);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 16);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 1, 32);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 2, 1);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 2, 2);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 2, 4);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 2, 8);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 2, 16);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 4, 1);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 4, 2);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 4, 4);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 4, 8);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 8, 1);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 8, 2);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 8, 4);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 16, 1);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 16, 2);				    \
def_arg_reduce_col_kernels(opname, op, atomic_op, dtype, T, 32, 1);

#define def_arg_reduce_col(opname, op, atomic_op)								    \
config_arg_reduce_col_kernels(opname, op, atomic_op, f32, float);				    \
config_arg_reduce_col_kernels(opname, op, atomic_op, i32, int);

def_arg_reduce_col(argmax, Argmax, AtomicArgmax);
def_arg_reduce_col(argmin, Argmin, AtomicArgmin);