#include "reduce.h"

template <class Op, class AtomicOp, class T, class R, uint tgrow, uint tgcol>
kernel void reduce_col(
    const constant isize &ncol [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<R> *output [[buffer(3)]],
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
    T default_val = op.template get_default<T>();
    T best_val = gcol < ncol ? input[offset[0] + grow * ncol + gcol] : default_val;
	threadgroup T ldata[tgrow][tgcol];
	best_val = simd_reduce(op, best_val);
	uint simd_per_row_group = lwidth / simd_size;
	
	if (simd_per_row_group > 1) {
		// Write per-SIMD partial reduction value to threadgroup memory.
		if (simd_lane_id == 0) {
			ldata[lrow][lcol / simd_size] = best_val;
		}
		// Wait for all partial reductions to complete.
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		best_val = (lcol < simd_per_row_group) ? ldata[lrow][lcol] : default_val;
		// Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
		best_val = simd_reduce(op, best_val);
	}
    
    if (lcol == 0) {
		AtomicOp()(output + offset[1] + grow, best_val);
    }
}

template <class Op, class AtomicOp, class T, class R, uint tgrow, uint tgcol>
kernel void strided_reduce_col(
	const constant isize &ncol [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
	const constant isize &ndim [[buffer(2)]],
    const constant isize *shape [[buffer(3)]],
	const constant isize *stride [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<R> *output [[buffer(6)]],
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
    T default_val = op.template get_default<T>();
    T best_val = gcol < ncol ? input[offset[0] + get_elm_loc(grow * ncol + gcol, ndim, shape, stride)] : default_val;
	threadgroup T ldata[tgrow][tgcol];
	best_val = simd_reduce(op, best_val);
	uint simd_per_row_group = lwidth / simd_size;
	
	if (simd_per_row_group > 1) {
		// Write per-SIMD partial reduction value to threadgroup memory.
		if (simd_lane_id == 0) {
			ldata[lrow][lcol / simd_size] = best_val;
		}
		// Wait for all partial reductions to complete.
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		best_val = (lcol < simd_per_row_group) ? ldata[lrow][lcol] : default_val;
		// Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
		best_val = simd_reduce(op, best_val);
	}
    
    if (lcol == 0) {
    	AtomicOp()(output + offset[1] + grow, best_val);
    }
}

#define def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, tgrow, tgcol)    \
template [[host_name(#opname "_col_" #tgrow "x" #tgcol "_" #dtype)]] [[kernel]] decltype(reduce_col<op, atomic_op, T, R, tgrow, tgcol>) reduce_col<op, atomic_op, T, R, tgrow, tgcol>;                              \
template [[host_name("strided_" #opname "_col_" #tgrow "x" #tgcol "_" #dtype)]] [[kernel]] decltype(strided_reduce_col<op, atomic_op, T, R, tgrow, tgcol>) strided_reduce_col<op, atomic_op, T, R, tgrow, tgcol>;

#define config_reduce_col_kernels(opname, op, atomic_op, dtype, T, R)               \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 1);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 2);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 4);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 8);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 16);                  \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 1, 32);                  \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 2, 1);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 2, 2);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 2, 4);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 2, 8);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 2, 16);                  \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 4, 1);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 4, 2);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 4, 4);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 4, 8);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 8, 1);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 8, 2);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 8, 4);                   \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 16, 1);                  \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 16, 2);                  \
def_reduce_col_kernels(opname, op, atomic_op, dtype, T, R, 32, 1);

#define def_reduce_col(opname, op, atomic_op_float, atomic_op_int)                  \
config_reduce_col_kernels(opname, op, atomic_op_float, f32, float, float);          \
config_reduce_col_kernels(opname, op, atomic_op_int, i32, int, int);

def_reduce_col(sum, Sum, AtomicSum, AtomicSum);
def_reduce_col(max, Max, AtomicMaxFloat, AtomicMaxInt);
def_reduce_col(min, Min, AtomicMinFloat, AtomicMinInt);