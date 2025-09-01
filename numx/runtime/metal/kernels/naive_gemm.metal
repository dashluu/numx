#include "utils.h"

template <class T, class R>
kernel void naive_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *l_shape [[buffer(1)]],
    const constant isize *r_shape [[buffer(2)]],
    const device T *lhs [[buffer(3)]],
    const device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint row = id.y, col = id.x;
    const isize M = l_shape[0], K = l_shape[1], N = r_shape[1];
    
    if (row < M && col < N) {
        R sum = 0;
        isize l_loc, r_loc, out_loc = offset[2] + row * N + col;
        
        for (isize i = 0; i < K; i++) {
            l_loc = offset[0] + row * K + i;
            r_loc = offset[1] + N * i + col;
            sum += lhs[l_loc] * rhs[r_loc];
        }
        
        output[out_loc] = sum;
    }
}

template <class T, class R>
kernel void strided_naive_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *l_shape [[buffer(1)]],
    const constant isize *r_shape [[buffer(2)]],
    const constant isize *l_stride [[buffer(3)]],
    const constant isize *r_stride [[buffer(4)]],
    const device T *lhs [[buffer(5)]],
    const device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint row = id.y, col = id.x;
    const isize M = l_shape[0], K = l_shape[1], N = r_shape[1];
    
    if (row < M && col < N) {
        R sum = 0;
        isize l_loc, r_loc, out_loc = offset[2] + row * N + col;
        
        for (isize i = 0; i < K; i++) {
            l_loc = offset[0] + get_elm_loc(row * K + i, 2, l_shape, l_stride);
            r_loc = offset[1] + get_elm_loc(N * i + col, 2, r_shape, r_stride);
            sum += lhs[l_loc] * rhs[r_loc];
        }
        
        output[out_loc] = sum;
    }
}

template <class T, class R>
kernel void naive_gemm3d(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *l_shape [[buffer(2)]],
    const constant isize *r_shape [[buffer(3)]],
    const device T *lhs [[buffer(4)]],
    const device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z, row = id.y, col = id.x;
    const isize M = l_shape[ndim - 2], K = l_shape[ndim - 1], N = r_shape[ndim - 1];
    isize B = 1;
    
    for (isize i = 0; i < ndim - 2; i++) {
        B *= l_shape[i];
    }
    
    if (batch < B && row < M && col < N) {
        R sum = 0;
        isize l_loc, r_loc, out_loc = offset[2] + batch * M * N + row * N + col;
        
        for (isize i = 0; i < K; i++) {
            l_loc = offset[0] + batch * M * K + row * K + i;
            r_loc = offset[1] + batch * K * N + N * i + col;
            sum += lhs[l_loc] * rhs[r_loc];
        }
        
        output[out_loc] = sum;
    }
}

template <class T, class R>
kernel void strided_naive_gemm3d(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *l_shape [[buffer(2)]],
    const constant isize *r_shape [[buffer(3)]],
    const constant isize *l_stride [[buffer(4)]],
    const constant isize *r_stride [[buffer(5)]],
    const device T *lhs [[buffer(6)]],
    const device T *rhs [[buffer(7)]],
    device R *output [[buffer(8)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z, row = id.y, col = id.x;
    const isize M = l_shape[ndim - 2], K = l_shape[ndim - 1], N = r_shape[ndim - 1];
    isize B = 1;
    
    for (isize i = 0; i < ndim - 2; i++) {
        B *= l_shape[i];
    }
    
    if (batch < B && row < M && col < N) {
        R sum = 0;
        isize l_loc, r_loc, out_loc = offset[2] + batch * M * N + row * N + col;
        
        for (isize i = 0; i < K; i++) {
            l_loc = offset[0] + get_elm_loc(batch * M * K + row * K + i, ndim, l_shape, l_stride);
            r_loc = offset[1] + get_elm_loc(batch * K * N + N * i + col, ndim, r_shape, r_stride);
            sum += lhs[l_loc] * rhs[r_loc];
        }
        
        output[out_loc] = sum;
    }
}

#define def_naive_gemm(dtype, T, R) \
template [[host_name("naive_gemm2d_" #dtype)]] [[kernel]] decltype(naive_gemm2d<T, R>) naive_gemm2d<T, R>;                            \
template [[host_name("naive_gemm3d_" #dtype)]] [[kernel]] decltype(naive_gemm3d<T, R>) naive_gemm3d<T, R>;                            \
template [[host_name("strided_naive_gemm2d_" #dtype)]] [[kernel]] decltype(strided_naive_gemm2d<T, R>) strided_naive_gemm2d<T, R>;    \
template [[host_name("strided_naive_gemm3d_" #dtype)]] [[kernel]] decltype(strided_naive_gemm3d<T, R>) strided_naive_gemm3d<T, R>;

def_naive_gemm(f32, float, float);
def_naive_gemm(i32, int, int);