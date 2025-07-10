#include "utils.h"

template <class T, class R>
kernel void naive_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *lshape [[buffer(1)]],
    const constant isize *rshape [[buffer(2)]],
    const device T *lhs [[buffer(3)]],
    const device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint row = id.y, col = id.x;
    const isize M = lshape[0], K = lshape[1], N = rshape[1];
    
    if (row < M && col < N) {
        R sum = 0;
        for (isize i = 0; i < K; i++) {
            sum += lhs[offset[0] + row * K + i] * rhs[offset[1] + N * i + col];
        }
        output[offset[2] + row * N + col] = sum;
    }
}

template <class T, class R>
kernel void strided_naive_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *lshape [[buffer(1)]],
    const constant isize *rshape [[buffer(2)]],
    const constant isize *lstride [[buffer(3)]],
    const constant isize *rstride [[buffer(4)]],
    const device T *lhs [[buffer(5)]],
    const device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint row = id.y, col = id.x;
    const isize M = lshape[0], K = lshape[1], N = rshape[1];
    
    if (row < M && col < N) {
        R sum = 0;
        
        for (isize i = 0; i < K; i++) {
            const isize lloc = offset[0] + get_elm_loc(row * K + i, 2, lshape, lstride);
            const isize rloc = offset[1] + get_elm_loc(N * i + col, 2, rshape, rstride);
            sum += lhs[lloc] * rhs[rloc];
        }
        
        output[offset[2] + row * N + col] = sum;
    }
}

template <class T, class R>
kernel void naive_gemm3d(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *lshape [[buffer(2)]],
    const constant isize *rshape [[buffer(3)]],
    const device T *lhs [[buffer(4)]],
    const device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z, row = id.y, col = id.x;
    const isize M = lshape[ndim - 2], K = lshape[ndim - 1], N = rshape[ndim - 1];
    isize B = 1;
    
    for (isize i = 0; i < ndim - 2; i++) {
        B *= lshape[i];
    }
    
    if (batch < B && row < M && col < N) {
        R sum = 0;
        for (isize i = 0; i < K; i++) {
            sum += lhs[offset[0] + batch * M * K + row * K + i] * rhs[offset[1] + batch * K * N + N * i + col];
        }
        output[offset[2] + batch * M * N + row * N + col] = sum;
    }
}

template <class T, class R>
kernel void strided_naive_gemm3d(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *lshape [[buffer(2)]],
    const constant isize *rshape [[buffer(3)]],
    const constant isize *lstride [[buffer(4)]],
    const constant isize *rstride [[buffer(5)]],
    const device T *lhs [[buffer(6)]],
    const device T *rhs [[buffer(7)]],
    device R *output [[buffer(8)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z, row = id.y, col = id.x;
    const isize M = lshape[ndim - 2], K = lshape[ndim - 1], N = rshape[ndim - 1];
    isize B = 1;
    
    for (isize i = 0; i < ndim - 2; i++) {
        B *= lshape[i];
    }
    
    if (batch < B && row < M && col < N) {
        R sum = 0;
        
        for (isize i = 0; i < K; i++) {
            const isize lloc = offset[0] + get_elm_loc(batch * M * K + row * K + i, ndim, lshape, lstride);
            const isize rloc = offset[1] + get_elm_loc(batch * K * N + N * i + col, ndim, rshape, rstride);
            sum += lhs[lloc] * rhs[rloc];
        }
        
        output[offset[2] + batch * M * N + row * N + col] = sum;
    }
}

#define def_naive_gemm(dtype, T, R) \
template [[host_name("naive_gemm2d_" #dtype)]] [[kernel]] decltype(naive_gemm2d<T, R>) naive_gemm2d<T, R>;                            \
template [[host_name("naive_gemm3d_" #dtype)]] [[kernel]] decltype(naive_gemm3d<T, R>) naive_gemm3d<T, R>;                            \
template [[host_name("strided_naive_gemm2d_" #dtype)]] [[kernel]] decltype(strided_naive_gemm2d<T, R>) strided_naive_gemm2d<T, R>;    \
template [[host_name("strided_naive_gemm3d_" #dtype)]] [[kernel]] decltype(strided_naive_gemm3d<T, R>) strided_naive_gemm3d<T, R>;

def_naive_gemm(f32, float, float);
def_naive_gemm(i32, int, int);