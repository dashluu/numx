#include "utils.h"

template <class T>
void reset_tile(thread T &tile) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            tile[j][k] = 0.0f;
        }
    }
}

template <class T, class HxW>
void load_ltile2d(const device T* lhs, thread HxW &ltile, const isize row, const isize i, const isize K, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            ltile[k][j] = metal::select(0.0f, lhs[(row + j) * K + i + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_ltile2d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* lhs, thread HxW &ltile, const isize row, const isize i, const isize K, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            ltile[k][j] = metal::select(0.0f, lhs[get_elm_loc((row + j) * K + i + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_ltile3d(const device T* lhs, thread HxW &ltile, const isize batch, const isize row, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            ltile[k][j] = metal::select(0.0f, lhs[batch * M * K + (row + j) * K + i + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_ltile3d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* lhs, thread HxW &ltile, const isize batch, const isize row, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            ltile[k][j] = metal::select(0.0f, lhs[get_elm_loc(batch * M * K + (row + j) * K + i + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_rtile2d(const device T* rhs, thread HxW &rtile, const isize col, const isize i, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            rtile[k][j] = metal::select(0.0f, rhs[(i + j) * N + col + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_rtile3d(const device T* rhs, thread HxW &rtile, const isize batch, const isize col, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            rtile[k][j] = metal::select(0.0f, rhs[batch * K * N + (i + j) * N + col + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_rtile2d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* rhs, thread HxW &rtile, const isize col, const isize i, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            rtile[k][j] = metal::select(0.0f, rhs[get_elm_loc((i + j) * N + col + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_rtile3d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* rhs, thread HxW &rtile, const isize batch,  const isize col, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            rtile[k][j] = metal::select(0.0f, rhs[get_elm_loc(batch * K * N + (i + j) * N + col + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void store_otile2d(device T* output, thread HxW &otile, const isize row, const isize col, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < tile_height; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < tile_width; ++k) {
            output[(row + j) * N + col + k] = otile[k][j];
        }
    }
}

template <class T, class HxW>
void store_otile3d(device T* output, thread HxW &otile, const isize batch, const isize row, const isize col, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < tile_height; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < tile_width; ++k) {
            output[batch * M * N + (row + j) * N + col + k] = otile[k][j];
        }
    }
}

template <class T, class R, class HxW>
void tiled_gemm2d_HxW(
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW ltile, rtile, otile;
    isize i, K4 = K / 4 * 4;
    reset_tile(otile);

    for (i = 0; i < K4; i += 4) {
        load_ltile2d(lhs, ltile, row, i, K, tile_height, 4);
        load_rtile2d(rhs, rtile, col, i, N, 4, tile_width);
        otile += ltile * rtile;
    }

    if (K - K4 > 0) {
        load_ltile2d(lhs, ltile, row, i, K, tile_height, K - K4);
        load_rtile2d(rhs, rtile, col, i, N, K - K4, tile_width);
        otile += ltile * rtile;
    }

    store_otile2d(output, otile, row, col, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void tiled_gemm3d_HxW(
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW ltile, rtile, otile;
    isize i, K4 = K / 4 * 4;
    reset_tile(otile);

    for (i = 0; i < K4; i += 4) {
        load_ltile3d(lhs, ltile, batch, row, i, M, K, N, tile_height, 4);
        load_rtile3d(rhs, rtile, batch, col, i, M, K, N, 4, tile_width);
        otile += ltile * rtile;
    }

    if (K - K4 > 0) {
        load_ltile3d(lhs, ltile, batch, row, i, M, K, N, tile_height, K - K4);
        load_rtile3d(rhs, rtile, batch, col, i, M, K, N, K - K4, tile_width);
        otile += ltile * rtile;
    }

    store_otile3d(output, otile, batch, row, col, M, K, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void strided_tiled_gemm2d_HxW(
    const isize ndim,
    const constant isize *lshape, const constant isize *rshape,
    const constant isize *lstride, const constant isize *rstride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW ltile, rtile, otile;
    isize i, K4 = K / 4 * 4;
    reset_tile(otile);

    for (i = 0; i < K4; i += 4) {
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile, row, i, K, tile_height, 4);
        load_strided_rtile2d(ndim, rshape, rstride, rhs, rtile, col, i, N, 4, tile_width);
        otile += ltile * rtile;
    }

    if (K - K4 > 0) {
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile, row, i, K, tile_height, K - K4);
        load_strided_rtile2d(ndim, rshape, rstride, rhs, rtile, col, i, N, K - K4, tile_width);
        otile += ltile * rtile;
    }

    store_otile2d(output, otile, row, col, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void strided_tiled_gemm3d_HxW(
    const isize ndim,
    const constant isize *lshape, const constant isize *rshape,
    const constant isize *lstride, const constant isize *rstride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW ltile, rtile, otile;
    isize i, K4 = K / 4 * 4;
    reset_tile(otile);

    for (i = 0; i < K4; i += 4) {
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile, batch, row, i, M, K, N, tile_height, 4);
        load_strided_rtile3d(ndim, rshape, rstride, rhs, rtile, batch, col, i, M, K, N, 4, tile_width);
        otile += ltile * rtile;
    }

    if (K - K4 > 0) {
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile, batch, row, i, M, K, N, tile_height, K - K4);
        load_strided_rtile3d(ndim, rshape, rstride, rhs, rtile, batch, col, i, M, K, N, K - K4, tile_width);
        otile += ltile * rtile;
    }

    store_otile3d(output, otile, batch, row, col, M, K, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void tiled_gemm2d_8x4(
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col)
{
    HxW ltile[2], rtile, otile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(otile[0]);
    reset_tile(otile[1]);

    for (i = 0; i < K4; i += 4) {
        load_ltile2d(lhs, ltile[0], row, i, K, 4, 4);
        load_ltile2d(lhs, ltile[1], row + 4, i, K, 4, 4);
        load_rtile2d(rhs, rtile, col, i, N, 4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }

    if (K > K4) {
        load_ltile2d(lhs, ltile[0], row, i, K, 4, K - K4);
        load_ltile2d(lhs, ltile[1], row + 4, i, K, 4, K - K4);
        load_rtile2d(rhs, rtile, col, i, N, K - K4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }
    
    store_otile2d(output, otile[0], row, col, N, 4, 4);
    store_otile2d(output, otile[1], row + 4, col, N, 4, 4);
}

template <class T, class R, class HxW>
void tiled_gemm3d_8x4(
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col)
{
    HxW ltile[2], rtile, otile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(otile[0]);
    reset_tile(otile[1]);

    for (i = 0; i < K4; i += 4) {
        load_ltile3d(lhs, ltile[0], batch, row, i, M, K, N, 4, 4);
        load_ltile3d(lhs, ltile[1], batch, row + 4, i, M, K, N, 4, 4);
        load_rtile3d(rhs, rtile, batch, col, i, M, K, N, 4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }

    if (K > K4) {
        load_ltile3d(lhs, ltile[0], batch, row, i, M, K, N, 4, K - K4);
        load_ltile3d(lhs, ltile[1], batch, row + 4, i, M, K, N, 4, K - K4);
        load_rtile3d(rhs, rtile, batch, col, i, M, K, N, K - K4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }
    
    store_otile3d(output, otile[0], batch, row, col, M, K, N, 4, 4);
    store_otile3d(output, otile[1], batch, row + 4, col, M, K, N, 4, 4);
}

template <class T, class R, class HxW>
void strided_tiled_gemm2d_8x4(
    const isize ndim,
    const constant isize *lshape, const constant isize *rshape,
    const constant isize *lstride, const constant isize *rstride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col)
{
    HxW ltile[2], rtile, otile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(otile[0]);
    reset_tile(otile[1]);

    for (i = 0; i < K4; i += 4) {
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile[0], row, i, K, 4, 4);
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile[1], row + 4, i, K, 4, 4);
        load_strided_rtile2d(ndim, rshape, rstride, rhs, rtile, col, i, N, 4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }

    if (K > K4) {
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile[0], row, i, K, 4, K - K4);
        load_strided_ltile2d(ndim, lshape, lstride, lhs, ltile[1], row + 4, i, K, 4, K - K4);
        load_strided_rtile2d(ndim, rshape, rstride, rhs, rtile, col, i, N, K - K4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }
    
    store_otile2d(output, otile[0], row, col, N, 4, 4);
    store_otile2d(output, otile[1], row + 4, col, N, 4, 4);
}

template <class T, class R, class HxW>
void strided_tiled_gemm3d_8x4(
    const isize ndim,
    const constant isize *lshape, const constant isize *rshape,
    const constant isize *lstride, const constant isize *rstride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col)
{
    HxW ltile[2], rtile, otile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(otile[0]);
    reset_tile(otile[1]);

    for (i = 0; i < K4; i += 4) {
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile[0], batch, row, i, M, K, N, 4, 4);
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile[1], batch, row + 4, i, M, K, N, 4, 4);
        load_strided_rtile3d(ndim, rshape, rstride, rhs, rtile, batch, col, i, M, K, N, 4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }

    if (K > K4) {
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile[0], batch, row, i, M, K, N, 4, K - K4);
        load_strided_ltile3d(ndim, lshape, lstride, lhs, ltile[1], batch, row + 4, i, M, K, N, 4, K - K4);
        load_strided_rtile3d(ndim, rshape, rstride, rhs, rtile, batch, col, i, M, K, N, K - K4, 4);
        otile[0] += ltile[0] * rtile;
        otile[1] += ltile[1] * rtile;
    }
    
    store_otile3d(output, otile[0], batch, row, col, M, K, N, 4, 4);
    store_otile3d(output, otile[1], batch, row + 4, col, M, K, N, 4, 4);
}

template <class T, class R, class HxW>
kernel void tiled_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *lshape [[buffer(1)]],
    const constant isize *rshape [[buffer(2)]],
    const device T *lhs [[buffer(3)]],
    const device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    const isize M = lshape[0], K = lshape[1], N = rshape[1];
    const isize row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        tiled_gemm2d_8x4<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, tile_width);
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
    } else {
        tiled_gemm2d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, K, N, row, col, tile_height, tile_width);
    }
}

template <class T, class R, class HxW>
kernel void tiled_gemm3d(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *lshape [[buffer(2)]],
    const constant isize *rshape [[buffer(3)]],
    const device T *lhs [[buffer(4)]],
    const device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    const isize M = lshape[ndim - 2], K = lshape[ndim - 1], N = rshape[ndim - 1];
    const isize batch = id.z, row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        tiled_gemm3d_8x4<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, tile_width);
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
    } else {
        tiled_gemm3d_HxW<T, R, HxW>(offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, tile_height, tile_width);
    }
}

template <class T, class R, class HxW>
kernel void strided_tiled_gemm2d(
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
    const isize M = lshape[0], K = lshape[1], N = rshape[1];
    const isize row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        strided_tiled_gemm2d_8x4<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, tile_width);
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
    } else {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, K, N, row, col, tile_height, tile_width);
    }
}

template <class T, class R, class HxW>
kernel void strided_tiled_gemm3d(
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
    const isize M = lshape[ndim - 2], K = lshape[ndim - 1], N = rshape[ndim - 1];
    const isize batch = id.z, row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        strided_tiled_gemm3d_8x4<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, tile_width);
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
    } else {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, lshape, rshape, lstride, rstride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, tile_height, tile_width);
    }
}

#define def_tiled_gemm(dtype, T, R, HxW)    \
template [[host_name("tiled_gemm2d_" #dtype)]] [[kernel]] decltype(tiled_gemm2d<T, R, HxW>) tiled_gemm2d<T, R, HxW>;                            \
template [[host_name("strided_tiled_gemm2d_" #dtype)]] [[kernel]] decltype(strided_tiled_gemm2d<T, R, HxW>) strided_tiled_gemm2d<T, R, HxW>;    \
template [[host_name("tiled_gemm3d_" #dtype)]] [[kernel]] decltype(tiled_gemm3d<T, R, HxW>) tiled_gemm3d<T, R, HxW>;                            \
template [[host_name("strided_tiled_gemm3d_" #dtype)]] [[kernel]] decltype(strided_tiled_gemm3d<T, R, HxW>) strided_tiled_gemm3d<T, R, HxW>;

def_tiled_gemm(f32, float, float, metal::float4x4);