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
void load_l_tile_2d(const device T* lhs, thread HxW &l_tile, const isize row, const isize i, const isize K, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            l_tile[k][j] = metal::select(0.0f, lhs[(row + j) * K + i + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_l_tile_2d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* lhs, thread HxW &l_tile, const isize row, const isize i, const isize K, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            l_tile[k][j] = metal::select(0.0f, lhs[get_elm_loc((row + j) * K + i + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_l_tile_3d(const device T* lhs, thread HxW &l_tile, const isize batch, const isize row, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            l_tile[k][j] = metal::select(0.0f, lhs[batch * M * K + (row + j) * K + i + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_l_tile_3d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* lhs, thread HxW &l_tile, const isize batch, const isize row, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            l_tile[k][j] = metal::select(0.0f, lhs[get_elm_loc(batch * M * K + (row + j) * K + i + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_r_tile_2d(const device T* rhs, thread HxW &r_tile, const isize col, const isize i, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            r_tile[k][j] = metal::select(0.0f, rhs[(i + j) * N + col + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_r_tile_3d(const device T* rhs, thread HxW &r_tile, const isize batch, const isize col, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            r_tile[k][j] = metal::select(0.0f, rhs[batch * K * N + (i + j) * N + col + k], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_r_tile_2d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* rhs, thread HxW &r_tile, const isize col, const isize i, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            r_tile[k][j] = metal::select(0.0f, rhs[get_elm_loc((i + j) * N + col + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void load_strided_r_tile_3d(const isize ndim, const constant isize *shape, const constant isize *stride, const device T* rhs, thread HxW &r_tile, const isize batch,  const isize col, const isize i, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < 4; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < 4; ++k) {
            r_tile[k][j] = metal::select(0.0f, rhs[get_elm_loc(batch * K * N + (i + j) * N + col + k, ndim, shape, stride)], (j < tile_height) && (k < tile_width));
        }
    }
}

template <class T, class HxW>
void store_out_tile_2d(device T* output, thread HxW &out_tile, const isize row, const isize col, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < tile_height; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < tile_width; ++k) {
            output[(row + j) * N + col + k] = out_tile[k][j];
        }
    }
}

template <class T, class HxW>
void store_out_tile_3d(device T* output, thread HxW &out_tile, const isize batch, const isize row, const isize col, const isize M, const isize K, const isize N, const isize tile_height, const isize tile_width) {
    #pragma unroll
    for (ubyte j = 0; j < tile_height; ++j) {
        #pragma unroll
        for (ubyte k = 0; k < tile_width; ++k) {
            output[batch * M * N + (row + j) * N + col + k] = out_tile[k][j];
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
    HxW l_tile, r_tile, out_tile;
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile);

    for (i = 0; i < K4; i += 4) {
        load_l_tile_2d(lhs, l_tile, row, i, K, tile_height, 4);
        load_r_tile_2d(rhs, r_tile, col, i, N, 4, tile_width);
        out_tile += l_tile * r_tile;
    }

    if (K - K4 > 0) {
        load_l_tile_2d(lhs, l_tile, row, i, K, tile_height, K - K4);
        load_r_tile_2d(rhs, r_tile, col, i, N, K - K4, tile_width);
        out_tile += l_tile * r_tile;
    }

    store_out_tile_2d(output, out_tile, row, col, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void tiled_gemm3d_HxW(
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW l_tile, r_tile, out_tile;
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile);

    for (i = 0; i < K4; i += 4) {
        load_l_tile_3d(lhs, l_tile, batch, row, i, M, K, N, tile_height, 4);
        load_r_tile_3d(rhs, r_tile, batch, col, i, M, K, N, 4, tile_width);
        out_tile += l_tile * r_tile;
    }

    if (K - K4 > 0) {
        load_l_tile_3d(lhs, l_tile, batch, row, i, M, K, N, tile_height, K - K4);
        load_r_tile_3d(rhs, r_tile, batch, col, i, M, K, N, K - K4, tile_width);
        out_tile += l_tile * r_tile;
    }

    store_out_tile_3d(output, out_tile, batch, row, col, M, K, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void strided_tiled_gemm2d_HxW(
    const isize ndim,
    const constant isize *l_shape, const constant isize *r_shape,
    const constant isize *l_stride, const constant isize *r_stride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW l_tile, r_tile, out_tile;
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile);

    for (i = 0; i < K4; i += 4) {
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile, row, i, K, tile_height, 4);
        load_strided_r_tile_2d(ndim, r_shape, r_stride, rhs, r_tile, col, i, N, 4, tile_width);
        out_tile += l_tile * r_tile;
    }

    if (K - K4 > 0) {
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile, row, i, K, tile_height, K - K4);
        load_strided_r_tile_2d(ndim, r_shape, r_stride, rhs, r_tile, col, i, N, K - K4, tile_width);
        out_tile += l_tile * r_tile;
    }

    store_out_tile_2d(output, out_tile, row, col, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void strided_tiled_gemm3d_HxW(
    const isize ndim,
    const constant isize *l_shape, const constant isize *r_shape,
    const constant isize *l_stride, const constant isize *r_stride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col,
    const isize tile_height, const isize tile_width)
{
    HxW l_tile, r_tile, out_tile;
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile);

    for (i = 0; i < K4; i += 4) {
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile, batch, row, i, M, K, N, tile_height, 4);
        load_strided_r_tile_3d(ndim, r_shape, r_stride, rhs, r_tile, batch, col, i, M, K, N, 4, tile_width);
        out_tile += l_tile * r_tile;
    }

    if (K - K4 > 0) {
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile, batch, row, i, M, K, N, tile_height, K - K4);
        load_strided_r_tile_3d(ndim, r_shape, r_stride, rhs, r_tile, batch, col, i, M, K, N, K - K4, tile_width);
        out_tile += l_tile * r_tile;
    }

    store_out_tile_3d(output, out_tile, batch, row, col, M, K, N, tile_height, tile_width);
}

template <class T, class R, class HxW>
void tiled_gemm2d_8x4(
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col)
{
    HxW l_tile[2], r_tile, out_tile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile[0]);
    reset_tile(out_tile[1]);

    for (i = 0; i < K4; i += 4) {
        load_l_tile_2d(lhs, l_tile[0], row, i, K, 4, 4);
        load_l_tile_2d(lhs, l_tile[1], row + 4, i, K, 4, 4);
        load_r_tile_2d(rhs, r_tile, col, i, N, 4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }

    if (K > K4) {
        load_l_tile_2d(lhs, l_tile[0], row, i, K, 4, K - K4);
        load_l_tile_2d(lhs, l_tile[1], row + 4, i, K, 4, K - K4);
        load_r_tile_2d(rhs, r_tile, col, i, N, K - K4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }
    
    store_out_tile_2d(output, out_tile[0], row, col, N, 4, 4);
    store_out_tile_2d(output, out_tile[1], row + 4, col, N, 4, 4);
}

template <class T, class R, class HxW>
void tiled_gemm3d_8x4(
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col)
{
    HxW l_tile[2], r_tile, out_tile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile[0]);
    reset_tile(out_tile[1]);

    for (i = 0; i < K4; i += 4) {
        load_l_tile_3d(lhs, l_tile[0], batch, row, i, M, K, N, 4, 4);
        load_l_tile_3d(lhs, l_tile[1], batch, row + 4, i, M, K, N, 4, 4);
        load_r_tile_3d(rhs, r_tile, batch, col, i, M, K, N, 4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }

    if (K > K4) {
        load_l_tile_3d(lhs, l_tile[0], batch, row, i, M, K, N, 4, K - K4);
        load_l_tile_3d(lhs, l_tile[1], batch, row + 4, i, M, K, N, 4, K - K4);
        load_r_tile_3d(rhs, r_tile, batch, col, i, M, K, N, K - K4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }
    
    store_out_tile_3d(output, out_tile[0], batch, row, col, M, K, N, 4, 4);
    store_out_tile_3d(output, out_tile[1], batch, row + 4, col, M, K, N, 4, 4);
}

template <class T, class R, class HxW>
void strided_tiled_gemm2d_8x4(
    const isize ndim,
    const constant isize *l_shape, const constant isize *r_shape,
    const constant isize *l_stride, const constant isize *r_stride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize K, const isize N,
    const isize row, const isize col)
{
    HxW l_tile[2], r_tile, out_tile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile[0]);
    reset_tile(out_tile[1]);

    for (i = 0; i < K4; i += 4) {
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile[0], row, i, K, 4, 4);
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile[1], row + 4, i, K, 4, 4);
        load_strided_r_tile_2d(ndim, r_shape, r_stride, rhs, r_tile, col, i, N, 4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }

    if (K > K4) {
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile[0], row, i, K, 4, K - K4);
        load_strided_l_tile_2d(ndim, l_shape, l_stride, lhs, l_tile[1], row + 4, i, K, 4, K - K4);
        load_strided_r_tile_2d(ndim, r_shape, r_stride, rhs, r_tile, col, i, N, K - K4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }
    
    store_out_tile_2d(output, out_tile[0], row, col, N, 4, 4);
    store_out_tile_2d(output, out_tile[1], row + 4, col, N, 4, 4);
}

template <class T, class R, class HxW>
void strided_tiled_gemm3d_8x4(
    const isize ndim,
    const constant isize *l_shape, const constant isize *r_shape,
    const constant isize *l_stride, const constant isize *r_stride,
    const device T *lhs, const device T *rhs, device R *output,
    const isize M, const isize K, const isize N,
    const isize batch, const isize row, const isize col)
{
    HxW l_tile[2], r_tile, out_tile[2];
    isize i, K4 = K / 4 * 4;
    reset_tile(out_tile[0]);
    reset_tile(out_tile[1]);

    for (i = 0; i < K4; i += 4) {
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile[0], batch, row, i, M, K, N, 4, 4);
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile[1], batch, row + 4, i, M, K, N, 4, 4);
        load_strided_r_tile_3d(ndim, r_shape, r_stride, rhs, r_tile, batch, col, i, M, K, N, 4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }

    if (K > K4) {
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile[0], batch, row, i, M, K, N, 4, K - K4);
        load_strided_l_tile_3d(ndim, l_shape, l_stride, lhs, l_tile[1], batch, row + 4, i, M, K, N, 4, K - K4);
        load_strided_r_tile_3d(ndim, r_shape, r_stride, rhs, r_tile, batch, col, i, M, K, N, K - K4, 4);
        out_tile[0] += l_tile[0] * r_tile;
        out_tile[1] += l_tile[1] * r_tile;
    }
    
    store_out_tile_3d(output, out_tile[0], batch, row, col, M, K, N, 4, 4);
    store_out_tile_3d(output, out_tile[1], batch, row + 4, col, M, K, N, 4, 4);
}

template <class T, class R, class HxW>
kernel void tiled_gemm2d(
    const constant isize *offset [[buffer(0)]],
    const constant isize *l_shape [[buffer(1)]],
    const constant isize *r_shape [[buffer(2)]],
    const device T *lhs [[buffer(3)]],
    const device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    const isize M = l_shape[0], K = l_shape[1], N = r_shape[1];
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
    const constant isize *l_shape [[buffer(2)]],
    const constant isize *r_shape [[buffer(3)]],
    const device T *lhs [[buffer(4)]],
    const device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    const isize M = l_shape[ndim - 2], K = l_shape[ndim - 1], N = r_shape[ndim - 1];
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
    const constant isize *l_shape [[buffer(1)]],
    const constant isize *r_shape [[buffer(2)]],
    const constant isize *l_stride [[buffer(3)]],
    const constant isize *r_stride [[buffer(4)]],
    const device T *lhs [[buffer(5)]],
    const device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    const isize M = l_shape[0], K = l_shape[1], N = r_shape[1];
    const isize row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        strided_tiled_gemm2d_8x4<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, tile_width);
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row, col, 4, 4);
    } else {
        strided_tiled_gemm2d_HxW<T, R, HxW>(2, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, K, N, row, col, tile_height, tile_width);
    }
}

template <class T, class R, class HxW>
kernel void strided_tiled_gemm3d(
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
    const isize M = l_shape[ndim - 2], K = l_shape[ndim - 1], N = r_shape[ndim - 1];
    const isize batch = id.z, row = id.y * 8, col = id.x * 4;
    const isize tile_height = 8 < (M - row) ? 8 : (M - row);
    const isize tile_width = 4 < (N - col) ? 4 : (N - col);
    const device T *offset_lhs = lhs + offset[0];
    const device T *offset_rhs = rhs + offset[1];
    device R *offset_output = output + offset[2];
    
    if (tile_height == 8 && tile_width == 4) {
        strided_tiled_gemm3d_8x4<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col);
    } else if (tile_height > 4 && tile_width == 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, 4);
    } else if (tile_height > 4 && tile_width < 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, tile_width);
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row + 4, col, tile_height - 4, tile_width);
    } else if (tile_height == 4 && tile_width == 4) {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, 4, 4);
    } else {
        strided_tiled_gemm3d_HxW<T, R, HxW>(ndim, l_shape, r_shape, l_stride, r_stride, offset_lhs, offset_rhs, offset_output, M, K, N, batch, row, col, tile_height, tile_width);
    }
}

#define def_tiled_gemm(dtype, T, R, HxW)    \
template [[host_name("tiled_gemm2d_" #dtype)]] [[kernel]] decltype(tiled_gemm2d<T, R, HxW>) tiled_gemm2d<T, R, HxW>;                            \
template [[host_name("strided_tiled_gemm2d_" #dtype)]] [[kernel]] decltype(strided_tiled_gemm2d<T, R, HxW>) strided_tiled_gemm2d<T, R, HxW>;    \
template [[host_name("tiled_gemm3d_" #dtype)]] [[kernel]] decltype(tiled_gemm3d<T, R, HxW>) tiled_gemm3d<T, R, HxW>;                            \
template [[host_name("strided_tiled_gemm3d_" #dtype)]] [[kernel]] decltype(strided_tiled_gemm3d<T, R, HxW>) strided_tiled_gemm3d<T, R, HxW>;

def_tiled_gemm(f32, float, float, metal::float4x4);