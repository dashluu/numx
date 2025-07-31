#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_dot_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        isize numel = l_op->get_data().get_numel();
        OpPtr reshaped_l_op = reshape(detach(l_op), {1, numel});
        OpPtr reshaped_r_op = reshape(detach(r_op), {numel, 1});
        share_buffer(reshaped_l_op, l_op);
        share_buffer(reshaped_r_op, r_op);
        run_gemm2d_kernel(encoder, reshaped_l_op, reshaped_r_op, out_op);
    }

    void MTLRunner::run_gemm2d_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize offset[] = {l_data.get_offset(), r_data.get_offset(), out_data.get_offset()};
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 3);
        encoder.encode_view(l_data);
        encoder.encode_view(r_data);
        const bool strided = !l_data.is_contiguous() || !r_data.is_contiguous();

        if (strided) {
            encoder.encode_stride(l_data);
            encoder.encode_stride(r_data);
        }

        encoder.encode_array_buffer(l_data);
        encoder.encode_array_buffer(r_data);
        encoder.encode_array_buffer(out_data);
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        std::string kernel_name;
        isize grid_width, grid_height;
        MTL::Size grid_size;

        if (l_data.get_dtype()->is_float()) {
            // Tiling and faster methods can only be used for floating-point
            kernel_name = (strided ? "strided_tiled_gemm2d_" : "tiled_gemm2d_") + l_data.get_dtype()->str();
            // Tiling uses 8x4 tiles
            grid_width = (r_view[1] + 3) / 4;
            grid_height = (l_view[0] + 7) / 8;
            grid_size = MTL::Size::Make(grid_width, grid_height, 1);
        } else {
            kernel_name = (strided ? "strided_naive_gemm2d_" : "naive_gemm2d_") + l_data.get_dtype()->str();
            grid_width = r_view[1];
            grid_height = l_view[0];
            grid_size = MTL::Size::Make(grid_width, grid_height, 1);
        }

        MTL::Size threadgroup_size = MTL::Size::Make(s_max_threadgroup_size, 1, 1);
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
    }

    void MTLRunner::run_gemm3d_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize offset[] = {l_data.get_offset(), r_data.get_offset(), out_data.get_offset()};
        const isize ndim = l_data.get_ndim();
        encoder.encode_mtl_buffer(&ndim, sizeof(isize));
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 3);
        encoder.encode_view(l_data);
        encoder.encode_view(r_data);
        const bool strided = !l_data.is_contiguous() || !r_data.is_contiguous();

        if (strided) {
            encoder.encode_stride(l_data);
            encoder.encode_stride(r_data);
        }

        encoder.encode_array_buffer(l_data);
        encoder.encode_array_buffer(r_data);
        encoder.encode_array_buffer(out_data);
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        const isize batch_size = std::accumulate(l_view.begin(), l_view.end() - 2, 1ll, std::multiplies<isize>());
        std::string kernel_name;
        isize grid_width, grid_height;
        MTL::Size grid_size;

        if (l_data.get_dtype()->is_float()) {
            // Tiling and faster methods can only be used for floating-point
            kernel_name = (strided ? "strided_tiled_gemm3d_" : "tiled_gemm3d_") + l_data.get_dtype()->str();
            // Tiling uses 8x4 tiles
            grid_width = (r_view[ndim - 1] + 3) / 4;
            grid_height = (l_view[ndim - 2] + 7) / 8;
            grid_size = MTL::Size::Make(grid_width, grid_height, batch_size);
        } else {
            kernel_name = (strided ? "strided_naive_gemm3d_" : "naive_gemm3d_") + l_data.get_dtype()->str();
            grid_width = r_view[ndim - 1];
            grid_height = l_view[ndim - 2];
            grid_size = MTL::Size::Make(grid_width, grid_height, batch_size);
        }

        MTL::Size threadgroup_size = MTL::Size::Make(s_max_threadgroup_size, 1, 1);
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
    }

    void MTLRunner::run_gemm_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);

        switch (l_op->get_data().get_ndim()) {
        case 1:
            run_dot_kernel(encoder, l_op, r_op, out_op);
            break;
        case 2:
            run_gemm2d_kernel(encoder, l_op, r_op, out_op);
            break;
        case 3:
            run_gemm3d_kernel(encoder, l_op, r_op, out_op);
            break;
        default:
            run_gemm3d_kernel(encoder, l_op, r_op, out_op);
            break;
        }

        pool->release();
    }
} // namespace nx::runtime::metal