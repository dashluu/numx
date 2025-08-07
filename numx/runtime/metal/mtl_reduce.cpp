#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_reduce_all_kernel(OpPtr in_op, OpPtr out_op) {
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &in_data = in_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize ndim = in_data.get_ndim();
        const isize numel = in_data.get_numel();
        const isize offset[] = {in_data.get_offset(), out_data.get_offset()};
        encoder.encode_mtl_buffer(&numel, sizeof(isize));
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 2);
        const bool strided = !in_data.is_contiguous();

        if (strided) {
            encoder.encode_mtl_buffer(&ndim, sizeof(isize));
            encoder.encode_view(in_data);
            encoder.encode_stride(in_data);
        }

        encoder.encode_array_buffer(in_data);
        encoder.encode_array_buffer(out_data);

        // Configure kernel
        DtypePtr dtype = in_data.get_dtype();
        const std::string kernel_name = (strided ? "strided_" : "") + out_op->get_opname() + "_all_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);

        // Calculate thread configuration
        // Make sure the number of threads aligned to simd size
        const isize aligned_numel = align_to(numel, s_simd_size);
        auto grid_size = MTL::Size::Make(aligned_numel, 1, 1);
        isize threadgroup_nthread = std::min(aligned_numel, s_max_threadgroup_size);
        auto threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
        pool->release();
    }

    std::pair<isize, isize> MTLRunner::select_reduce_col_kernel_size(isize nrow, isize ncol) {
        isize lhs, rhs;
        isize best_row_groups, best_col_groups;
        isize best_min = -1, best_max = -1;

        for (isize i = 1; i <= s_simd_size; i <<= 1) {
            for (isize j = 1; i * j <= s_simd_size; j <<= 1) {
                lhs = std::abs(nrow - i);
                rhs = std::abs(ncol - j * s_simd_size);
                auto [min, max] = std::minmax(lhs, rhs);
                // std::println("{} {} {} {}", i, j, min, max);
                if (best_max == -1 || best_max > max || (best_max == max && (best_min == -1 || best_min > min))) {
                    best_max = max;
                    best_min = min;
                    best_row_groups = i;
                    best_col_groups = j;
                }
            }
        }

        return {best_row_groups, best_col_groups};
    }

    void MTLRunner::run_reduce_col_kernel(OpPtr in_op, OpPtr out_op) {
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &in_data = in_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        ReduceOpPtr reduce_op = std::static_pointer_cast<ReduceOp>(out_op);
        const ShapeDims &remaining_dims = reduce_op->get_remaining_dims();
        const ShapeDims &reduce_dims = reduce_op->get_reduce_dims();
        const ShapeView &in_view = in_data.get_view();

        // Move reduction dimensions to the end
        ShapeDims permutation_dims;
        permutation_dims.reserve(remaining_dims.size() + reduce_dims.size());
        permutation_dims.insert(permutation_dims.end(), remaining_dims.begin(), remaining_dims.end());
        permutation_dims.insert(permutation_dims.end(), reduce_dims.begin(), reduce_dims.end());
        // Detach input op so the computational graph is not affected
        OpPtr permutation_op = permute(detach(in_op), permutation_dims);
        const ArrayData &permutation_data = permutation_op->get_data();
        share_buffer(permutation_op, in_op);
        const isize ndim = in_data.get_ndim();
        const isize nrow = std::accumulate(remaining_dims.begin(), remaining_dims.end(), 1ll, [&](isize acc, isize dim) { return acc * in_view[dim]; });
        const isize ncol = std::accumulate(reduce_dims.begin(), reduce_dims.end(), 1ll, [&](isize acc, isize dim) { return acc * in_view[dim]; });
        const isize offset[] = {permutation_data.get_offset(), out_data.get_offset()};
        encoder.encode_mtl_buffer(&ncol, sizeof(isize));
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 2);
        const bool strided = !permutation_data.is_contiguous();

        if (strided) {
            encoder.encode_mtl_buffer(&ndim, sizeof(isize));
            encoder.encode_view(permutation_data);
            encoder.encode_stride(permutation_data);
        }

        encoder.encode_array_buffer(permutation_data);
        encoder.encode_array_buffer(out_data);

        // Configure kernel
        DtypePtr dtype = permutation_data.get_dtype();
        auto [row_groups, col_groups] = select_reduce_col_kernel_size(nrow, ncol);
        // std::println("row_groups: {}, col_groups: {}", row_groups, col_groups);
        const std::string kernel_name = (strided ? "strided_" : "") + std::format("{}_col_{}x{}_{}", out_op->get_opname(), row_groups, col_groups, dtype->str());
        encoder.set_pipeline_state(kernel_name);

        // Calculate thread configuration
        auto grid_size = MTL::Size::Make(align_to(ncol, s_simd_size), nrow, 1);
        auto threadgroup_size = MTL::Size::Make(col_groups * s_simd_size, row_groups, 1);

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        // double time = encoder.time_to_complete();
        // std::println("time: {}", time);
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace nx::runtime::metal