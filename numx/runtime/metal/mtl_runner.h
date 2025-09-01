#pragma once

#include "../runner.h"
#include "mtl_encoder.h"

namespace nx::runtime::metal {
    class MTLRunner : public Runner {
    private:
        static constexpr isize s_simd_size = 32;
        static constexpr isize s_max_threadgroup_size = 256;

        void run_full_kernel(OpPtr op, isize constant) override;
        void run_arange_kernel(OpPtr op, isize start, isize step) override;
        void run_uniform_kernel(OpPtr op, isize key, isize low, isize high) override;
        void run_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) override;
        void run_contiguous_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op);
        void run_strided_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op);
        void run_dot_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op);
        void run_gemm2d_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op);
        void run_gemm3d_kernel(MTLEncoder &encoder, OpPtr l_op, OpPtr r_op, OpPtr out_op);
        void run_gemm_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) override;
        void run_unary_kernel(OpPtr in_op, OpPtr out_op) override;
        void run_contiguous_unary_kernel(OpPtr in_op, OpPtr out_op);
        void run_strided_unary_kernel(OpPtr in_op, OpPtr out_op);
        void run_copy_kernel(OpPtr in_op, OpPtr out_op) override;
        void run_contiguous_copy_kernel(OpPtr in_op, OpPtr out_op);
        void run_strided_copy_kernel(OpPtr in_op, OpPtr out_op);
        void run_reduce_all_kernel(OpPtr in_op, OpPtr out_op) override;
        std::pair<isize, isize> select_reduce_col_kernel_size(isize nrow, isize ncol);
        void run_reduce_col_kernel(OpPtr in_op, OpPtr out_op) override;
        void run_initializer_op(OpPtr op) override;
        void run_unary_op(OpPtr op) override;
        void run_binary_op(OpPtr op) override;
        void run_transform_op(OpPtr op) override;
        void run_reduce_op(OpPtr op) override;

        template <class O>
        void run_simple_transform_op(OpPtr op) {
            auto transform_op = std::static_pointer_cast<O>(op);
            OpPtr operand = transform_op->get_operand();
            share_buffer(op, operand);
        }

        void alloc_buffer(OpPtr op);
        void share_buffer(OpPtr l_op, OpPtr r_op);

    public:
        MTLRunner(GraphPtr graph, RuntimeContextPtr ctx) : Runner(graph, ctx) {}
    };
} // namespace nx::runtime::metal