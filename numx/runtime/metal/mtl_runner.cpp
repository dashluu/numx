#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_initializer_op(OpPtr op) {
        switch (op->get_opcode()) {
        case Opcode::FULL: {
            alloc_buffer(op);
            std::shared_ptr<FullOp> full_op = std::static_pointer_cast<FullOp>(op);
            run_full_kernel(op, full_op->get_const());
            break;
        }
        case Opcode::ARANGE: {
            alloc_buffer(op);
            std::shared_ptr<ArangeOp> arange_op = std::static_pointer_cast<ArangeOp>(op);
            run_arange_kernel(op, arange_op->get_start(), arange_op->get_step());
            break;
        }
        case Opcode::UNIFORM: {
            alloc_buffer(op);
            std::shared_ptr<UniformOp> uniform_op = std::static_pointer_cast<UniformOp>(op);
            run_uniform_kernel(op, uniform_op->get_key(), uniform_op->get_low(), uniform_op->get_high());
            break;
        }
        case Opcode::EMPTY: {
            alloc_buffer(op);
            break;
        }
        default:
            break;
        }
    }

    void MTLRunner::run_unary_op(OpPtr op) {
        UnaryOpPtr unary_op = std::static_pointer_cast<UnaryOp>(op);
        OpPtr operand = unary_op->get_operand();

        if (unary_op->is_in_place()) {
            share_buffer(op, operand);
        } else {
            alloc_buffer(op);
        }

        if (op->get_opcode() == Opcode::COPY) {
            run_copy_kernel(operand, op);
        } else {
            run_unary_kernel(operand, op);
        }
    }

    void MTLRunner::run_binary_op(OpPtr op) {
        BinaryOpPtr binary_op = std::static_pointer_cast<BinaryOp>(op);
        OpPtr lop = binary_op->get_lhs();
        OpPtr rop = binary_op->get_rhs();

        if (binary_op->get_mode() == BinaryMode::ELMWISE) {
            std::shared_ptr<ElmwiseBinaryOp> elmwise_op = std::static_pointer_cast<ElmwiseBinaryOp>(binary_op);
            if (elmwise_op->is_in_place()) {
                share_buffer(op, lop);
            } else {
                alloc_buffer(op);
            }
        } else {
            alloc_buffer(op);
        }

        if (binary_op->get_mode() == BinaryMode::MATMUL) {
            run_gemm_kernel(lop, rop, op);
        } else {
            run_binary_kernel(lop, rop, op);
        }
    }

    void MTLRunner::run_transform_op(OpPtr op) {
        switch (op->get_opcode()) {
        case Opcode::RESHAPE: {
            std::shared_ptr<ReshapeOp> reshape_op = std::static_pointer_cast<ReshapeOp>(op);
            OpPtr operand = reshape_op->get_operand();

            if (!operand->get_data().copy_when_reshape(reshape_op->get_data().get_view())) {
                share_buffer(op, operand);
            } else {
                alloc_buffer(op);
                run_copy_kernel(operand, op);
            }

            break;
        }
        case Opcode::SLICE: {
            run_simple_transform_op<SliceOp>(op);
            break;
        }
        case Opcode::BROADCAST: {
            run_simple_transform_op<BroadcastOp>(op);
            break;
        }
        case Opcode::PERMUTE: {
            run_simple_transform_op<PermuteOp>(op);
            break;
        }
        case Opcode::SQUEEZE: {
            run_simple_transform_op<SqueezeOp>(op);
            break;
        }
        case Opcode::UNSQUEEZE: {
            run_simple_transform_op<UnsqueezeOp>(op);
            break;
        }
        case Opcode::ASTYPE: {
            std::shared_ptr<AstypeOp> as_type_op = std::static_pointer_cast<AstypeOp>(op);
            OpPtr operand = as_type_op->get_operand();
            alloc_buffer(op);
            run_copy_kernel(operand, op);
            break;
        }
        default:
            break;
        }
    }

    void MTLRunner::run_reduce_op(OpPtr op) {
        ReduceOpPtr reduce_op = std::static_pointer_cast<ReduceOp>(op);
        OpPtr operand = reduce_op->get_operand();
        alloc_buffer(op);

        // Fill up array with default value
        if (reduce_op->get_opcode() == Opcode::MAX) {
            run_full_kernel(op, reduce_op->get_data().get_dtype()->min());
        } else if (reduce_op->get_opcode() == Opcode::MIN) {
            run_full_kernel(op, reduce_op->get_data().get_dtype()->max());
        } else {
            run_full_kernel(op, 0);
        }

        if (reduce_op->get_remaining_dims().size() == 0) {
            run_reduce_all_kernel(operand, op);
        } else {
            run_reduce_col_kernel(operand, op);
        }
    }

    void MTLRunner::alloc_buffer(OpPtr op) {
        ArrayData &data = op->get_data();

        if (!data.is_buffer_valid()) {
            BufferBlock *block = m_ctx->get_memory()->alloc_block(data.get_nbytes());
            data.set_primary_buffer(block);
            MemoryProfilerPtr memory_profiler = m_ctx->get_memory_profiler();

            if (memory_profiler->is_enabled()) {
                memory_profiler->trace_alloc_block(data);
            }
        }
    }

    void MTLRunner::share_buffer(OpPtr l_op, OpPtr r_op) {
        ArrayData &l_data = l_op->get_data();

        if (!l_data.is_buffer_valid()) {
            BufferBlock *block = r_op->get_data().get_buffer().get_block();
            l_data.set_view_buffer(block);
        }
    }
} // namespace nx::runtime::metal