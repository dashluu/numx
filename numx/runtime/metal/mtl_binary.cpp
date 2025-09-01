#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_contiguous_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize offset[] = {l_data.get_offset(), r_data.get_offset(), out_data.get_offset()};
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 3);
        encoder.encode_array_buffer(l_data);
        encoder.encode_array_buffer(r_data);
        encoder.encode_array_buffer(out_data);
        const std::string kernel_name = std::format("{}_{}", out_op->get_opname(), l_data.get_dtype()->str());
        encoder.set_pipeline_state(kernel_name);
        const isize numel = l_data.get_numel();
        encoder.dispatch_threads(numel, std::min(numel, s_max_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_strided_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize ndim = l_data.get_ndim();
        const isize offset[] = {l_data.get_offset(), r_data.get_offset(), out_data.get_offset()};
        const bool strided[] = {!l_data.is_contiguous(), !r_data.is_contiguous(), !out_data.is_contiguous()};
        encoder.encode_mtl_buffer(&ndim, sizeof(isize));
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 3);
        encoder.encode_view(l_data);
        encoder.encode_stride(l_data);
        encoder.encode_stride(r_data);
        encoder.encode_stride(out_data);
        encoder.encode_mtl_buffer(strided, sizeof(bool) * 3);
        encoder.encode_array_buffer(l_data);
        encoder.encode_array_buffer(r_data);
        encoder.encode_array_buffer(out_data);
        const std::string kernel_name = std::format("strided_{}_{}", out_op->get_opname(), l_data.get_dtype()->str());
        encoder.set_pipeline_state(kernel_name);
        const isize numel = l_data.get_numel();
        encoder.dispatch_threads(numel, std::min(numel, s_max_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) {
        if (l_op->get_data().is_contiguous() && r_op->get_data().is_contiguous() && out_op->get_data().is_contiguous()) {
            run_contiguous_binary_kernel(l_op, r_op, out_op);
        } else {
            run_strided_binary_kernel(l_op, r_op, out_op);
        }
    }
} // namespace nx::runtime::metal