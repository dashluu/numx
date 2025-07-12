#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_copy_kernel(OpPtr in_op, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &in_data = in_op->get_data();
        const ArrayData &out_data = out_op->get_data();
        const isize ndim = in_data.get_ndim();
        const isize offset[] = {in_data.get_offset(), out_data.get_offset()};
        const bool strided[] = {!in_data.is_contiguous(), !out_data.is_contiguous()};
        encoder.encode_mtl_buffer(&ndim, sizeof(isize));
        encoder.encode_mtl_buffer(offset, sizeof(isize) * 2);
        encoder.encode_view(in_data);
        encoder.encode_stride(in_data);
        encoder.encode_stride(out_data);
        encoder.encode_mtl_buffer(strided, sizeof(bool) * 2);
        encoder.encode_array_buffer(in_data);
        encoder.encode_array_buffer(out_data);
        const std::string kernel_name = "copy_" + in_data.get_dtype()->str() + "_" + out_data.get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        const isize numel = in_data.get_numel();
        encoder.dispatch_threads(numel, std::min(numel, s_max_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace nx::runtime::metal