#include "mtl_runner.h"

namespace nx::runtime::metal {
    void MTLRunner::run_full_kernel(OpPtr op, isize constant) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &data = op->get_data();
        DtypePtr dtype = data.get_dtype();
        encoder.encode_mtl_buffer(&constant, dtype->get_size());
        encoder.encode_array_buffer(data);
        const std::string kernel_name = "full_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);
        const isize numel = data.get_numel();
        encoder.dispatch_threads(numel, std::min(numel, s_max_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_arange_kernel(OpPtr op, isize start, isize step) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTLEncoder encoder(m_ctx);
        const ArrayData &data = op->get_data();
        encoder.encode_mtl_buffer(&start, sizeof(isize));
        encoder.encode_mtl_buffer(&step, sizeof(isize));
        encoder.encode_array_buffer(data);
        const std::string kernel_name = "arange_" + data.get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        const isize numel = data.get_numel();
        encoder.dispatch_threads(numel, std::min(numel, s_max_threadgroup_size));
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_uniform_kernel(OpPtr op) {
    }
} // namespace nx::runtime::metal