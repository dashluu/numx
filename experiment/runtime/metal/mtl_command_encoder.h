#pragma once

#include "../../core/array_descriptor.h"
#include "mtl_context.h"

namespace nx::runtime::metal {
    class MTLCommandEncoder {
    private:
        MTLContextPtr ctx;
        MTL::CommandBuffer *cmd_buff;
        MTL::ComputeCommandEncoder *encoder;
        isize buff_idx = 0;

    public:
        MTLCommandEncoder(RunnerContextPtr ctx) {
            this->ctx = std::static_pointer_cast<MTLContext>(ctx);
            cmd_buff = this->ctx->get_cmd_queue()->commandBuffer();
            encoder = cmd_buff->computeCommandEncoder();
        }

        MTLCommandEncoder(const MTLCommandEncoder &) = delete;
        ~MTLCommandEncoder() = default;
        MTLCommandEncoder &operator=(const MTLCommandEncoder &) = delete;
        MTL::ComputeCommandEncoder *get_internal_encoder() const { return encoder; }

        void encode_buffer(const void *buff, isize size) {
            MTL::Buffer *mtl_buff = ctx->get_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
            encoder->setBuffer(mtl_buff, 0, buff_idx++);
        }

        void encode_view(LazyPtr lazy) { encode_buffer(lazy->get_view().data(), sizeof(isize) * lazy->get_ndim()); }
        void encode_stride(LazyPtr lazy) { encode_buffer(lazy->get_stride().data(), sizeof(isize) * lazy->get_ndim()); }
        void encode_lazy(LazyPtr lazy) { encode_buffer(lazy->get_buff_ptr(), lazy->get_buff_size()); }

        void set_pipeline_state(const std::string &kernel_name) {
            MTLKernelPtr kernel = ctx->get_kernel(kernel_name);
            if (!kernel) {
                throw std::invalid_argument(std::format("Kernel {} not found.", kernel_name));
            }
            encoder->setComputePipelineState(kernel->get_state().get());
        }

        void dispatch_threads(isize grid_nthread, isize threadgroup_nthread) {
            MTL::Size grid_size = MTL::Size::Make(grid_nthread, 1, 1);
            MTL::Size threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);
            dispatch_threads(grid_size, threadgroup_size);
        }

        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size) {
            encoder->dispatchThreads(grid_size, threadgroup_size);
            encoder->endEncoding();
            cmd_buff->commit();
        }

        void wait_to_complete() { cmd_buff->waitUntilCompleted(); }

        double time_to_complete() {
            cmd_buff->waitUntilCompleted();
            CFTimeInterval start = cmd_buff->GPUStartTime();
            CFTimeInterval end = cmd_buff->GPUEndTime();
            return end - start;
        }
    };
} // namespace nx::runtime::metal