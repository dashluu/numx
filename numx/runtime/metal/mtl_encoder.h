#pragma once

#include "mtl_context.h"

namespace nx::runtime::metal {
    class MTLEncoder {
    private:
        MTLContextPtr m_ctx;
        MTL::CommandBuffer *m_cmd_buffer;
        MTL::ComputeCommandEncoder *m_encoder;
        std::vector<MTL::Buffer *> m_buffers;
        isize m_buffer_index = 0;

    public:
        MTLEncoder(RuntimeContextPtr ctx) {
            m_ctx = std::static_pointer_cast<MTLContext>(ctx);
            m_cmd_buffer = m_ctx->get_cmd_queue()->commandBuffer();
            m_encoder = m_cmd_buffer->computeCommandEncoder();
        }

        MTLEncoder(const MTLEncoder &) = delete;
        MTLEncoder(MTLEncoder &&) noexcept = delete;

        ~MTLEncoder() {
            for (MTL::Buffer *buffer : m_buffers) {
                buffer->release();
            }
        }

        MTLEncoder &operator=(const MTLEncoder &) = delete;
        MTLEncoder &operator=(MTLEncoder &&) noexcept = delete;
        MTL::ComputeCommandEncoder *get_internal_encoder() const { return m_encoder; }

        void encode_mtl_buffer(const void *buff, isize size) {
            MTL::Buffer *mtl_buffer = m_ctx->get_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
            m_buffers.push_back(mtl_buffer);
            m_encoder->setBuffer(mtl_buffer, 0, m_buffer_index++);
        }

        void encode_view(const ArrayData &data) { encode_mtl_buffer(data.get_view().data(), sizeof(isize) * data.get_ndim()); }
        void encode_stride(const ArrayData &data) { encode_mtl_buffer(data.get_stride().data(), sizeof(isize) * data.get_ndim()); }

        void encode_array_buffer(const ArrayData &data) {
            const ArrayBuffer &buffer = data.get_buffer();
            encode_mtl_buffer(buffer.get_ptr(), buffer.get_size());
        }

        void set_pipeline_state(const std::string &kernel_name) {
            MTLKernelPtr kernel = m_ctx->get_kernel(kernel_name);

            if (!kernel) {
                throw std::runtime_error(std::format("No kernel named {}.", kernel_name));
            }

            m_encoder->setComputePipelineState(kernel->get_state().get());
        }

        void dispatch_threads(isize grid_nthread, isize threadgroup_nthread) {
            MTL::Size grid_size = MTL::Size::Make(grid_nthread, 1, 1);
            MTL::Size threadgroup_size = MTL::Size::Make(threadgroup_nthread, 1, 1);
            dispatch_threads(grid_size, threadgroup_size);
        }

        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size) {
            m_encoder->dispatchThreads(grid_size, threadgroup_size);
            m_encoder->endEncoding();
            m_cmd_buffer->commit();
        }

        void wait_to_complete() { m_cmd_buffer->waitUntilCompleted(); }

        double time_to_complete() {
            m_cmd_buffer->waitUntilCompleted();
            CFTimeInterval start = m_cmd_buffer->GPUStartTime();
            CFTimeInterval end = m_cmd_buffer->GPUEndTime();
            return end - start;
        }
    };
} // namespace nx::runtime::metal