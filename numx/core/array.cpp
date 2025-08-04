#include "array.h"

namespace nx::core {
    Array::~Array() {
        if (m_graph) {
            RuntimeContextPtr runtime_ctx = get_runtime_context();
            MemoryPtr memory = runtime_ctx->get_memory();
            MemoryProfilerPtr memory_profiler = runtime_ctx->get_memory_profiler();

            // Free buffers on the forward tape
            for (auto iter = m_graph->fw_begin(); iter != m_graph->fw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                const ArrayBuffer &buffer = data.get_buffer();

                if (buffer.get_type() == ArrayBufferType::Managed) {
                    memory->free_block(buffer.get_block());
                    data.invalidate_buffer();

                    if (memory_profiler->is_enabled()) {
                        memory_profiler->trace_free_block(data);
                    }
                }
            }

            // Free buffers on the backward tape
            for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                const ArrayBuffer &buffer = data.get_buffer();

                if (buffer.get_type() == ArrayBufferType::Managed) {
                    memory->free_block(buffer.get_block());
                    data.invalidate_buffer();

                    if (memory_profiler->is_enabled()) {
                        memory_profiler->trace_free_block(data);
                    }
                }
            }
        }
    }

    void Array::eval() {
        if (!m_runner) {
            m_graph = get_graph_builder()(m_op);
            RuntimeContextPtr runtime_ctx = get_runtime_context();
            m_runner = get_runner_builder()(m_graph, runtime_ctx);
            m_graph->forward();
            m_runner->forward();
        }
    }

    void Array::backward() {
        eval();
        m_graph->backward();
        m_runner->backward();
    }
} // namespace nx::core
