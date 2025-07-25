#include "array.h"

namespace nx::core {
    Array::~Array() {
        if (m_graph) {
            AllocatorPtr allocator = get_runtime_context()->get_allocator();
            ProfilerPtr profiler = m_runner->get_profiler();

            // Free buffers on the forward tape
            for (auto iter = m_graph->fw_begin(); iter != m_graph->fw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                ArrayBuffer &buffer = data.m_buffer;

                if (buffer.is_primary()) {
                    // Free only the owning buffers, that is, buffers that actually reference
                    // some valid memory allocated by the allocator, not just memory view
                    const Block &block = buffer.get_block();
                    allocator->free_block(block);
                    buffer.invalidate();

                    if (profiler) {
                        profiler->record_free(data);
                    }
                }
            }

            // Free buffers on the backward tape
            for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                ArrayBuffer &buffer = data.m_buffer;

                if (buffer.is_primary()) {
                    const Block &block = buffer.get_block();
                    allocator->free_block(block);
                    buffer.invalidate();

                    if (profiler) {
                        profiler->record_free(data);
                    }
                }
            }
        }
    }

    void Array::eval() {
        if (!m_runner) {
            m_graph = get_graph_builder()(m_op);
            RuntimeContextPtr runtime_ctx = get_runtime_context();
            ProfilerPtr profiler = get_profiler();
            m_runner = get_runner_builder()(m_graph, runtime_ctx);
            m_runner->hook_profiler(profiler);
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
