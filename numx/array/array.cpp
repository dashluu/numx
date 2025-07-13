#include "array.h"

namespace nx::array {
    Backend &Array::get_backend() {
        static Backend backend;
        backend.init();
        return backend;
    }

    Array::~Array() {
        if (m_graph) {
            RuntimeContextPtr ctx = get_context();
            AllocatorPtr allocator = ctx->get_allocator();

            // Free buffers on the forward tape
            for (auto iter = m_graph->fw_begin(); iter != m_graph->fw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                ArrayBuffer &buffer = data.m_buffer;

                if (buffer.is_primary()) {
                    // Free only the owning buffers, that is, buffers that actually reference
                    // some valid memory allocated by the allocator, not just memory views
                    const Block &block = buffer.get_block();
                    allocator->free(block);
                    buffer.invalidate();
                }
            }

            // Free buffers on the backward tape
            for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();
                ArrayBuffer &buffer = data.m_buffer;

                if (buffer.is_primary()) {
                    const Block &block = buffer.get_block();
                    allocator->free(block);
                    buffer.invalidate();
                }
            }
        }
    }

    void Array::eval() {
        if (!m_runner) {
            m_graph = get_graph_factory()(m_op);
            RuntimeContextPtr ctx = get_context();
            m_runner = get_runner_factory()(m_graph, ctx);
            m_graph->forward();
            m_runner->forward();
        }
    }

    void Array::backward() {
        eval();
        m_graph->backward();
        m_runner->backward();
    }
} // namespace nx::array
