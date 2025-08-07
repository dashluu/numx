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

                if (data.is_buffer_valid()) {
                    const ArrayBuffer &buffer = data.get_buffer();

                    if (!buffer.is_view()) {
                        memory->free_block(buffer.get_block());
                        data.invalidate_buffer();

                        if (memory_profiler->is_enabled()) {
                            memory_profiler->trace_free_block(data);
                        }
                    }
                }
            }

            // Free buffers on the backward tape
            for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
                ArrayData &data = (*iter)->get_data();

                if (data.is_buffer_valid()) {
                    const ArrayBuffer &buffer = data.get_buffer();

                    if (!buffer.is_view()) {
                        memory->free_block(buffer.get_block());
                        data.invalidate_buffer();

                        if (memory_profiler->is_enabled()) {
                            memory_profiler->trace_free_block(data);
                        }
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

    std::pair<isize, isize> compute_fan_in_and_fan_out(const ShapeView &view) {
        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for a view of {} dimensions, which is fewer than 2.", view.size()));
        }

        isize receptive_field_size = std::accumulate(view.begin() + 2, view.end(), 1ll, std::multiplies<>());
        isize fan_in = view[1] * receptive_field_size;
        isize fan_out = view[0] * receptive_field_size;
        // Note: fan-in and fan-out cannot be 0 since each dimension > 0
        return {fan_in, fan_out};
    }

    std::pair<isize, isize> compute_fan_in_and_fan_out(const Array &array) {
        const ShapeView &view = array.get_view();

        if (view.size() < 2) {
            throw std::invalid_argument(std::format("Fan-in and fan-out cannot be computed for array {} of {} dimensions, which is fewer than 2.", array.get_id(), view.size()));
        }

        return compute_fan_in_and_fan_out(view);
    }
} // namespace nx::core
