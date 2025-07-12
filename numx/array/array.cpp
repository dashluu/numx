#include "array.h"

namespace nx::array {
    Backend &Array::get_backend() {
        static Backend backend;
        backend.init();
        return backend;
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
