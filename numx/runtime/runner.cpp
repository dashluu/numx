#include "runner.h"

namespace nx::runtime {
    void Runner::run_op(OpPtr op) {
        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            run_initializer_op(op);
            break;
        }
        case Optype::UNARY: {
            run_unary_op(op);
            break;
        }
        case Optype::BINARY: {
            run_binary_op(op);
            break;
        }
        case Optype::TRANSFORM: {
            run_transform_op(op);
            break;
        }
        default: {
            run_reduce_op(op);
            break;
        }
        }
    }

    void Runner::forward() {
        for (auto iter = m_graph->fw_begin(); iter != m_graph->fw_end(); ++iter) {
            run_op(*iter);
        }
    }

    void Runner::backward() {
        for (auto iter = m_graph->bw_begin(); iter != m_graph->bw_end(); ++iter) {
            run_op(*iter);
        }
    }
} // namespace nx::runtime