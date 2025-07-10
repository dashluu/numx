#include "graph.h"

namespace nx::graph {
    void Graph::fw_toposort(OpPtr op) {
        const ArrayId &id = op->get_descriptor().get_id();

        if (m_toposort_visited.contains(id)) {
            return;
        }

        m_toposort_visited.insert(id);

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            m_fw_ops.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            fw_toposort(operand);
            m_fw_ops.push_back(op);
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            fw_toposort(lhs);
            fw_toposort(rhs);
            m_fw_ops.push_back(op);
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            fw_toposort(operand);
            m_fw_ops.push_back(op);
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            fw_toposort(operand);
            m_fw_ops.push_back(op);
            break;
        }
        }
    }

    void Graph::bw_toposort(OpPtr op) {
        const ArrayId &id = op->get_descriptor().get_id();

        if (m_toposort_visited.contains(id)) {
            return;
        }

        m_toposort_visited.insert(id);

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            m_bw_ops.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            bw_toposort(operand);
            m_bw_ops.push_back(op);
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            bw_toposort(lhs);
            bw_toposort(rhs);
            m_bw_ops.push_back(op);
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            bw_toposort(operand);
            m_bw_ops.push_back(op);
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            bw_toposort(operand);
            m_bw_ops.push_back(op);
            break;
        }
        }
    }

    void Graph::forward() {
        if (m_fw_ops.empty()) {
            fw_toposort(m_output);
        }
    }

    void Graph::backward() {
        if (m_fw_ops.empty()) {
            throw std::runtime_error("Graph has not been forwarded.");
        }

        if (m_bw_ops.empty()) {
            if (m_output->get_descriptor().get_numel() > 1) {
                throw std::invalid_argument(std::format("Array {} must be a singleton to do gradient backpropation.", m_output->get_descriptor().get_id().str()));
            }

            // Initialize output's gradient with 1's
            if (m_output->is_grad_enabled()) {
                m_output->one_grad();
            }

            // Initialize gradient structure without allocating buffer memory
            for (auto &op : std::views::reverse(m_fw_ops)) {
                if (op->is_grad_enabled()) {
                    op->backward();
                }
            }

            // Order the gradient to be computed
            for (auto &op : std::views::reverse(m_fw_ops)) {
                // grad is null when backward is not implemented for op or that gradient is disabled
                if (op->get_grad_fn() != nullptr) {
                    bw_toposort(op->get_grad_fn());
                }
            }
        }
    }
} // namespace nx::graph