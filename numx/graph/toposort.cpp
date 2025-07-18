#include "graph.h"

namespace nx::graph {
    void Graph::fw_toposort(OpPtr op) {
        const ArrayId &id = op->get_data().get_id();

        if (m_marked.contains(id)) {
            return;
        }

        m_marked.insert(id);

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            m_fw_tape.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            fw_toposort(operand);
            m_fw_tape.push_back(op);
            num_fw_edges++;
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            fw_toposort(lhs);
            fw_toposort(rhs);
            m_fw_tape.push_back(op);
            num_fw_edges += 2;
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            fw_toposort(operand);
            m_fw_tape.push_back(op);
            num_fw_edges++;
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            fw_toposort(operand);
            m_fw_tape.push_back(op);
            num_fw_edges++;
            break;
        }
        }
    }

    void Graph::bw_toposort(OpPtr op) {
        const ArrayId &id = op->get_data().get_id();

        if (m_marked.contains(id)) {
            return;
        }

        m_marked.insert(id);

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            m_bw_tape.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            bw_toposort(operand);
            m_bw_tape.push_back(op);
            num_bw_edges++;
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            bw_toposort(lhs);
            bw_toposort(rhs);
            m_bw_tape.push_back(op);
            num_bw_edges += 2;
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            bw_toposort(operand);
            m_bw_tape.push_back(op);
            num_bw_edges++;
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            bw_toposort(operand);
            m_bw_tape.push_back(op);
            num_bw_edges++;
            break;
        }
        }
    }

    void Graph::forward() {
        if (m_fw_tape.empty()) {
            fw_toposort(m_output);
        }
    }

    void Graph::backward() {
        if (m_fw_tape.empty()) {
            throw std::runtime_error("Graph has not been forwarded.");
        }

        if (m_bw_tape.empty()) {
            if (m_output->get_data().get_numel() > 1) {
                throw std::runtime_error(std::format("Array {} must be a singleton to do gradient backpropation.", m_output->get_data().get_id()));
            }

            // Initialize output's gradient with 1's
            if (m_output->is_grad_enabled()) {
                m_output->one_grad();
            }

            // Initialize gradient structure without allocating buffer memory
            for (auto &op : std::views::reverse(m_fw_tape)) {
                if (op->is_grad_enabled()) {
                    op->grad_fn();
                }
            }

            // Play tape backward to compute gradient
            for (auto &op : std::views::reverse(m_fw_tape)) {
                // grad is null when backward is not implemented for op or that gradient is disabled
                if (op->get_partial_grad() != nullptr) {
                    bw_toposort(op->get_partial_grad());
                }
            }
        }
    }
} // namespace nx::graph