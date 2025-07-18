#include "profiler.h"

namespace nx::runtime {
    void Profiler::log_node(OpPtr op, std::ostream &stream) {
        const ArrayData &data = op->get_data();
        stream << "{";
        stream << "\"id\": \"" << data.get_id() << "\",";
        stream << "\"label\": \"" << op->repr() << "\",";
        stream << "\"type\": \"" << op->optype_str() << "\",";
        stream << "\"op\": \"" << op->get_opname() << "\",";
        stream << "\"view\": \"(" << join_nums(data.get_view()) << ")\",";
        stream << "\"dtype\": \"" << data.get_dtype()->str() << "\",";

        if (m_arrays.contains(data)) {
            stream << "\"size\": " << data.m_buffer.get_size();
        } else {
            stream << "\"size\": 0";
        }

        stream << "}";
    }

    bool Profiler::log_edge(OpPtr op, std::ostream &stream) {
        const ArrayId &id = op->get_data().get_id();

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            return false;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            stream << "[\"" << id << "\", \"" << lhs->get_data().get_id() << "\"],";
            stream << "[\"" << id << "\", \"" << rhs->get_data().get_id() << "\"]";
            return true;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        default: {
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        }
    }

    void Profiler::log_memory(std::ostream &stream) {
        if (!has_memory_leaks()) {
            stream << "No array memory leaks..." << std::endl;
            return;
        }

        stream << "Array memory leak log:" << std::endl;

        for (const auto &array : m_arrays) {
            stream << array.get_id() << ": " << array.m_buffer.get_ptr() << ", " << array.m_buffer.get_size() << " bytes" << std::endl;
        }
    }

    void Profiler::log_graph(GraphPtr graph, std::ostream &stream) {
        if (graph->fw_tape_size() == 0) {
            return;
        }

        stream << "{\"nodes\": [";
        size_t i = 0;
        bool logged;

        for (auto iter = graph->fw_begin(); iter != graph->fw_end(); ++iter) {
            log_node(*iter, stream);

            if (++i < graph->fw_tape_size()) {
                stream << ",";
            }
        }

        if (graph->bw_tape_size() > 0) {
            i = 0;
            stream << ",";

            for (auto iter = graph->bw_begin(); iter != graph->bw_end(); ++iter) {
                log_node(*iter, stream);

                if (++i < graph->bw_tape_size()) {
                    stream << ",";
                }
            }
        }

        i = 0;
        stream << "],\"edges\": [";

        for (auto iter = graph->fw_begin(); iter != graph->fw_end(); ++iter) {
            logged = log_edge(*iter, stream);
            ++i;

            if (logged && i < graph->count_fw_edges()) {
                stream << ",";
            }
        }

        if (graph->count_bw_edges() > 0) {
            i = 0;
            stream << ",";

            for (auto iter = graph->bw_begin(); iter != graph->bw_end(); ++iter) {
                logged = log_edge(*iter, stream);
                ++i;

                if (logged && i < graph->count_bw_edges()) {
                    stream << ",";
                }
            }
        }

        stream << "]}";
    }
} // namespace nx::runtime