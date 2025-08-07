#include "graph_profiler.h"

namespace nx::graph {
    void GraphProfiler::stream_node(std::ostream &stream, OpPtr op) {
        const ArrayData &data = op->get_data();
        stream << "{";
        stream << "\"id\": \"" << data.get_id() << "\",";
        stream << "\"label\": \"" << op->dump() << "\",";
        stream << "\"type\": \"" << op->optype_str() << "\",";
        stream << "\"op\": \"" << op->get_opname() << "\",";
        stream << "\"view\": \"(" << join_nums(data.get_view()) << ")\",";
        stream << "\"dtype\": \"" << data.get_dtype()->str() << "\",";

        if (data.get_buffer().is_view()) {
            stream << "\"size\": 0";
        } else {
            stream << "\"size\": " << data.get_buffer().get_size();
        }

        stream << "}";
    }

    bool GraphProfiler::stream_edge(std::ostream &stream, OpPtr op) {
        const ArrayId &id = op->get_data().get_id();

        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            return false;
        }
        case Optype::UNARY: {
            UnaryOpPtr unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        case Optype::BINARY: {
            BinaryOpPtr binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            stream << "[\"" << id << "\", \"" << lhs->get_data().get_id() << "\"],";
            stream << "[\"" << id << "\", \"" << rhs->get_data().get_id() << "\"]";
            return true;
        }
        case Optype::TRANSFORM: {
            TransformOpPtr transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        default: {
            ReduceOpPtr reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            stream << "[\"" << id << "\", \"" << operand->get_data().get_id() << "\"]";
            return true;
        }
        }
    }

    void GraphProfiler::save_profile(GraphPtr graph, const std::string &file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw UnableToOpenFileToSaveGraphProfile(file_name);
        }

        stream_profile(graph, file);
        file.close();
    }

    void GraphProfiler::stream_profile(GraphPtr graph, std::ostream &stream) {
        if (!stream) {
            throw InvalidGraphProfileStream();
        }

        if (graph->fw_tape_size() == 0) {
            return;
        }

        stream << "{\"nodes\": [";
        size_t i = 0;
        bool logged;

        for (auto iter = graph->fw_begin(); iter != graph->fw_end(); ++iter) {
            stream_node(stream, *iter);

            if (++i < graph->fw_tape_size()) {
                stream << ",";
            }
        }

        if (graph->bw_tape_size() > 0) {
            i = 0;
            stream << ",";

            for (auto iter = graph->bw_begin(); iter != graph->bw_end(); ++iter) {
                stream_node(stream, *iter);

                if (++i < graph->bw_tape_size()) {
                    stream << ",";
                }
            }
        }

        i = 0;
        stream << "],\"edges\": [";

        for (auto iter = graph->fw_begin(); iter != graph->fw_end(); ++iter) {
            logged = stream_edge(stream, *iter);
            ++i;

            if (logged && i < graph->count_fw_edges()) {
                stream << ",";
            }
        }

        if (graph->count_bw_edges() > 0) {
            i = 0;
            stream << ",";

            for (auto iter = graph->bw_begin(); iter != graph->bw_end(); ++iter) {
                logged = stream_edge(stream, *iter);
                ++i;

                if (logged && i < graph->count_bw_edges()) {
                    stream << ",";
                }
            }
        }

        stream << "]}";
    }
} // namespace nx::graph