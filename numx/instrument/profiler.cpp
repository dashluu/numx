#include "profiler.h"

namespace nx::instrument {
    void Profiler::stream_node(std::ostream &stream, OpPtr op) {
        const ArrayData &data = op->get_data();
        stream << "{";
        stream << "\"id\": \"" << data.get_id() << "\",";
        stream << "\"label\": \"" << op->dump() << "\",";
        stream << "\"type\": \"" << op->optype_str() << "\",";
        stream << "\"op\": \"" << op->get_opname() << "\",";
        stream << "\"view\": \"(" << join_nums(data.get_view()) << ")\",";
        stream << "\"dtype\": \"" << data.get_dtype()->str() << "\",";

        if (m_memory_snapshot.find(data.get_id()) == m_memory_snapshot.end()) {
            stream << "\"size\": 0";
        } else {
            stream << "\"size\": " << m_memory_snapshot.at(data.get_id()).get_block().get_size();
        }

        stream << "}";
    }

    bool Profiler::stream_edge(std::ostream &stream, OpPtr op) {
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

    void Profiler::stream_memory_snapshot_info(std::ostream &stream, const MemorySnapshotInfo &snapshot_info) {
        stream << "{";
        stream << "\"size\": " << snapshot_info.get_block().get_size() << ",";
        stream << "\"start\": " << snapshot_info.get_alloc_time().time_since_epoch().count() << ",";
        stream << "\"end\": " << snapshot_info.get_free_time().time_since_epoch().count();
        stream << "}";
    }

    void Profiler::record_alloc(const ArrayData &data) {
        const Block &block = data.m_buffer.get_block();
        m_memory_snapshot.emplace(data.get_id(), MemorySnapshotInfo(block));
        m_memory_usage += block.get_size();
        m_peak_memory = std::max(m_peak_memory, m_memory_usage);
    }

    void Profiler::record_free(const ArrayData &data) {
        MemorySnapshotInfo &snapshot_info = m_memory_snapshot.at(data.get_id());
        snapshot_info.tock();
        m_memory_usage -= snapshot_info.get_block().get_size();
    }

    void Profiler::write_memory_profile(const std::string &file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw std::runtime_error(std::format("Cannot log memory leaks due to failing to open file '{}'...", file_name));
        }

        stream_memory_profile(file);
        file.close();
    }

    void Profiler::write_graph_profile(GraphPtr graph, const std::string &file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            throw std::runtime_error(std::format("Cannot log graph due to failing to open file '{}'...", file_name));
        }

        stream_graph_profile(graph, file);
        file.close();
    }

    void Profiler::stream_memory_profile(std::ostream &stream) {
        stream << "{\"Peak memory\": " << m_peak_memory << ", \"Memory usage\": {";
        size_t snapshot_size = m_memory_snapshot.size();
        size_t num_leaks = 0;
        size_t i = 0;

        for (const auto &[id, snapshot_info] : m_memory_snapshot) {
            stream << "\"" << id << "\":";
            stream_memory_snapshot_info(stream, snapshot_info);

            if (++i < snapshot_size) {
                stream << ",";
            }

            if (snapshot_info.is_alive()) {
                num_leaks++;
            }
        }

        stream << "}, \"Leaks\": {";
        i = 0;

        for (const auto &[id, snapshot_info] : m_memory_snapshot) {
            if (snapshot_info.is_alive()) {
                stream << "\"" << id << "\":";
                stream_memory_snapshot_info(stream, snapshot_info);

                if (++i < num_leaks) {
                    stream << ",";
                }
            }
        }

        stream << "}}";
    }

    void Profiler::stream_graph_profile(GraphPtr graph, std::ostream &stream) {
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
} // namespace nx::instrument