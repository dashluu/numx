#include "graph.h"

namespace nx::graph {
    const std::string Graph::str() const {
        if (m_fw_tape.empty()) {
            return "Empty graph...";
        }

        std::string s = "";

        if (!m_fw_tape.empty()) {
            s += "Forward toposort:\n";

            for (auto &op : m_fw_tape) {
                s += op->str() + "\n";
            }

            s += "\n";
        }

        if (!m_bw_tape.empty()) {
            s += "Backward toposort:\n";

            for (auto &op : m_bw_tape) {
                s += op->str() + "\n";
            }

            s += "\n";
        }

        return s;
    }
} // namespace nx::graph