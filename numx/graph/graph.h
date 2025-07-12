#pragma once

#include "functional.h"

namespace nx::graph {
    class Graph : public std::enable_shared_from_this<Graph> {
    protected:
        OpPtr m_output;
        std::unordered_set<ArrayId> m_marked;
        std::vector<OpPtr> m_fw_tape;
        std::vector<OpPtr> m_bw_tape;

        void fw_toposort(OpPtr op);
        void bw_toposort(OpPtr op);

    public:
        Graph(OpPtr output) : m_output(output) {}
        Graph(const Graph &) = delete;
        virtual ~Graph() = default;
        Graph &operator=(const Graph &) = delete;
        OpPtr get_output() const { return m_output; }
        void forward();
        void backward();
        const std::string str() const;
        std::vector<OpPtr>::const_iterator fw_begin() const { return m_fw_tape.cbegin(); }
        std::vector<OpPtr>::const_iterator fw_end() const { return m_fw_tape.cend(); }
        std::vector<OpPtr>::const_iterator bw_begin() const { return m_bw_tape.cbegin(); }
        std::vector<OpPtr>::const_iterator bw_end() const { return m_bw_tape.cend(); }
    };

    using GraphPtr = std::shared_ptr<Graph>;
} // namespace nx::graph