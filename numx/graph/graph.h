#pragma once

#include "../primitive/functional.h"

namespace nx::graph {
    using namespace nx::utils;
    using namespace nx::primitive;

    class Graph : public std::enable_shared_from_this<Graph> {
    protected:
        OpPtr m_output;
        std::unordered_set<ArrayId> m_marked;
        std::vector<OpPtr> m_fw_tape;
        std::vector<OpPtr> m_bw_tape;
        size_t m_num_fw_edges = 0;
        size_t m_num_bw_edges = 0;

        void fw_toposort(OpPtr op);
        void bw_toposort(OpPtr op);

    public:
        Graph(OpPtr output) : m_output(output) {}
        Graph(const Graph &) = delete;
        virtual ~Graph() = default;
        Graph &operator=(const Graph &) = delete;
        OpPtr get_output() const { return m_output; }
        size_t fw_tape_size() const { return m_fw_tape.size(); }
        size_t bw_tape_size() const { return m_bw_tape.size(); }
        size_t count_fw_edges() const { return m_num_fw_edges; }
        size_t count_bw_edges() const { return m_num_bw_edges; }
        void forward();
        void backward();
        void clear_grad();
        const std::string str() const;
        friend std::ostream &operator<<(std::ostream &os, const Graph &graph) { return os << graph.str(); }
        std::vector<OpPtr>::const_iterator fw_begin() const { return m_fw_tape.cbegin(); }
        std::vector<OpPtr>::const_iterator fw_end() const { return m_fw_tape.cend(); }
        std::vector<OpPtr>::const_iterator bw_begin() const { return m_bw_tape.cbegin(); }
        std::vector<OpPtr>::const_iterator bw_end() const { return m_bw_tape.cend(); }
    };

    using GraphPtr = std::shared_ptr<Graph>;
} // namespace nx::graph

namespace std {
    template <>
    struct formatter<nx::graph::Graph> : formatter<string> {
        auto format(const nx::graph::Graph &graph, format_context &ctx) const {
            return formatter<string>::format(graph.str(), ctx);
        }
    };
} // namespace std