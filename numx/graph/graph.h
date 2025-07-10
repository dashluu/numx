#pragma once

#include "../core/op_impl.h"

namespace nx::graph {
    using namespace nx::core;

    class Graph : public std::enable_shared_from_this<Graph> {
    protected:
        OpPtr m_output;
        std::unordered_set<ArrayId> m_toposort_visited;
        std::vector<OpPtr> m_fw_ops;
        std::vector<OpPtr> m_bw_ops;

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
        std::vector<OpPtr>::const_iterator fw_begin() const { return m_fw_ops.cbegin(); }
        std::vector<OpPtr>::const_iterator fw_end() const { return m_fw_ops.cend(); }
        std::vector<OpPtr>::const_reverse_iterator fw_rbegin() const { return m_fw_ops.crbegin(); }
        std::vector<OpPtr>::const_reverse_iterator fw_rend() const { return m_fw_ops.crend(); }
        std::vector<OpPtr>::const_iterator bw_begin() const { return m_bw_ops.cbegin(); }
        std::vector<OpPtr>::const_iterator bw_end() const { return m_bw_ops.cend(); }
        std::vector<OpPtr>::const_reverse_iterator bw_rbegin() const { return m_bw_ops.crbegin(); }
        std::vector<OpPtr>::const_reverse_iterator bw_rend() const { return m_bw_ops.crend(); }
    };

    using ComputeGraphPtr = std::shared_ptr<Graph>;
} // namespace nx::graph