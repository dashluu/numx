#pragma once

#include "profiler.h"
#include "runtime_context.h"

namespace nx::runtime {
    class Runner : public std::enable_shared_from_this<Runner> {
    protected:
        GraphPtr m_graph;
        RuntimeContextPtr m_ctx;
        ProfilerPtr m_profiler;

        virtual void run_full_kernel(OpPtr op, isize constant) = 0;
        virtual void run_arange_kernel(OpPtr op, isize start, isize step) = 0;
        virtual void run_binary_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) = 0;
        virtual void run_gemm_kernel(OpPtr l_op, OpPtr r_op, OpPtr out_op) = 0;
        virtual void run_unary_kernel(OpPtr in_op, OpPtr out_op) = 0;
        virtual void run_copy_kernel(OpPtr in_op, OpPtr out_op) = 0;
        virtual void run_reduce_all_kernel(OpPtr in_op, OpPtr out_op) = 0;
        virtual void run_reduce_col_kernel(OpPtr in_op, OpPtr out_op) = 0;
        virtual void run_initializer_op(OpPtr op) = 0;
        virtual void run_unary_op(OpPtr op) = 0;
        virtual void run_binary_op(OpPtr op) = 0;
        virtual void run_transform_op(OpPtr op) = 0;
        virtual void run_reduce_op(OpPtr op) = 0;
        void run_op(OpPtr op);

    public:
        Runner(GraphPtr graph, RuntimeContextPtr ctx, ProfilerPtr profiler) : m_graph(graph), m_ctx(ctx), m_profiler(profiler) {}
        Runner(const Runner &) = delete;
        virtual ~Runner() = default;
        Runner &operator=(const Runner &) = delete;
        ProfilerPtr get_profiler() { return m_profiler; }
        void forward();
        void backward();
    };

    using RunnerPtr = std::shared_ptr<Runner>;
} // namespace nx::runtime