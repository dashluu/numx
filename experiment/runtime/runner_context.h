#pragma once

#include "pool_allocator.h"

namespace nx::runtime {
    class RunnerContext : public std::enable_shared_from_this<RunnerContext> {
    protected:
        PoolAllocatorPtr m_allocator;

    public:
        RunnerContext() = default;
        RunnerContext(const RunnerContext &) = delete;
        virtual ~RunnerContext() = default;
        RunnerContext &operator=(const RunnerContext &) = delete;
        PoolAllocatorPtr get_allocator() const { return m_allocator; }
    };

    using RunnerContextPtr = std::shared_ptr<RunnerContext>;
} // namespace nx::runtime