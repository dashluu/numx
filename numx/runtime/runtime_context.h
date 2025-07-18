#pragma once

#include "../graph/graph.h"
#include "../memory/allocator.h"

namespace nx::runtime {
    using namespace nx::utils;
    using namespace nx::core;
    using namespace nx::graph;
    using namespace nx::memory;

    class RuntimeContext : public std::enable_shared_from_this<RuntimeContext> {
    protected:
        AllocatorPtr m_allocator;

    public:
        RuntimeContext() = default;
        RuntimeContext(const RuntimeContext &) = delete;
        virtual ~RuntimeContext() = default;
        RuntimeContext &operator=(const RuntimeContext &) = delete;
        AllocatorPtr get_allocator() const { return m_allocator; }
    };

    using RuntimeContextPtr = std::shared_ptr<RuntimeContext>;
} // namespace nx::runtime