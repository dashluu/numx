#pragma once

#include "../graph/graph.h"
#include "../memory/cache_manager.h"

namespace nx::runtime {
    using namespace nx::utils;
    using namespace nx::primitive;
    using namespace nx::graph;
    using namespace nx::memory;

    class RuntimeContext : public std::enable_shared_from_this<RuntimeContext> {
    protected:
        MemoryManagerPtr m_memory_manager;

    public:
        RuntimeContext() = default;
        RuntimeContext(const RuntimeContext &) = delete;
        virtual ~RuntimeContext() = default;
        RuntimeContext &operator=(const RuntimeContext &) = delete;
        MemoryManagerPtr get_memory_manager() const { return m_memory_manager; }
    };

    using RuntimeContextPtr = std::shared_ptr<RuntimeContext>;
} // namespace nx::runtime