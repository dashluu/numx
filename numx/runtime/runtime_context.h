#pragma once

#include "../graph/graph.h"
#include "memory.h"

namespace nx::runtime {
    using namespace nx::utils;

    class RuntimeContext : public std::enable_shared_from_this<RuntimeContext> {
    protected:
        MemoryPtr m_memory;
        MemoryProfilerPtr m_memory_profiler;

    public:
        explicit RuntimeContext(MemoryProfilerPtr memory_profiler) : m_memory_profiler(memory_profiler) {}
        RuntimeContext(const RuntimeContext &) = delete;
        RuntimeContext(RuntimeContext &&) noexcept = delete;
        virtual ~RuntimeContext() = default;
        RuntimeContext &operator=(const RuntimeContext &) = delete;
        RuntimeContext &operator=(RuntimeContext &&) noexcept = delete;
        MemoryPtr get_memory() const { return m_memory; }
        MemoryProfilerPtr get_memory_profiler() { return m_memory_profiler; }
    };

    using RuntimeContextPtr = std::shared_ptr<RuntimeContext>;
} // namespace nx::runtime