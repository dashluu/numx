#pragma once

#include "../utils.h"

namespace nx::runtime {
    using namespace nx::core;

    struct StoragePool {
    private:
        uint8_t *m_base;
        uint8_t *m_ptr;
        isize m_capacity;

    public:
        StoragePool(uint8_t *base, isize capacity) : m_base(base), m_ptr(base), m_capacity(capacity) {}
        StoragePool(const StoragePool &pool) : m_base(pool.m_base), m_ptr(pool.m_ptr), m_capacity(pool.m_capacity) {}

        StoragePool &operator=(const StoragePool &pool) {
            m_base = pool.m_base;
            m_ptr = pool.m_ptr;
            m_capacity = pool.m_capacity;
            return *this;
        }

        uint8_t *get_base() const { return m_base; }
        uint8_t *get_ptr() const { return m_ptr; }
        isize get_capacity() const { return m_capacity; }
        isize get_size() const { return m_ptr - m_base; }
        void reset() { m_ptr = m_base; }

        uint8_t *alloc(isize size) {
            if (m_ptr + size > m_base + m_capacity) {
                throw std::runtime_error(std::format("Expected to allocate {} bytes but can only allocate {} additional bytes from the pool.", size, m_base + m_capacity - m_ptr));
            }

            uint8_t *ptr = m_ptr;
            m_ptr += size;
            return ptr;
        }
    };

    struct PoolAllocator : public std::enable_shared_from_this<PoolAllocator> {
    public:
        PoolAllocator() = default;
        PoolAllocator(const PoolAllocator &) = delete;
        virtual ~PoolAllocator() = default;
        PoolAllocator &operator=(const PoolAllocator &) = delete;
        virtual StoragePool alloc(isize capacity) = 0;
        virtual void free(const StoragePool &pool) = 0;
    };

    using PoolAllocatorPtr = std::shared_ptr<PoolAllocator>;
} // namespace nx::runtime