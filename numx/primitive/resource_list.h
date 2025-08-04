#pragma once

#include "../utils.h"

namespace nx::primitive {
    using namespace nx::utils;
    struct ResourceList;
    struct ResourceListIterator;

    struct Resource {
    protected:
        Resource *m_prev = nullptr;
        Resource *m_next = nullptr;

    public:
        friend struct ResourceList;
        friend struct ResourceListIterator;
        Resource() = default;
        Resource(const Resource &) = delete;
        virtual ~Resource() = default;
        Resource &operator=(const Resource &) = delete;
    };

    struct ResourceListIterator {
    private:
        Resource *m_current;
        Resource *m_next;

    public:
        ResourceListIterator(Resource *current) : m_current(current) {
            m_next = m_current ? m_current->m_next : nullptr;
        }

        ResourceListIterator(const ResourceListIterator &) = default;
        ~ResourceListIterator() = default;
        ResourceListIterator &operator=(const ResourceListIterator &) = default;
        bool operator==(const ResourceListIterator &iterator) const { return m_current == iterator.m_current; }
        Resource *operator*() const { return m_current; }

        ResourceListIterator &operator++() {
            m_current = m_next;
            m_next = m_current ? m_current->m_next : nullptr;
            return *this;
        }
    };

    struct ResourceList {
    private:
        Resource *m_head = nullptr;

    public:
        friend struct ResourceListIterator;
        ResourceList() = default;
        ResourceList(const ResourceList &) = delete;
        ~ResourceList() = default;
        ResourceList &operator=(const ResourceList &) = delete;
        bool empty() const { return !m_head; }
        Resource *peek() const { return m_head; }
        ResourceListIterator begin() const { return ResourceListIterator(m_head); }
        ResourceListIterator end() const { return ResourceListIterator(nullptr); }
        void push(Resource *item);
        Resource *pop();
        void unlink(Resource *item);
    };

    using ResourceListPtr = std::shared_ptr<ResourceList>;
} // namespace nx::primitive