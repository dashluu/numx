#pragma once

#include "../utils.h"

namespace nx::memory {
    using namespace nx::utils;
    struct ResourceList;
    struct ResourceListIterator;

    struct ResourceNode {
    protected:
        ResourceNode *m_prev = nullptr;
        ResourceNode *m_next = nullptr;

    public:
        friend struct ResourceList;
        friend struct ResourceListIterator;
        ResourceNode() = default;
        ResourceNode(const ResourceNode &) = delete;
        virtual ~ResourceNode() = default;
        ResourceNode &operator=(const ResourceNode &) = delete;
    };

    struct ResourceListIterator {
    private:
        ResourceNode *m_current;
        ResourceNode *m_next;

    public:
        ResourceListIterator(ResourceNode *current) : m_current(current) {
            m_next = m_current ? m_current->m_next : nullptr;
        }

        ResourceListIterator(const ResourceListIterator &) = default;
        ~ResourceListIterator() = default;
        ResourceListIterator &operator=(const ResourceListIterator &) = default;
        bool operator==(const ResourceListIterator &iterator) const { return m_current == iterator.m_current; }
        ResourceNode *operator*() const { return m_current; }

        ResourceListIterator &operator++() {
            m_current = m_next;
            m_next = m_current ? m_current->m_next : nullptr;
            return *this;
        }
    };

    struct ResourceListRange {
    private:
        ResourceNode *m_start;
        ResourceNode *m_end = nullptr;

    public:
        ResourceListRange(ResourceNode *start) : m_start(start) {}
        ResourceListRange(const ResourceListRange &) = default;
        ~ResourceListRange() = default;
        ResourceListRange &operator=(const ResourceListRange &) = default;
        ResourceListIterator begin() const { return ResourceListIterator(m_start); }
        ResourceListIterator end() const { return ResourceListIterator(m_end); }
    };

    struct ResourceList {
    private:
        ResourceNode *m_head = nullptr;

    public:
        friend struct ResourceListIterator;
        ResourceList() = default;
        ResourceList(const ResourceList &) = delete;
        ~ResourceList() = default;
        ResourceList &operator=(const ResourceList &) = delete;
        bool empty() const { return !m_head; }
        ResourceNode *peek() const { return m_head; }
        ResourceListIterator begin() const { return ResourceListIterator(m_head); }
        ResourceListIterator end() const { return ResourceListIterator(nullptr); }
        ResourceListRange range() const { return ResourceListRange(m_head); }
        void push(ResourceNode *item);
        ResourceNode *pop();
        void unlink(ResourceNode *item);
    };

    using ResourceListPtr = std::shared_ptr<ResourceList>;
} // namespace nx::memory