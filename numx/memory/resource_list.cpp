#include "resource_list.h"

namespace nx::memory {
    void ResourceList::push(ResourceNode *item) {
        if (empty()) {
            m_head = item;
            return;
        }

        item->m_next = m_head;
        m_head->m_prev = item;
        m_head = item;
    }

    ResourceNode *ResourceList::pop() {
        if (empty()) {
            return nullptr;
        }

        ResourceNode *item = m_head;
        m_head = m_head->m_next;
        item->m_next = nullptr;

        if (m_head) {
            m_head->m_prev = nullptr;
        }

        return item;
    }

    void ResourceList::unlink(ResourceNode *item) {
        if (!item) {
            return;
        }

        if (item == m_head) {
            pop();
            return;
        }

        ResourceNode *prev = item->m_prev;
        ResourceNode *next = item->m_next;
        item->m_prev = nullptr;
        item->m_next = nullptr;

        if (prev) {
            prev->m_next = next;
        }

        if (next) {
            next->m_prev = prev;
        }
    }
} // namespace nx::memory