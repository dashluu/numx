#include "resource_list.h"

namespace nx::primitive {
    void ResourceList::push(Resource *item) {
        if (empty()) {
            m_head = item;
            return;
        }

        item->m_next = m_head;
        m_head->m_prev = item;
        m_head = item;
    }

    Resource *ResourceList::pop() {
        if (empty()) {
            return nullptr;
        }

        Resource *item = m_head;
        m_head = m_head->m_next;
        item->m_next = nullptr;

        if (m_head) {
            m_head->m_prev = nullptr;
        }

        return item;
    }

    void ResourceList::unlink(Resource *item) {
        if (!item) {
            return;
        }

        if (item == m_head) {
            pop();
            return;
        }

        Resource *prev = item->m_prev;
        Resource *next = item->m_next;
        item->m_prev = nullptr;
        item->m_next = nullptr;

        if (prev) {
            prev->m_next = next;
        }

        if (next) {
            next->m_prev = prev;
        }
    }
} // namespace nx::primitive