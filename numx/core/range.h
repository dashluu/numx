#pragma once

#include "../utils.h"

namespace nx::core {
    using namespace nx::utils;

    struct Range {
    private:
        isize m_start;
        // Stop is exclusive
        isize m_stop;
        isize m_step;

    public:
        Range(isize start, isize stop, isize step = 1) : m_start(start), m_stop(stop), m_step(step) {}
        Range(const Range &range) : Range(range.m_start, range.m_stop, range.m_step) {}

        Range &operator=(const Range &range) {
            m_start = range.m_start;
            m_stop = range.m_stop;
            m_step = range.m_step;
            return *this;
        }

        bool operator==(const Range &range) const { return m_start == range.m_start && m_stop == range.m_stop && m_step == range.m_step; }
        isize get_start() const { return m_start; }
        isize get_stop() const { return m_stop; }
        isize get_step() const { return m_step; }
        std::string str() const { return std::format("({},{},{})", m_start, m_stop, m_step); }
    };

    using RangeVec = std::vector<Range>;
} // namespace nx::core