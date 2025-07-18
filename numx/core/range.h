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
        const std::string str() const { return std::format("({},{},{})", m_start, m_stop, m_step); }
        friend std::ostream &operator<<(std::ostream &os, const Range &range) { return os << range.str(); }
    };

    using RangeVec = std::vector<Range>;
} // namespace nx::core

namespace std {
    template <>
    struct formatter<nx::core::Range> : formatter<string> {
        auto format(const nx::core::Range &range, format_context &ctx) const {
            return formatter<string>::format(range.str(), ctx);
        }
    };
} // namespace std