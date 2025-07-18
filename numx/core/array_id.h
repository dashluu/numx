#pragma once

#include "../utils.h"

namespace nx::core {
    using namespace nx::utils;

    struct ArrayId {
    private:
        isize m_val;

    public:
        ArrayId(isize val) : m_val(val) {}
        ArrayId(const ArrayId &id) : m_val(id.m_val) {}
        ~ArrayId() = default;

        ArrayId &operator=(const ArrayId &id) {
            m_val = id.m_val;
            return *this;
        }

        isize get_data() const { return m_val; }
        bool operator==(const ArrayId &id) const { return m_val == id.m_val; }
        auto operator<=>(const ArrayId &id) const { return m_val <=> id.m_val; }
        const std::string str() const { return std::to_string(m_val); }
        friend std::ostream &operator<<(std::ostream &os, const ArrayId &id) { return os << id.str(); }
    };

    struct ArrayIdGenerator {
    private:
        static isize s_counter;

    public:
        ArrayIdGenerator() = default;
        ArrayIdGenerator(const ArrayIdGenerator &) = delete;
        ~ArrayIdGenerator() = default;
        ArrayIdGenerator &operator=(const ArrayIdGenerator &) = delete;
        ArrayId generate() { return ArrayId(s_counter++); }
    };

    inline isize ArrayIdGenerator::s_counter = 1;
} // namespace nx::core

namespace std {
    template <>
    struct hash<nx::core::ArrayId> {
        std::size_t operator()(const nx::core::ArrayId &id) const {
            return std::hash<nx::core::isize>()(id.get_data());
        }
    };
} // namespace std

namespace std {
    template <>
    struct formatter<nx::core::ArrayId> : formatter<string> {
        auto format(const nx::core::ArrayId &id, format_context &ctx) const {
            return formatter<string>::format(id.str(), ctx);
        }
    };
} // namespace std