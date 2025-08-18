#pragma once

#include "../utils.h"

namespace nx::primitive {
    using namespace nx::utils;

    struct ArrayId {
    private:
        isize m_val;

    public:
        ArrayId(isize val) : m_val(val) {}
        ArrayId(const ArrayId &) = default;
        ArrayId(ArrayId &&) noexcept = default;
        ~ArrayId() = default;
        ArrayId &operator=(const ArrayId &) = default;
        ArrayId &operator=(ArrayId &&) noexcept = default;
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
        ArrayIdGenerator(ArrayIdGenerator &&) noexcept = delete;
        ~ArrayIdGenerator() = default;
        ArrayIdGenerator &operator=(const ArrayIdGenerator &) = delete;
        ArrayIdGenerator &operator=(ArrayIdGenerator &&) noexcept = delete;
        ArrayId next() { return ArrayId(s_counter++); }
    };

    inline isize ArrayIdGenerator::s_counter = 1;
} // namespace nx::primitive

namespace std {
    template <>
    struct hash<nx::primitive::ArrayId> {
        size_t operator()(const nx::primitive::ArrayId &id) const {
            return hash<nx::primitive::isize>()(id.get_data());
        }
    };
} // namespace std

namespace std {
    template <>
    struct formatter<nx::primitive::ArrayId> : formatter<string> {
        auto format(const nx::primitive::ArrayId &id, format_context &ctx) const {
            return formatter<string>::format(id.str(), ctx);
        }
    };
} // namespace std