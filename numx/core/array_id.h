#pragma once

#include "../utils.h"

namespace nx::core {
    struct ArrayId {
    private:
        isize m_data;

    public:
        ArrayId() : m_data(0) {}
        ArrayId(isize id) : m_data(id) {}
        ArrayId(const ArrayId &id) : m_data(id.m_data) {}
        bool operator==(const ArrayId &id) const { return m_data == id.m_data; }

        ArrayId &operator=(const ArrayId &id) {
            m_data = id.m_data;
            return *this;
        }

        isize get_data() const { return m_data; }
        const std::string str() const { return std::to_string(m_data); }
    };

    struct ArrayIdGenerator {
    private:
        static isize s_counter;

    public:
        ArrayIdGenerator() = default;
        ArrayIdGenerator(const ArrayIdGenerator &) = delete;
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