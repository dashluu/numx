#pragma once

#include "../utils.h"

namespace nx::core {
    using namespace nx::utils;

    enum struct DtypeName {
        F32,
        F64,
        I8,
        I16,
        I32,
        I64,
        B8
    };

    enum struct DtypeCategory {
        FLOAT = 1,
        INT = 2,
        BOOL = 4,
        ALL = FLOAT | INT | BOOL,
        NUMERIC = FLOAT | INT
    };

    struct Dtype {
    private:
        DtypeName m_name;
        DtypeCategory m_category;
        isize m_size;

    public:
        Dtype(DtypeName name, DtypeCategory category, isize size) : m_name(name), m_category(category), m_size(size) {}
        Dtype(const Dtype &) = delete;
        virtual ~Dtype() = default;
        Dtype &operator=(const Dtype &) = delete;
        DtypeName get_name() const { return m_name; }
        DtypeCategory get_category() const { return m_category; }
        virtual const std::string get_name_str() const = 0;
        bool is_float() const { return m_category == DtypeCategory::FLOAT; }
        bool is_int() const { return m_category == DtypeCategory::INT; }
        bool is_bool() const { return m_category == DtypeCategory::BOOL; }
        bool has_category(DtypeCategory category) const { return static_cast<int>(m_category) & static_cast<int>(category); }
        bool is_numeric() const { return has_category(DtypeCategory::NUMERIC); }
        isize get_size() const { return m_size; }
        bool operator==(const Dtype &dtype) const { return m_name == dtype.m_name; }
        std::string str() const { return get_name_str(); }
        virtual std::string value_str(uint8_t *ptr) const = 0;
        virtual std::string value_str(isize val) const = 0;
        virtual isize bit_cast(uint8_t *ptr) const = 0;
        virtual isize one() const = 0;
        virtual isize max() const = 0;
        virtual isize min() const = 0;
    };

    template <class T>
    struct Float : public Dtype {
    public:
        Float(DtypeName name, isize size) : Dtype(name, DtypeCategory::FLOAT, size) {}

        std::string value_str(uint8_t *ptr) const override {
            T val = *reinterpret_cast<T *>(ptr);
            if (0 < val && val <= 1e-5) {
                return std::format("{:.4e}", val);
            }
            return std::format("{:.4f}", val);
        }
    };

    template <class T>
    struct Int : public Dtype {
    public:
        Int(DtypeName name, isize size) : Dtype(name, DtypeCategory::INT, size) {}
        std::string value_str(uint8_t *ptr) const override { return std::to_string(*reinterpret_cast<T *>(ptr)); }
        std::string value_str(isize val) const override { return std::to_string(val); }
        isize bit_cast(uint8_t *ptr) const override { return *reinterpret_cast<T *>(ptr); }
        isize one() const override { return 1; }
        isize max() const override { return std::numeric_limits<T>::max(); }
        isize min() const override { return std::numeric_limits<T>::min(); }
    };

    struct F32 : public Float<float> {
    public:
        F32() : Float<float>(DtypeName::F32, 4) {}
        const std::string get_name_str() const override { return "f32"; }
        std::string value_str(isize val) const override { return std::to_string(std::bit_cast<float>(static_cast<int>(val))); }
        isize bit_cast(uint8_t *ptr) const override { return std::bit_cast<int>(*reinterpret_cast<float *>(ptr)); }
        isize one() const override { return std::bit_cast<int>(1.0f); }
        isize max() const override { return std::bit_cast<int>(std::numeric_limits<float>::infinity()); }
        isize min() const override { return std::bit_cast<int>(-std::numeric_limits<float>::infinity()); }
    };

    struct F64 : public Float<double> {
    public:
        F64() : Float<double>(DtypeName::F64, 8) {}
        const std::string get_name_str() const override { return "f64"; }
        std::string value_str(isize val) const override { return std::to_string(std::bit_cast<double>(static_cast<int64_t>(val))); }
        isize bit_cast(uint8_t *ptr) const override { return std::bit_cast<int64_t>(*reinterpret_cast<double *>(ptr)); }
        isize one() const override { return std::bit_cast<int64_t>(1.0); }
        isize max() const override { return std::bit_cast<int64_t>(std::numeric_limits<double>::infinity()); }
        isize min() const override { return std::bit_cast<int64_t>(-std::numeric_limits<double>::infinity()); }
    };

    struct I8 : public Int<int8_t> {
    public:
        I8() : Int<int8_t>(DtypeName::I8, 1) {}
        const std::string get_name_str() const override { return "i8"; }
    };

    struct I16 : public Int<int16_t> {
    public:
        I16() : Int<int16_t>(DtypeName::I16, 2) {}
        const std::string get_name_str() const override { return "i16"; }
    };

    struct I32 : public Int<int32_t> {
    public:
        I32() : Int<int32_t>(DtypeName::I32, 4) {}
        const std::string get_name_str() const override { return "i32"; }
    };

    struct I64 : public Int<int64_t> {
    public:
        I64() : Int<int64_t>(DtypeName::I64, 8) {}
        const std::string get_name_str() const override { return "i64"; }
    };

    struct Bool : public Dtype {
    public:
        Bool() : Dtype(DtypeName::B8, DtypeCategory::BOOL, 1) {}
        const std::string get_name_str() const override { return "b8"; }
        std::string value_str(uint8_t *ptr) const override { return *ptr ? "True" : "False"; }
        std::string value_str(isize val) const override { return std::to_string(static_cast<bool>(val)); }
        isize bit_cast(uint8_t *ptr) const override { return *ptr; }
        isize one() const override { return 1; }
        isize max() const override { return std::numeric_limits<bool>::max(); }
        isize min() const override { return std::numeric_limits<bool>::min(); }
    };

    using DtypePtr = const Dtype *;

    inline const F32 f32;
    inline const F64 f64;
    inline const I8 i8;
    inline const I16 i16;
    inline const I32 i32;
    inline const I64 i64;
    inline const Bool b8;

    inline std::vector<DtypePtr> all_dtypes = {&b8, &i32, &f32};

    inline DtypePtr float_dtype_by_dtype(DtypePtr dtype) {
        if (!dtype->is_numeric()) {
            return nullptr;
        }

        if (dtype->get_size() == 4) {
            return &f32;
        }

        return &f64;
    }

    template <Numeric T>
    isize dtype_bitcast_numeric(DtypePtr dtype, T constant) {
        if (dtype->is_float()) {
            switch (dtype->get_size()) {
            default:
                return std::bit_cast<int>(static_cast<float>(constant));
            }
        } else if (dtype->is_int()) {
            return static_cast<isize>(constant);
        }

        return static_cast<bool>(constant);
    }
} // namespace nx::core