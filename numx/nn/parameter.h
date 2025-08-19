#pragma once

#include "../core/array.h"

namespace nx::nn {
    using namespace nx::primitive;
    using namespace nx::core;

    struct Parameter : public Array {
    public:
        Parameter(const ShapeView &shape, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) : Array(nx::graph::empty(shape, dtype, nx::core::get_device(device_name))) {
            set_parameter(true);
        }

        Parameter(const Array &array) : Array(array.detach()) { set_parameter(true); }
        Parameter(Array &&array) noexcept : Array(std::move(array)) { set_parameter(true); }
        Parameter(const Parameter &) = default;
        Parameter(Parameter &&) noexcept = default;
        ~Parameter() = default;
        Parameter &operator=(const Parameter &) = default;
        Parameter &operator=(Parameter &&) noexcept = default;
    };

    using ParameterPtr = std::shared_ptr<Parameter>;
    using ParameterVector = std::vector<Parameter>;
    using ParameterPtrVector = std::vector<ParameterPtr>;
} // namespace nx::nn