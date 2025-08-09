#pragma once

#include "../core/array.h"

namespace nx::nn {
    using namespace nx::core;

    class Module {
    protected:
        ParameterVector m_parameters;

    public:
        Module() = default;
        Module(const Module &module) = delete;
        virtual ~Module() = default;
        Module &operator=(const Module &module) = delete;
        const ParameterVector &get_parameters() const { return m_parameters; }
        ParameterVector::const_iterator begin() const { return m_parameters.cbegin(); }
        ParameterVector::const_iterator end() const { return m_parameters.cend(); }
        virtual Array forward(const Array &x) = 0;
        Array operator()(const Array &x) { return forward(x); }
    };
} // namespace nx::nn