#pragma once

#include "parameter.h"

namespace nx::nn {
    class Module {
    protected:
        ParameterPtrVector m_params;

        void add_parameter(ParameterPtr param) { m_params.push_back(param); }

    public:
        Module() = default;
        Module(const Module &) = delete;
        Module(Module &&) noexcept = delete;
        virtual ~Module() = default;
        Module &operator=(const Module &) = delete;
        Module &operator=(Module &&) noexcept = delete;
        const ParameterPtrVector &get_parameters() const { return m_params; }
        ParameterPtrVector::const_iterator begin() const { return m_params.cbegin(); }
        ParameterPtrVector::const_iterator end() const { return m_params.cend(); }
        virtual Array forward(const Array &x) = 0;
        Array operator()(const Array &x) { return forward(x); }
    };
} // namespace nx::nn