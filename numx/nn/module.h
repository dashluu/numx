#pragma once

#include "../core/array.h"

namespace nx::nn {
    using namespace nx::core;

    class Module {
    protected:
        ArrayVec m_parameters;

    public:
        Module() = default;
        Module(const Module &module) = delete;
        virtual ~Module() = default;
        Module &operator=(const Module &module) = delete;
        ArrayVec::const_iterator begin() const { return m_parameters.cbegin(); }
        ArrayVec::const_iterator end() const { return m_parameters.cend(); }
        virtual Array forward(const Array &x) = 0;
        Array operator()(const Array &x) { return forward(x); }
        const ArrayVec &get_parameters() const { return m_parameters; }
    };
} // namespace nx::nn