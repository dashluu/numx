#pragma once

#include "../random/random.h"
#include "functional.h"
#include "module.h"

namespace nx::nn {
    using namespace nx::random;

    class Linear : public Module {
    private:
        ArrayPtr m_weight_holder;
        ArrayPtr m_bias_holder;
        ParameterPtr m_weight;
        ParameterPtr m_bias;

    public:
        Linear(isize in_features, isize out_features, bool has_bias = true) {
            Array weight = kaiming_uniform({out_features, in_features});
            m_weight_holder = std::make_shared<Array>(weight);
            m_weight_holder->eval();
            m_weight = std::make_shared<Parameter>(*m_weight_holder);
            add_parameter(m_weight);

            if (has_bias) {
                auto [fan_in, fan_out] = compute_fan_in_and_fan_out(*m_weight);
                float bound = 1.0f / std::sqrt(fan_in);
                Array bias = uniform({out_features}, -bound, bound);
                m_bias_holder = std::make_shared<Array>(bias);
                m_bias_holder->eval();
                m_bias = std::make_shared<Parameter>(*m_bias_holder);
                add_parameter(m_bias);
            }
        }

        ~Linear() = default;
        ParameterPtr get_weight() { return m_weight; }
        ParameterPtr get_bias() { return m_bias; }
        Array forward(const Array &x) override { return m_bias ? linear_with_bias(x, *m_weight, *m_bias) : linear(x, *m_weight); }
    };
} // namespace nx::nn