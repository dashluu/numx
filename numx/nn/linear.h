#pragma once

#include "../random/random.h"
#include "functional.h"
#include "module.h"

namespace nx::nn {
    using namespace nx::random;

    class Linear : public Module {
    private:
        Array m_weight_holder;
        Array m_weight;
        Array m_bias_holder;
        std::optional<Array> m_bias;

    public:
        Linear(isize in_features, isize out_features, bool bias = true) {
            m_weight_holder = kaiming_uniform({out_features, in_features});
            m_weight_holder.eval();
            m_weight = m_weight_holder.detach();
            m_parameters.push_back(m_weight);

            if (bias) {
                auto [fan_in, fan_out] = compute_fan_in_and_fan_out(m_weight);
                float bound = 1.0f / std::sqrt(fan_in);
                m_bias_holder = uniform({out_features}, -bound, bound);
                m_bias_holder.eval();
                this->m_bias = m_bias_holder.detach();
                m_parameters.push_back(this->m_bias.value());
            }
        }

        ~Linear() = default;
        Array get_weight() { return m_weight; }
        std::optional<Array> get_bias() { return m_bias; }
        Array forward(const Array &x) override { return m_bias.has_value() ? linear_with_bias(x, m_weight, m_bias.value()) : linear(x, m_weight); }
    };
} // namespace nx::nn