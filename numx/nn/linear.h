#pragma once

#include "../random/random.h"
#include "functional.h"
#include "module.h"

namespace nx::nn {
    using namespace nx::random;

    class Linear : public Module {
    private:
        Array m_weight;
        Array m_bias;
        Array m_weight_view;
        std::optional<Array> m_bias_view;

    public:
        Linear(isize in_features, isize out_features, bool bias = true) {
            m_weight = kaiming_uniform({out_features, in_features});
            m_weight.eval();
            m_weight_view = m_weight;
            m_parameters.push_back(&m_weight_view);

            if (bias) {
                auto [fan_in, fan_out] = compute_fan_in_and_fan_out(m_weight_view);
                float bound = 1.0f / std::sqrt(fan_in);
                m_bias = uniform({out_features}, -bound, bound);
                m_bias.eval();
                m_bias_view = m_bias;
                m_parameters.push_back(&m_bias_view.value());
            }
        }

        ~Linear() = default;
        Array get_weight() { return m_weight_view; }
        std::optional<Array> get_bias() { return m_bias_view; }
        Array forward(const Array &x) override { return m_bias_view.has_value() ? linear_with_bias(x, m_weight_view, m_bias_view.value()) : linear(x, m_weight_view); }
    };
} // namespace nx::nn