#pragma once

#include "../core/functional.h"
#include "../nn/parameter.h"

namespace nx::optim {
    using namespace nx::core;
    using namespace nx::nn;

    class Optimizer {
    protected:
        float m_learning_rate;
        ArrayVector m_params;
        ArrayVector m_grads;

    public:
        explicit Optimizer(float learning_rate) : m_learning_rate(learning_rate) {}
        Optimizer(const Optimizer &) = delete;
        Optimizer(Optimizer &&) noexcept = delete;
        virtual ~Optimizer() = default;
        Optimizer &operator=(const Optimizer &) = delete;
        Optimizer &operator=(Optimizer &&) noexcept = delete;
        virtual void forward() = 0;

        void update(const ParameterPtrVector &params) {
            m_params.clear();
            m_grads.clear();
            m_params.reserve(params.size());
            m_grads.reserve(params.size());

            // Initialize gradients and parameters if not already initialized
            for (auto &param : params) {
                auto grad = param->get_grad();

                // Check if gradient exists
                if (!grad) {
                    throw std::invalid_argument(std::format("Array {} has no gradient for optimizer.", param->get_id().str()));
                }

                // Store detached gradient and parameters
                m_params.push_back(param->detach());
                m_grads.push_back(grad.value().detach());
            }

            forward();

            // Evaluate all parameters
            for (Array &param : m_params) {
                param.eval();
            }
        }
    };

    class GradientDescent : public Optimizer {
    public:
        explicit GradientDescent(float learning_rate = 1e-3) : Optimizer(learning_rate) {}

        void forward() override {
            for (size_t i = 0; i < m_params.size(); i++) {
                m_params[i] -= m_learning_rate * m_grads[i];
            }
        }
    };
} // namespace nx::optim