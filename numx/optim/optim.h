#pragma once

#include "../core/functional.h"

namespace nx::optim {
    using namespace nx::core;

    class Optimizer {
    protected:
        float m_learning_rate;
        ArrayVector m_parameters;
        ArrayVector m_grads;

    public:
        Optimizer(float learning_rate) : m_learning_rate(learning_rate) {}
        Optimizer(const Optimizer &) = delete;
        virtual ~Optimizer() = default;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual void forward() = 0;

        void update(const ParameterVector &arrays) {
            m_parameters.clear();
            m_grads.clear();
            m_parameters.reserve(arrays.size());
            m_grads.reserve(arrays.size());

            // Initialize gradients and parameters if not already initialized
            for (auto &array : arrays) {
                auto grad = array->get_grad();

                // Check if gradient exists
                if (!grad) {
                    throw std::invalid_argument(std::format("Array {} has no gradient for optimizer.", array->get_id().str()));
                }

                // Store detached gradient and parameters
                m_parameters.push_back(array->detach());
                m_grads.push_back(grad.value().detach());
            }

            forward();

            // Evaluate all parameters
            for (Array &parameter : m_parameters) {
                parameter.eval();
            }
        }
    };

    class GradientDescent : public Optimizer {
    public:
        GradientDescent(float learning_rate = 1e-3) : Optimizer(learning_rate) {}

        void forward() override {
            for (size_t i = 0; i < m_parameters.size(); i++) {
                m_parameters[i] -= m_learning_rate * m_grads[i];
            }
        }
    };
} // namespace nx::optim