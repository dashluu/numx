#pragma once

#include "../core/array.h"

namespace nx::optim {
    using namespace nx::core;

    class Optimizer {
    protected:
        float lr;
        ArrayVec params;
        ArrayVec grads;

    public:
        Optimizer(float lr) : lr(lr) {}
        virtual ~Optimizer() = default;
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual void forward() = 0;

        void update(const ArrayVec &arrays) {
            params.clear();
            grads.clear();
            params.reserve(arrays.size());
            grads.reserve(arrays.size());

            // Initialize gradients and parameters if not already initialized
            for (const auto &array : arrays) {
                const auto &grad = array.get_grad();

                // Check if gradient exists
                if (!grad) {
                    throw std::invalid_argument(std::format("Array {} has no gradient for optimizer.", array.get_id().str()));
                }

                // Store detached gradient and parameters
                params.push_back(array.detach());
                grads.push_back(grad.value().detach());
            }

            forward();

            // Evaluate all parameters
            for (Array &param : params) {
                param.eval();
            }
        }
    };

    class GradientDescent : public Optimizer {
    public:
        GradientDescent(float lr = 1e-3) : Optimizer(lr) {}

        void forward() override {
            for (size_t i = 0; i < params.size(); i++) {
                params[i] -= lr * grads[i];
            }
        }
    };
} // namespace nx::optim