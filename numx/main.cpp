#include "mnist.h"
#include "optim/optim.h"
#include "profiler/profiler.h"

void run_basic() {
    auto x1 = nx::core::full({2, 3, 4}, 3);
    auto x2 = nx::core::ones({1, 3, 1});
    auto x3 = x1 + x2;
    auto x4 = x1 - x2;
    auto x5 = x3 + x4;
    std::println("{}", x5);
}

void run_advanced() {
    auto x1 = nx::core::full({2, 3, 4}, 3);
    auto x2 = nx::core::ones({1, 3, 1});
    x1 -= x2;
    auto x3 = x1.exp();
    auto x4 = x2.exp();
    auto x5 = x3 * x4;
    auto x6 = x5.sum();
    std::println("{}", x1);
    std::println("{}", x2);
    std::println("{}", x6);
}

void run_random() {
    auto x1 = nx::random::normal<float>({2, 3, 4});
    auto x2 = x1.cos();
    std::println("{}", x1);
    std::println("{}", x2);
}

void run_linear() {
    auto x1 = nx::random::normal<float>({3, 10});
    nx::nn::Linear model(10, 4);
    auto x2 = model(x1);
    std::println("{}", x2);
}

void run_optimizer() {
    auto x1 = nx::core::full({2, 3, 4}, 5);
    auto x2 = nx::core::full({1, 3, 4}, 0.2);
    auto x3 = x1 + x2;
    auto x4 = x3.sum();
    x4.backward();
    auto x1_grad = x1.get_grad().value();
    auto x2_grad = x2.get_grad().value();
    std::println("{}\n", x1);
    std::println("{}\n", x2);
    std::println("{}\n", x4);
    std::println("{}\n", x1_grad);
    std::println("{}\n", x2_grad);
    nx::optim::GradientDescent optimizer(1);
    optimizer.update({&x1, &x2});
    std::println("{}\n", x1);
    std::println("{}\n", x2);
}

void run_multipass_with_optimizer() {
    MnistModel model;
    nx::optim::GradientDescent optimizer(1);

    for (size_t i = 0; i < 3; i++) {
        auto input = nx::random::normal<float>({64, 784});
        auto label = nx::core::full({64}, 9, &nx::core::i32);
        auto logits = model(input);
        auto loss = nx::nn::cross_entropy_loss(logits, label);
        loss.backward();
        optimizer.update(model.get_parameters());
        std::println("{}", loss);
    }
}

int main() {
    nx::profiler::enable_memory_profile();
    run_multipass_with_optimizer();
    nx::profiler::save_memory_profile("memory_profile.json");
    return 0;
}