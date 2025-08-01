#include "random/random.h"

void run_basic() {
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

int main() {
    auto profiler = std::make_shared<nx::instrument::Profiler>();
    nx::core::Backend::hook_profiler("mps:0", profiler);
    // run_basic(profiler, "graph.json");
    // profiler->write_memory_profile("memory.json");
    run_random();
    profiler->write_memory_profile("memory.json");
    return 0;
}