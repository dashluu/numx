#include "array/array.h"

using namespace nx::array;

void run_test(ProfilerPtr profiler, const std::string &file_name) {
    std::ofstream stream(file_name);

    if (!stream.is_open()) {
        throw std::runtime_error(std::format("Cannot log graph due to failing to open file '{}'...", file_name));
    }

    auto x1 = Array::full({2, 3, 4}, 3);
    auto x2 = Array::ones({1, 3, 1});
    x1 -= x2;
    auto x3 = x1.exp();
    auto x4 = x2.exp();
    auto x5 = x3 * x4;
    auto x6 = x5.sum();
    x6.backward();
    std::println("{}", x1);
    std::println("{}", x2);
    std::println("{}", x5);
    GraphPtr graph = x6.get_graph();
    profiler->log_graph(graph, stream);
    stream.close();
}

int main() {
    auto profiler = std::make_shared<Profiler>();
    Backend::use_profiler(profiler);
    run_test(profiler, "graph.json");
    profiler->log_memory(std::cout);
    return 0;
}