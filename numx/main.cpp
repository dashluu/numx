#include "array/array.h"

using namespace nx::array;

int main() {
    auto x1 = Array::full({2, 3, 4}, 3);
    auto x2 = Array::ones({1, 3, 1});
    x1 -= x2;
    std::println("{}", x2.str());
    std::println("{}", x1.str());
    auto x3 = x1.exp();
    auto x4 = x2.exp();
    auto x5 = x3 * x4;
    std::println("{}", x5.str());
    return 0;
}