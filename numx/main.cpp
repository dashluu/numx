#include "array/array.h"

using namespace nx::array;

int main() {
    auto x1 = Array::full({2, 3, 4}, 3);
    // auto x2 = Array::ones({1, 3, 1});
    // x1 -= x2;
    x1.eval();
    // std::cout << x1.str() << std::endl;
    return 0;
}