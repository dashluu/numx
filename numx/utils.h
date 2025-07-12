#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nx::utils {
    using isize = int64_t;

    template <class T>
    concept Numeric = std::is_arithmetic_v<T>;

    template <class T>
    concept NumericOrBool = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

    template <class T>
    inline size_t vsize(const std::vector<T> &v) {
        return v.size() * sizeof(T);
    }

    template <class T>
    inline const std::string join(const std::vector<T> &v, const std::function<std::string(T)> &f, const std::string &sep = ",") {
        std::string s = "";

        for (size_t i = 0; i < v.size(); i++) {
            if (i > 0) s += sep;
            s += f(v[i]);
        }

        return s;
    }

    template <Numeric T>
    inline const std::string join_nums(const std::vector<T> &v, const std::string &sep = ",") {
        return join<T>(v, [](T a) { return std::to_string(a); }, sep);
    }

    inline const std::string join(const std::vector<std::string> &v, const std::string &sep = ",") {
        return join<std::string>(v, [](std::string a) { return a; }, sep);
    }
} // namespace nx::utils