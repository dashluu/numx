#pragma once

#include "utils.h"

struct Exp {
    template <class T>
    float operator()(T x) const {
        return metal::exp(static_cast<float>(x));
    }
};

struct Log {
    template <class T>
    float operator()(T x) const {
        return metal::log(static_cast<float>(x));
    }
};

struct Neg {
    template <class T>
    T operator()(T x) const {
        return -x;
    }
};

struct Recip {
    template <class T>
    float operator()(T x) const {
        return 1.0f / x;
    }
};

struct Sin {
    template <class T>
    float operator()(T x) const {
        return metal::sin(static_cast<float>(x));
    }
};

struct Cos {
    template <class T>
    float operator()(T x) const {
        return metal::cos(static_cast<float>(x));
    }
};

struct Sqrt {
    template <class T>
    float operator()(T x) const {
        return metal::sqrt(static_cast<float>(x));
    }
};

struct Sq {
    template <class T>
    float operator()(T x) const {
        return x * x;
    }
};