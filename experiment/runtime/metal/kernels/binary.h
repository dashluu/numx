#pragma once

#include "utils.h"

struct Add {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct Sub {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs - rhs; }
};

struct Mul {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs * rhs; }
};

struct Div {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs / rhs; }
};

struct Eq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs == rhs; }
};

struct Neq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs != rhs; }
};

struct Lt {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs < rhs; }
};

struct Gt {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs > rhs; }
};

struct Leq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs <= rhs; }
};

struct Geq {
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs >= rhs; }
};

struct Minimum {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs < rhs ? lhs : rhs; }
};

struct Maximum {
    template <class T>
    T operator()(T lhs, T rhs) { return lhs > rhs ? lhs : rhs; }
};