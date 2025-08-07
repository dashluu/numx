#pragma once

#include "bind.h"

namespace nx::bind {
    struct PyModule : nxn::Module {
        NB_TRAMPOLINE(nxn::Module, 1);

        nxc::Array forward(const nxc::Array &x) override {
            NB_OVERRIDE_PURE(forward, x);
        }
    };
} // namespace nx::bind