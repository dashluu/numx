#pragma once

#include "bind.h"

namespace nx::bind {
    struct PyOptimizer : no::Optimizer {
        NB_TRAMPOLINE(no::Optimizer, 1);

        void forward() override {
            NB_OVERRIDE_PURE(forward);
        }
    };
} // namespace nx::bind