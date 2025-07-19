#pragma once

#include "bind.h"

namespace nx::bind {
    struct PyOptimizer : nxo::Optimizer {
        NB_TRAMPOLINE(nxo::Optimizer, 1);

        void forward() override {
            NB_OVERRIDE_PURE(forward);
        }
    };
} // namespace nx::bind