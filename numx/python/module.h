#pragma once

#include "bind.h"

namespace nx::bind {
    class ModulePublicist : public nxn::Module {
    public:
        using nxn::Module::add_parameter;
    };

    struct PyModule : nxn::Module {
        NB_TRAMPOLINE(nxn::Module, 1);

        nxc::Array forward(const nxc::Array &x) override {
            NB_OVERRIDE_PURE(forward, x);
        }
    };
} // namespace nx::bind