#pragma once

#include "../nn/functional.h"
#include "../optim/optim.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace nx::bind {
}

namespace nb = nanobind;
namespace nxc = nx::core;
namespace nxr = nx::runtime;
namespace nxp = nx::profiler;
namespace nxa = nx::array;
namespace nxn = nx::nn;
namespace nxo = nx::optim;
namespace nxb = nx::bind;
using namespace nb::literals;
