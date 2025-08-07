#pragma once

#include "../nn/linear.h"
#include "../optim/optim.h"
#include "../profiler/profiler.h"
#include "../random/random.h"
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
namespace nxp = nx::primitive;
namespace nxr = nx::random;
namespace nxf = nx::profiler;
namespace nxc = nx::core;
namespace nxn = nx::nn;
namespace nxo = nx::optim;
namespace nxb = nx::bind;
using namespace nb::literals;
