#pragma once

#include "bind.h"

namespace nx::bind {
    nxc::Array uniform(const nxp::ShapeView &view, const nb::object &low, const nb::object &high, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
    nxc::Array normal(const nxp::ShapeView &view, const nb::object &mean, const nb::object &std, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name);
} // namespace nx::bind