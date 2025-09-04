#pragma once

#include "bind.h"

namespace nx::bind {
    inline nxc::Array uniform(const nxp::ShapeView &view, const nb::object &low, const nb::object &high, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name) {
        return nxr::uniform(view, nb::cast<float>(low), nb::cast<float>(high), dtype, device_name);
    }

    inline nxc::Array normal(const nxp::ShapeView &view, const nb::object &mean, const nb::object &std, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name) {
        return nxr::normal(view, nb::cast<float>(mean), nb::cast<float>(std), dtype, device_name);
    }

    inline nxc::Array randint(const nxp::ShapeView &view, const nb::object &low, const nb::object &high, nxp::DtypePtr dtype, const std::string &device_name = nxp::default_device_name) {
        return nxr::randint(view, nb::cast<int>(low), nb::cast<int>(high), dtype, device_name);
    }

    inline nxc::Array randbool(const nxp::ShapeView &view, const std::string &device_name = nxp::default_device_name) {
        return nxr::randbool(view, device_name);
    }
} // namespace nx::bind