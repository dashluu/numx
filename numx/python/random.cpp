#include "random.h"

namespace nx::bind {
    nxc::Array uniform(const nxp::ShapeView &view, const nb::object &low, const nb::object &high, nxp::DtypePtr dtype, const std::string &device_name) {
        return nxr::uniform(view, nb::cast<float>(low), nb::cast<float>(high), dtype, device_name);
    }

    nxc::Array normal(const nxp::ShapeView &view, const nb::object &mean, const nb::object &std, nxp::DtypePtr dtype, const std::string &device_name) {
        return nxr::normal(view, nb::cast<float>(mean), nb::cast<float>(std), dtype, device_name);
    }
} // namespace nx::bind