#pragma once

#include "../utils.h"

namespace nx::primitive {
    using namespace nx::utils;

    enum struct DeviceType {
        CPU,
        MPS
    };

    struct Device : public std::enable_shared_from_this<Device> {
    private:
        DeviceType m_type;
        isize m_id;
        std::string m_name;

        void init_name_from_type_and_id() {
            std::string type_str;

            switch (m_type) {
            case DeviceType::CPU:
                type_str = "cpu";
                break;
            default:
                type_str = "mps";
                break;
            }

            m_name = std::format("{}:{}", type_str, m_id);
        }

    public:
        Device(DeviceType type, isize id) : m_type(type), m_id(id) { init_name_from_type_and_id(); }
        Device(const Device &) = delete;
        Device(Device &&) noexcept = delete;
        ~Device() = default;
        Device &operator=(const Device &) = delete;
        Device &operator=(Device &&) noexcept = delete;
        DeviceType get_type() const { return m_type; }
        isize get_id() const { return m_id; }
        const std::string &get_name() const { return m_name; }
        bool operator==(const Device &device) const { return m_type == device.m_type && m_id == device.m_id; }
        const std::string str() const { return get_name(); }
        friend std::ostream &operator<<(std::ostream &os, const Device &device) { return os << device.str(); }
    };

    using DevicePtr = std::shared_ptr<Device>;
    const std::string default_device_name = "mps:0";
} // namespace nx::primitive

namespace std {
    template <>
    struct formatter<nx::primitive::Device> : formatter<string> {
        auto format(const nx::primitive::Device &device, format_context &ctx) const {
            return formatter<string>::format(device.str(), ctx);
        }
    };
} // namespace std