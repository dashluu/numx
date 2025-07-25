#pragma once

#include "../utils.h"
#include <stdexcept>

namespace nx::primitive {
    using namespace nx::utils;

    class IncompatShapesForOp : public std::invalid_argument {
    public:
        IncompatShapesForOp(const std::string &op, const std::string l_view, const std::string r_view) : std::invalid_argument(std::format("Cannot run operator {} on incompatible shapes {} and {}.", op, l_view, r_view)) {}
    };

    class IncompatDtypesForOp : public std::invalid_argument {
    public:
        IncompatDtypesForOp(const std::string &op, const std::string l_dtype, const std::string r_dtype) : std::invalid_argument(std::format("Cannot run operator {} on incompatible data types {} and {}.", op, l_dtype, r_dtype)) {}
    };

    class IncompatDtypeForOp : public std::invalid_argument {
    public:
        IncompatDtypeForOp(const std::string &op, const std::string dtype) : std::invalid_argument(std::format("Cannot run operator {} on incompatible data type {}.", op, dtype)) {}
    };

    class IncompatDevicesForOp : public std::invalid_argument {
    public:
        IncompatDevicesForOp(const std::string &op, const std::string l_device, const std::string r_device) : std::invalid_argument(std::format("Cannot run operator {} on incompatible devices {} and {}.", op, l_device, r_device)) {}
    };

    class OutOfRange : public std::out_of_range {
    public:
        OutOfRange(isize index, isize start, isize stop) : std::out_of_range(std::format("Index {} is not in the range [{}, {}).", index, start, stop)) {}
    };

    class NanobindInvalidArgumentType : public std::invalid_argument {
    public:
        NanobindInvalidArgumentType(const std::string input_argtype, const std::string &expected_argtype) : std::invalid_argument(std::format("Expected an argument of type {} but received an argument of type {}.", expected_argtype, input_argtype)) {}
    };
} // namespace nx::primitive