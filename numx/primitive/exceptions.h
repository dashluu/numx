#pragma once

#include "../utils.h"
#include <stdexcept>

namespace nx::primitive {
    using namespace nx::utils;

    class IncompatShapesForOp : public std::invalid_argument {
    public:
        IncompatShapesForOp(std::string_view opname, std::string_view l_view_str, std::string_view r_view_str) : std::invalid_argument(std::format("Cannot run operator {} on incompatible shapes {} and {}.", opname, l_view_str, r_view_str)) {}
    };

    class IncompatDtypesForOp : public std::invalid_argument {
    public:
        IncompatDtypesForOp(std::string_view opname, std::string_view l_dtype_str, std::string_view r_dtype_str) : std::invalid_argument(std::format("Cannot run operator {} on incompatible data types {} and {}.", opname, l_dtype_str, r_dtype_str)) {}
    };

    class IncompatDtypeForOp : public std::invalid_argument {
    public:
        IncompatDtypeForOp(std::string_view opname, std::string_view dtype_str) : std::invalid_argument(std::format("Cannot run operator {} on incompatible data type {}.", opname, dtype_str)) {}
    };

    class IncompatDevicesForOp : public std::invalid_argument {
    public:
        IncompatDevicesForOp(std::string_view opname, std::string_view l_device_str, std::string_view r_device_str) : std::invalid_argument(std::format("Cannot run operator {} on incompatible devices {} and {}.", opname, l_device_str, r_device_str)) {}
    };

    class IndexOutOfRange : public std::out_of_range {
    public:
        IndexOutOfRange(isize index, isize start, isize stop) : std::out_of_range(std::format("Index {} is out of range [{}, {}).", index, start, stop)) {}
    };

    class UnableToOpenFileToSaveMemoryProfile : public std::runtime_error {
    public:
        explicit UnableToOpenFileToSaveMemoryProfile(std::string_view file_name) : std::runtime_error(std::format("Cannot save memory profile due to failing to open file '{}'.", file_name)) {}
    };

    class UnableToOpenFileToSaveGraphProfile : public std::runtime_error {
    public:
        explicit UnableToOpenFileToSaveGraphProfile(std::string_view file_name) : std::runtime_error(std::format("Cannot save graph profile due to failing to open file '{}'.", file_name)) {}
    };

    class InvalidMemoryProfileStream : public std::invalid_argument {
    public:
        InvalidMemoryProfileStream() : std::invalid_argument("Cannot stream memory profile due to invalid stream.") {}
    };

    class InvalidGraphProfileStream : public std::invalid_argument {
    public:
        InvalidGraphProfileStream() : std::invalid_argument("Cannot stream graph profile due to invalid stream.") {}
    };

    class NanobindInvalidArgumentType : public std::invalid_argument {
    public:
        NanobindInvalidArgumentType(std::string_view input_type_name, std::string_view expected_type_name) : std::invalid_argument(std::format("Expected an argument of type {} but received an argument of type {}.", expected_type_name, input_type_name)) {}
    };
} // namespace nx::primitive