#include "array.h"
#include "optim.h"

NB_MODULE(arrayx, m) {
    auto m_core = m.def_submodule("core", "Core module");
    auto m_instrument = m.def_submodule("instrument", "Instrument module");
    auto m_nn = m.def_submodule("nn", "Neural network module");
    auto m_optim = m.def_submodule("optim", "Optimizer module");

    // Dtype class and operations
    nb::enum_<nxp::DtypeCategory>(m_core, "DtypeCategory")
        .value("BOOL", nxp::DtypeCategory::BOOL)
        .value("INT", nxp::DtypeCategory::INT)
        .value("FLOAT", nxp::DtypeCategory::FLOAT);

    nb::class_<nxp::Dtype>(m_core, "Dtype")
        .def_prop_ro("name", &nxp::Dtype::get_name_str, "Get data type's name as string")
        .def_prop_ro("size", &nxp::Dtype::get_size, "Get data type's size in bytes")
        .def_prop_ro("category", &nxp::Dtype::get_category, "Get data type's category")
        .def("__str__", &nxp::Dtype::str, "String representation of dtype");

    // Derived dtype classes
    nb::class_<nxp::F32, nxp::Dtype>(m_core, "F32", "32-bit floating point dtype");
    nb::class_<nxp::I32, nxp::Dtype>(m_core, "I32", "32-bit integer dtype");
    nb::class_<nxp::Bool, nxp::Dtype>(m_core, "Bool", "Boolean dtype");

    // Global dtype instances
    m_core.attr("f32") = &nxp::f32;
    m_core.attr("i32") = &nxp::i32;
    m_core.attr("b8") = &nxp::b8;

    // Shape class
    nb::class_<nxp::Shape>(m_core, "Shape")
        .def_prop_ro("offset", &nxp::Shape::get_offset, "Get shape's offset")
        .def_prop_ro("view", &nxp::Shape::get_view, "Get shape's view")
        .def_prop_ro("stride", &nxp::Shape::get_stride, "Get shape's stride")
        .def_prop_ro("ndim", &nxp::Shape::get_ndim, "Get shape's number of dimensions")
        .def_prop_ro("numel", &nxp::Shape::get_numel, "Get shape's total number of elements")
        .def("__str__", &nxp::Shape::str, "String representation of shape");

    // Device class
    nb::enum_<nxp::DeviceType>(m_core, "DeviceType")
        .value("CPU", nxp::DeviceType::CPU)
        .value("MPS", nxp::DeviceType::MPS);

    nb::class_<nxp::Device>(m_core, "Device")
        .def_prop_ro("type", &nxp::Device::get_type, "Get device's type")
        .def_prop_ro("id", &nxp::Device::get_id, "Get device's ID")
        .def_prop_ro("name", &nxp::Device::get_name, "Get device's name")
        .def("__str__", &nxp::Device::str, "String representation of device");

    nb::class_<nxi::Profiler>(m_instrument, "Profiler")
        .def(nb::init<>(), "Profiler")
        .def("record_alloc", &nxi::Profiler::record_alloc, "Record array's memory allocation")
        .def("record_free", &nxi::Profiler::record_free, "Record array's memory deallocation")
        .def("print_memory_profile", &nxi::Profiler::print_memory_profile, "Log memory profile to the console")
        .def("print_graph_profile", &nxi::Profiler::print_graph_profile, "Log computational graph profile to the console")
        .def("write_memory_profile", &nxi::Profiler::write_memory_profile, "Log memory profile to a file")
        .def("write_graph_profile", &nxi::Profiler::write_graph_profile, "Log computational graph profile to a file");

    // Array class
    nb::class_<nxc::Array>(m_core, "Array")
        // Properties
        .def_prop_ro("id", [](const nxc::Array &array) { return array.get_id().str(); }, "Get array's ID")
        .def_prop_ro("shape", &nxc::Array::get_shape, "Get array's shape")
        .def_prop_ro("dtype", &nxc::Array::get_dtype, "Get array's data type")
        .def_prop_ro("device", &nxc::Array::get_device, "Get array's device")
        .def_prop_ro("grad", &nxc::Array::get_grad, "Get array's gradient")
        .def_prop_ro("ndim", &nxc::Array::get_ndim, "Get array's number of dimensions")
        .def_prop_ro("numel", &nxc::Array::get_numel, "Get array's total number of elements")
        .def_prop_ro("offset", &nxc::Array::get_offset, "Get array's offset")
        .def_prop_ro("view", &nxc::Array::get_view, "Get array's view")
        .def_prop_ro("stride", &nxc::Array::get_stride, "Get array's stride")
        .def_prop_ro("ptr", &nxc::Array::get_ptr, "Get array's raw data pointer")
        .def_prop_ro("itemsize", &nxc::Array::get_itemsize, "Get array's element size in bytes")
        .def_prop_ro("nbytes", &nxc::Array::get_nbytes, "Get array's total size in bytes")
        .def_prop_ro("is_contiguous", &nxc::Array::is_contiguous, "Check if array is contiguous")
        .def_prop_rw("grad_enabled", &nxc::Array::is_grad_enabled, &nxc::Array::enable_grad, "enabled"_a, "Get/set array's gradient tracking, setter can only be used before compilation and forwarding, otherwise, there is no effect")

        // N-dimensional array
        .def("numpy", &nxb::array_to_numpy, nb::rv_policy::reference_internal, "Convert array to numpy array")
        .def_static("from_numpy", &nxb::array_from_numpy, "array"_a, "Convert numpy array to array")
        .def("torch", &nxb::array_to_torch, nb::rv_policy::reference_internal, "Convert array to Pytorch tensor")
        // .def_static("from_torch", &nxb::array_from_torch, "tensor"_a, "Convert Pytorch tensor to array")
        .def("item", &nxb::item, "Get array's only value")
        .def("graph", &nxc::Array::graph_str, "Get array's computation graph representation")

        // Initializer operations
        .def_static("full", &nxb::full, "view"_a, "c"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array filled with specified value")
        .def_static("full_like", &nxb::full_like, "array"_a, "c"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array filled with specified value with same shape as the input array")
        .def_static("zeros", &nxc::zeros, "view"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array filled with zeros")
        .def_static("ones", &nxc::ones, "view"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array filled with ones")
        .def_static("arange", &nxc::arange, "view"_a, "start"_a, "step"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array with evenly spaced values")
        .def_static("zeros_like", &nxc::zeros_like, "array"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array of zeros with same shape as input")
        .def_static("ones_like", &nxc::ones_like, "array"_a, "dtype"_a = &nxp::f32, "device"_a = nxp::default_device_name, "Create a new array of ones with same shape as input")

        // Element-wise operations
        .def("__add__", &nxb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__radd__", &nxb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__sub__", &nxb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__rsub__", &nxb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__mul__", &nxb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__rmul__", &nxb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__truediv__", &nxb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__rtruediv__", &nxb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__iadd__", &nxb::iadd, "rhs"_a, "In-place add two arrays element-wise")
        .def("__isub__", &nxb::isub, "rhs"_a, "In-place subtract two arrays element-wise")
        .def("__imul__", &nxb::imul, "rhs"_a, "In-place multiply two arrays element-wise")
        .def("__itruediv__", &nxb::idiv, "rhs"_a, "In-place divide two arrays element-wise")
        .def("__matmul__", &nxc::Array::matmul, "rhs"_a, "Matrix multiply two arrays")
        .def("detach", &nxc::Array::detach, "Detach array from computation graph")
        .def("exp", &nxc::Array::exp, "in_place"_a = false, "Compute exponential of array elements")
        .def("log", &nxc::Array::log, "in_place"_a = false, "Compute natural logarithm of array elements")
        .def("sqrt", &nxc::Array::sqrt, "in_place"_a = false, "Compute square root of array elements")
        .def("sq", &nxc::Array::sq, "in_place"_a = false, "Compute square of array elements")
        .def("neg", &nxc::Array::neg, "in_place"_a = false, "Compute negative of array elements")
        .def("__neg__", &nxb::neg, "Compute negative of array elements")
        .def("recip", &nxc::Array::recip, "in_place"_a = false, "Compute reciprocal of array elements")

        // Comparison operations
        .def("__eq__", &nxb::eq, "rhs"_a, "Element-wise equality comparison")
        .def("__ne__", &nxb::neq, "rhs"_a, "Element-wise inequality comparison")
        .def("__lt__", &nxb::lt, "rhs"_a, "Element-wise less than comparison")
        .def("__gt__", &nxb::gt, "rhs"_a, "Element-wise greater than comparison")
        .def("__le__", &nxb::leq, "rhs"_a, "Element-wise less than or equal comparison")
        .def("__ge__", &nxb::geq, "rhs"_a, "Element-wise greater than or equal comparison")
        .def("minimum", &nxb::minimum, "rhs"_a, "Element-wise minimum comparison")
        .def("maximum", &nxb::maximum, "rhs"_a, "Element-wise maximum comparison")

        // Reduction operations
        .def("sum", &nxb::sum, "dims"_a = nxp::ShapeDims{}, "Sum array elements along specified dimensions")
        .def("mean", &nxb::mean, "dims"_a = nxp::ShapeDims{}, "Mean value along specified dimensions")
        .def("max", &nxb::max, "dims"_a = nxp::ShapeDims{}, "Maximum value along specified dimensions")
        .def("min", &nxb::min, "dims"_a = nxp::ShapeDims{}, "Minimum value along specified dimensions")
        .def("argmax", &nxb::argmax, "dims"_a = nxp::ShapeDims{}, "Indices of maximum values along specified dimensions")
        .def("argmin", &nxb::argmin, "dims"_a = nxp::ShapeDims{}, "Indices of minimum values along specified dimensions")

        // Shape operations
        .def("broadcast", &nxc::Array::broadcast, "view"_a, "Broadcast array to new shape")
        .def("broadcast_to", &nxc::Array::broadcast_to, "view"_a, "Broadcast array to target shape")
        .def("__getitem__", &nxb::slice, "index"_a, "Slice array along specified dimensions")
        .def("reshape", &nxc::Array::reshape, "view"_a, "Reshape array to new dimensions")
        .def("flatten", &nxb::flatten, "start_dim"_a = 0, "end_dim"_a = -1, "Flatten dimensions from start to end")
        .def("squeeze", &nxb::squeeze, "dims"_a = nxp::ShapeDims{}, "Remove single-dimensional entry from array")
        .def("unsqueeze", &nxb::unsqueeze, "dims"_a = nxp::ShapeDims{}, "Add single-dimensional entry to array")
        .def("permute", &nxb::permute, "dims"_a, "Permute array dimensions")
        .def("transpose", &nxb::transpose, "start_dim"_a = -2, "end_dim"_a = -1, "Transpose array dimensions")

        // Type operations
        .def("astype", &nxc::Array::astype, "dtype"_a, "Cast array to specified dtype")

        // Evaluation and backward
        .def("eval", &nxc::Array::eval, "Evaluate array and materialize values")
        .def("backward", &nxc::Array::backward, "Compute gradients through backpropagation")

        // String representation
        .def("__str__", &nxc::Array::str, "String representation of array");

    nb::class_<nxo::Optimizer, nxb::PyOptimizer>(m_optim, "Optimizer")
        .def(nb::init<float>(), "lr"_a = 1e-3, "Base optimizer")
        .def("forward", &nxo::Optimizer::forward, "Parameter update function")
        .def("update", &nxo::Optimizer::update, "arrays"_a, "Update module parameters");

    nb::class_<nxo::GradientDescent, nxo::Optimizer>(m_optim, "GradientDescent")
        .def(nb::init<float>(), "lr"_a = 1e-3, "Gradient Descent optimizer");

    m_nn.def("linear", &nxn::linear, "x"_a, "weight"_a, "Functional linear without bias");
    m_nn.def("linear_with_bias", &nxn::linear_with_bias, "x"_a, "weight"_a, "bias"_a, "Functional linear with bias");
    m_nn.def("relu", &nxn::relu, "x"_a, "ReLU activation function");
    m_nn.def("onehot", &nxn::onehot, "x"_a, "num_classes"_a = -1, "One-hot encode input array");
    m_nn.def("cross_entropy_loss", &nxn::cross_entropy_loss, "x"_a, "y"_a, "Compute cross-entropy loss between input x and target y");
}
