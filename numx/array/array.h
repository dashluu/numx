#pragma once

#include "backend.h"

namespace nx::array {
    struct Array {
    private:
        OpPtr m_op = nullptr;
        GraphPtr m_graph = nullptr;
        RunnerPtr m_runner = nullptr;

        static Backend &get_backend() { return Backend::get_instance(); }
        static DevicePtr get_device_by_name(const std::string &name) { return get_backend().get_device_by_name(name); }
        RuntimeContextPtr get_context() const { return get_backend().get_context_by_device_name(m_op->get_data().get_device_name()); }
        RunnerFactory get_runner_factory() const { return get_backend().get_runner_factory_by_device_name(m_op->get_data().get_device_name()); }
        GraphFactory get_graph_factory() { return get_backend().get_graph_factory_by_device_name(m_op->get_data().get_device_name()); }

    public:
        Array() = default;
        Array(OpPtr op) : m_op(op) {}
        Array(const Array &array) : m_op(array.m_op), m_graph(array.m_graph), m_runner(array.m_runner) {}
        ~Array();

        Array &operator=(const Array &array) {
            m_op = array.m_op;
            m_graph = array.m_graph;
            m_runner = array.m_runner;
            return *this;
        }

        OpPtr get_op() const { return m_op; }
        GraphPtr get_graph() const { return m_graph; }
        const ArrayData &get_data() const { return m_op->get_data(); }
        const ArrayId &get_id() const { return get_data().get_id(); }
        const Shape &get_shape() const { return get_data().get_shape(); }
        isize get_offset() const { return get_data().get_offset(); }
        const ShapeView &get_view() const { return get_data().get_view(); }
        const ShapeStride &get_stride() const { return get_data().get_stride(); }
        uint8_t *get_ptr() const { return m_op->get_data().get_ptr(); }
        DtypePtr get_dtype() const { return get_data().get_dtype(); }
        DevicePtr get_device() const { return get_data().get_device(); }

        std::optional<const Array> get_grad() const {
            OpPtr grad = m_op->get_grad();
            return grad ? std::optional<const Array>(Array(nx::graph::detach(grad))) : std::nullopt;
        }

        isize get_numel() const { return get_data().get_numel(); }
        isize get_ndim() const { return get_data().get_ndim(); }
        isize get_itemsize() const { return get_data().get_itemsize(); }
        isize get_nbytes() const { return get_data().get_nbytes(); }
        bool is_grad_enabled() const { return m_op->is_grad_enabled(); }
        // This can only be used before compilation or forwarding, otherwise, there is no effect
        void enable_grad(bool enabled) { m_op->enable_grad(enabled); }
        bool is_contiguous() const { return get_data().is_contiguous(); }

        isize item() {
            eval();
            return nx::graph::item(m_op);
        }

        const std::string graph_str() {
            eval();
            return m_graph->str();
        }

        const std::string str() {
            eval();
            return m_op->get_data().str();
        }

        void eval();
        void backward();
        friend std::ostream &operator<<(std::ostream &os, Array &array) { return os << array.str(); }
        Array detach() const { return Array(nx::graph::detach(m_op)); }

        static Array from_buffer(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::from_buffer(ptr, size, shape, dtype, device));
        }

        // Initializer operations
        template <class T>
        static Array full(const ShapeView &view, T constant, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::full(view, constant, dtype, device));
        }

        template <class T>
        static Array full_like(const Array &other, T constant, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::full_like(other.m_op, constant, dtype, device));
        }

        static Array arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::arange(view, start, step, dtype, device));
        }

        static Array zeros(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::zeros(view, dtype, device));
        }

        static Array ones(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::ones(view, dtype, device));
        }

        static Array zeros_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::zeros_like(other.m_op, dtype, device));
        }

        static Array ones_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::ones_like(other.m_op, dtype, device));
        }

        static Array empty(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::empty(view, dtype, device));
        }

        static Array empty_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_device_by_name(device_name);
            return Array(nx::graph::empty_like(other.m_op, dtype, device));
        }

        static Array empty_like(const Array &other) { return Array(nx::graph::empty_like(other.m_op)); }

        // Element-wise operations
        Array operator+(const Array &rhs) const { return Array(nx::graph::add(m_op, rhs.m_op)); }

        template <Numeric T>
        Array operator+(T constant) const { return Array(nx::graph::add(m_op, constant)); }

        Array operator-(const Array &rhs) const { return Array(nx::graph::sub(m_op, rhs.m_op)); }

        template <Numeric T>
        Array operator-(T constant) const { return Array(nx::graph::sub(m_op, constant)); }

        Array operator*(const Array &rhs) const { return Array(nx::graph::mul(m_op, rhs.m_op)); }

        template <Numeric T>
        Array operator*(T constant) const { return Array(nx::graph::mul(m_op, constant)); }

        Array operator/(const Array &rhs) const { return Array(nx::graph::div(m_op, rhs.m_op)); }

        template <Numeric T>
        Array operator/(T constant) const { return Array(nx::graph::div(m_op, constant)); }

        Array &operator+=(const Array &rhs) {
            m_op = nx::graph::iadd(m_op, rhs.m_op);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        template <Numeric T>
        Array &operator+=(T constant) {
            m_op = nx::graph::iadd(m_op, constant);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        Array &operator-=(const Array &rhs) {
            m_op = nx::graph::isub(m_op, rhs.m_op);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        template <Numeric T>
        Array &operator-=(T constant) {
            m_op = nx::graph::isub(m_op, constant);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        Array &operator*=(const Array &rhs) {
            m_op = nx::graph::imul(m_op, rhs.m_op);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        template <Numeric T>
        Array &operator*=(T constant) {
            m_op = nx::graph::imul(m_op, constant);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        Array &operator/=(const Array &rhs) {
            m_op = nx::graph::idiv(m_op, rhs.m_op);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        template <Numeric T>
        Array &operator/=(T constant) {
            m_op = nx::graph::idiv(m_op, constant);
            m_graph = nullptr;
            m_runner = nullptr;
            return *this;
        }

        Array matmul(const Array &rhs) const { return Array(nx::graph::matmul(m_op, rhs.m_op)); }
        Array exp(bool in_place = false) const { return Array(nx::graph::exp(m_op, in_place)); }
        Array log(bool in_place = false) const { return Array(nx::graph::log(m_op, in_place)); }
        Array sqrt(bool in_place = false) const { return Array(nx::graph::sqrt(m_op, in_place)); }
        Array sq(bool in_place = false) const { return Array(nx::graph::sq(m_op, in_place)); }
        Array neg(bool in_place = false) const { return Array(nx::graph::neg(m_op, in_place)); }
        Array operator-() const { return Array(nx::graph::neg(m_op)); }
        Array recip(bool in_place = false) const { return Array(nx::graph::recip(m_op, in_place)); }
        Array operator==(const Array &rhs) const { return Array(nx::graph::eq(m_op, rhs.m_op)); }
        Array operator!=(const Array &rhs) const { return Array(nx::graph::neq(m_op, rhs.m_op)); }
        Array operator<(const Array &rhs) const { return Array(nx::graph::lt(m_op, rhs.m_op)); }
        Array operator>(const Array &rhs) const { return Array(nx::graph::gt(m_op, rhs.m_op)); }
        Array operator<=(const Array &rhs) const { return Array(nx::graph::leq(m_op, rhs.m_op)); }
        Array operator>=(const Array &rhs) const { return Array(nx::graph::geq(m_op, rhs.m_op)); }
        Array minimum(const Array &rhs) const { return Array(nx::graph::minimum(m_op, rhs.m_op)); }
        Array maximum(const Array &rhs) const { return Array(nx::graph::maximum(m_op, rhs.m_op)); }

        template <Numeric T>
        Array operator==(T constant) const { return Array(nx::graph::eq(m_op, constant)); }

        template <Numeric T>
        Array operator!=(T constant) const { return Array(nx::graph::neq(m_op, constant)); }

        template <Numeric T>
        Array operator<(T constant) const { return Array(nx::graph::lt(m_op, constant)); }

        template <Numeric T>
        Array operator>(T constant) const { return Array(nx::graph::gt(m_op, constant)); }

        template <Numeric T>
        Array operator<=(T constant) const { return Array(nx::graph::leq(m_op, constant)); }

        template <Numeric T>
        Array operator>=(T constant) const { return Array(nx::graph::geq(m_op, constant)); }

        template <Numeric T>
        Array minimum(T constant) const { return Array(nx::graph::minimum(m_op, constant)); }

        template <Numeric T>
        Array maximum(T constant) const { return Array(nx::graph::maximum(m_op, constant)); }

        // Reduction operations
        Array sum(const ShapeDims &dims = {}) const { return Array(nx::graph::sum(m_op, dims)); }
        Array mean(const ShapeDims &dims = {}) const { return Array(nx::graph::mean(m_op, dims)); }
        Array max(const ShapeDims &dims = {}) const { return Array(nx::graph::max(m_op, dims)); }
        Array min(const ShapeDims &dims = {}) const { return Array(nx::graph::min(m_op, dims)); }
        Array argmax(const ShapeDims &dims = {}) const { return Array(nx::graph::argmax(m_op, dims)); }
        Array argmin(const ShapeDims &dims = {}) const { return Array(nx::graph::argmin(m_op, dims)); }

        // Shape operations
        Array broadcast(const ShapeView &view) const { return Array(nx::graph::broadcast(m_op, view)); }
        Array broadcast_to(const ShapeView &view) const { return Array(nx::graph::broadcast_to(m_op, view)); }
        Array slice(const RangeVec &ranges) const { return Array(nx::graph::slice(m_op, ranges)); }
        Array reshape(const ShapeView &view) const { return Array(nx::graph::reshape(m_op, view)); }
        Array flatten(isize start_dim, isize end_dim) const { return Array(nx::graph::flatten(m_op, start_dim, end_dim)); }
        Array squeeze(const ShapeDims &dims = {}) const { return Array(nx::graph::squeeze(m_op, dims)); }
        Array unsqueeze(const ShapeDims &dims = {}) const { return Array(nx::graph::unsqueeze(m_op, dims)); }
        Array permute(const ShapeDims &dims) const { return Array(nx::graph::permute(m_op, dims)); }
        Array transpose(isize start_dim, isize end_dim) const { return Array(nx::graph::transpose(m_op, start_dim, end_dim)); }

        // Type operations
        Array astype(DtypePtr dtype) const { return Array(nx::graph::astype(m_op, dtype)); }
    };

    template <Numeric T>
    Array operator+(T constant, const Array &array) {
        return array + constant;
    }

    template <Numeric T>
    Array operator-(T constant, const Array &array) {
        return array - constant;
    }

    template <Numeric T>
    Array operator*(T constant, const Array &array) {
        return array * constant;
    }

    template <Numeric T>
    Array operator/(T constant, const Array &array) {
        return array.recip() * constant;
    }

    using ArrayVec = std::vector<Array>;
} // namespace nx::array

namespace std {
    template <>
    struct formatter<nx::array::Array> : formatter<string> {
        auto format(nx::array::Array &array, format_context &ctx) const {
            return formatter<string>::format(array.str(), ctx);
        }
    };
} // namespace std