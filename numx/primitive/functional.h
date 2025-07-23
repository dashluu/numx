#pragma once

#include "op.h"

namespace nx::primitive {
    isize item(OpPtr op);
    OpPtr detach(OpPtr op);
    OpPtr from_buffer(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype, DevicePtr device);
    OpPtr empty(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr empty_like(OpPtr op, DtypePtr dtype, DevicePtr device);
    OpPtr empty_like(OpPtr op);
    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device);
    OpPtr zeros_like(OpPtr in_op);
    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device);
    OpPtr ones_like(OpPtr in_op);
    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device);
    OpPtr uniform(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr broadcast(OpPtr in_op, const ShapeView &view);
    OpPtr broadcast_to(OpPtr in_op, const ShapeView &view);
    OpPtr slice(OpPtr in_op, const RangeVec &ranges);
    OpPtr astype(OpPtr in_op, DtypePtr dtype);
    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims);
    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims);
    OpPtr add(OpPtr l_op, OpPtr r_op);
    OpPtr sub(OpPtr l_op, OpPtr r_op);
    OpPtr mul(OpPtr l_op, OpPtr r_op);
    OpPtr div(OpPtr l_op, OpPtr r_op);
    OpPtr matmul(OpPtr l_op, OpPtr r_op);
    OpPtr iadd(OpPtr l_op, OpPtr r_op);
    OpPtr isub(OpPtr l_op, OpPtr r_op);
    OpPtr imul(OpPtr l_op, OpPtr r_op);
    OpPtr idiv(OpPtr l_op, OpPtr r_op);
    OpPtr eq(OpPtr l_op, OpPtr r_op);
    OpPtr neq(OpPtr l_op, OpPtr r_op);
    OpPtr lt(OpPtr l_op, OpPtr r_op);
    OpPtr gt(OpPtr l_op, OpPtr r_op);
    OpPtr leq(OpPtr l_op, OpPtr r_op);
    OpPtr geq(OpPtr l_op, OpPtr r_op);
    OpPtr minimum(OpPtr l_op, OpPtr r_op);
    OpPtr maximum(OpPtr l_op, OpPtr r_op);
    OpPtr sq(OpPtr in_op, bool in_place = false);
    OpPtr sqrt(OpPtr in_op, bool in_place = false);
    OpPtr neg(OpPtr in_op, bool in_place = false);
    OpPtr copy(OpPtr in_op);
    OpPtr exp(OpPtr in_op, bool in_place = false);
    OpPtr log(OpPtr in_op, bool in_place = false);
    OpPtr recip(OpPtr in_op, bool in_place = false);
    OpPtr reshape(OpPtr in_op, const ShapeView &view);
    OpPtr permute(OpPtr in_op, const ShapeDims &dims);
    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr sum(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr mean(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr max(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr min(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr expand(OpPtr in_op, const ShapeView &reduce_operand_view, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims);

    template <NumericOrBool T>
    OpPtr full(const ShapeView &view, T constant, DtypePtr dtype, DevicePtr device) {
        return std::make_shared<FullOp>(ArrayData(Shape(view), dtype, device), dtype_bitcast_numeric(dtype, constant));
    }

    template <NumericOrBool T>
    OpPtr full_like(OpPtr in_op, T constant, DtypePtr dtype, DevicePtr device) {
        return full(in_op->get_data().get_view(), constant, dtype, device);
    }

    template <NumericOrBool T>
    OpPtr full_like(OpPtr in_op, T constant) {
        const ArrayData &in_data = in_op->get_data();
        return full(in_data.get_view(), constant, in_data.get_dtype(), in_data.get_device());
    }

    template <Numeric T>
    OpPtr uniform(const ShapeView &view, isize key, T low, T high, DtypePtr dtype, DevicePtr device) {
        return std::make_shared<UniformOp>(ArrayData(Shape(view), dtype, device), key, dtype_bitcast_numeric(dtype, low), dtype_bitcast_numeric(dtype, high));
    }

    template <Numeric T>
    OpPtr binary_with_scalar(OpPtr l_op, T constant, OpPtr (*op_fn)(OpPtr, OpPtr)) {
        const ArrayData &l_data = l_op->get_data();
        DtypePtr l_dtype = l_data.get_dtype();
        OpPtr r_op = full(l_data.get_view(), constant, l_dtype, l_data.get_device());
        r_op->enable_grad(false);
        return op_fn(l_op, r_op);
    }

    template <NumericOrBool T>
    OpPtr eq_with_scalar(OpPtr l_op, T constant, OpPtr (*op_fn)(OpPtr, OpPtr)) {
        const ArrayData &l_data = l_op->get_data();
        DtypePtr l_dtype = l_data.get_dtype();
        OpPtr r_op = full(l_data.get_view(), constant, l_dtype, l_data.get_device());
        r_op->enable_grad(false);
        return op_fn(l_op, r_op);
    }

    template <Numeric T>
    OpPtr add(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, add); }

    template <Numeric T>
    OpPtr iadd(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, iadd); }

    template <Numeric T>
    OpPtr sub(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, sub); }

    template <Numeric T>
    OpPtr isub(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, isub); }

    template <Numeric T>
    OpPtr mul(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, mul); }

    template <Numeric T>
    OpPtr imul(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, imul); }

    template <Numeric T>
    OpPtr div(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, div); }

    template <Numeric T>
    OpPtr idiv(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, idiv); }

    template <NumericOrBool T>
    OpPtr eq(OpPtr l_op, T constant) { return eq_with_scalar(l_op, constant, eq); }

    template <NumericOrBool T>
    OpPtr neq(OpPtr l_op, T constant) { return eq_with_scalar(l_op, constant, neq); }

    template <Numeric T>
    OpPtr lt(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, lt); }

    template <Numeric T>
    OpPtr gt(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, gt); }

    template <Numeric T>
    OpPtr leq(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, leq); }

    template <Numeric T>
    OpPtr geq(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, geq); }

    template <Numeric T>
    OpPtr minimum(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, minimum); }

    template <Numeric T>
    OpPtr maximum(OpPtr l_op, T constant) { return binary_with_scalar(l_op, constant, maximum); }

    template <class O>
    OpPtr elmwise_binary(OpPtr l_op, OpPtr r_op) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        DtypePtr l_dtype = l_data.get_dtype(), r_dtype = r_data.get_dtype();
        DevicePtr l_device = l_data.get_device(), r_device = r_data.get_device();

        if (!l_data.get_shape().broadcastable(r_view)) {
            throw IncompatShapesForOp(O::s_opname, join_nums(l_view), join_nums(r_view));
        }

        if (!l_dtype->is_numeric() || *l_dtype != *r_dtype) {
            throw IncompatDtypesForOp(O::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw IncompatDevicesForOp(O::s_opname, l_device->str(), r_device->str());
        }

        OpPtr broadcasted_l_op = broadcast(l_op, r_view);
        OpPtr broadcasted_r_op = broadcast(r_op, l_view);
        const ArrayData out_data(Shape(broadcasted_l_op->get_data().get_view()), l_dtype, l_device);
        return std::make_shared<O>(out_data, broadcasted_l_op, broadcasted_r_op, false);
    }

    template <class O>
    OpPtr in_place_binary(OpPtr l_op, OpPtr r_op) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const Shape &l_shape = l_data.get_shape();
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        DtypePtr l_dtype = l_data.get_dtype(), r_dtype = r_data.get_dtype();
        DevicePtr l_device = l_data.get_device(), r_device = r_data.get_device();

        if (!l_data.get_shape().broadcastable(r_view)) {
            throw IncompatShapesForOp(O::s_opname, join_nums(l_view), join_nums(r_view));
        }

        if (!l_dtype->is_numeric() || *l_dtype != *r_dtype) {
            throw IncompatDtypesForOp(O::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw IncompatDevicesForOp(O::s_opname, l_device->str(), r_device->str());
        }

        OpPtr broadcasted_r_op = broadcast_to(r_op, l_view);
        const ArrayData out_data(l_shape, l_dtype, l_device);
        return std::make_shared<O>(out_data, l_op, broadcasted_r_op, true);
    }

    template <class O>
    OpPtr unary(OpPtr in_op, bool in_place) {
        const ArrayData &in_data = in_op->get_data();
        DtypePtr in_dtype = in_data.get_dtype();

        if (!in_dtype->is_numeric()) {
            throw IncompatDtypeForOp(O::s_opname, in_dtype->str());
        }

        const ArrayData out_data(Shape(in_data.get_view()), in_dtype, in_data.get_device());
        return std::make_shared<O>(out_data, in_op, in_place);
    }

    template <class O>
    OpPtr unary_float(OpPtr in_op, bool in_place) {
        const ArrayData &in_data = in_op->get_data();
        DtypePtr in_dtype = in_data.get_dtype();

        if (in_place) {
            if (!in_dtype->is_float()) {
                // This method requires the operand to be of floating-point type
                // to do in-place operation since the result is of floating-point type
                throw IncompatDtypeForOp(O::s_opname, in_dtype->str());
            }
        }

        DtypePtr out_dtype = float_dtype_by_dtype(in_dtype);

        if (!out_dtype) {
            throw IncompatDtypeForOp(O::s_opname, in_dtype->str());
        }

        const ArrayData out_data(Shape(in_data.get_view()), out_dtype, in_data.get_device());
        return std::make_shared<O>(out_data, in_op, in_place);
    }

    template <class O>
    OpPtr cmp(OpPtr l_op, OpPtr r_op, DtypeCategory dtype_category) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        DtypePtr l_dtype = l_data.get_dtype(), r_dtype = r_data.get_dtype();
        DevicePtr l_device = l_data.get_device(), r_device = r_data.get_device();

        if (!l_data.get_shape().broadcastable(r_view)) {
            throw IncompatShapesForOp(O::s_opname, join_nums(l_view), join_nums(r_view));
        }

        if (!l_dtype->has_category(dtype_category) || *l_dtype != *r_dtype) {
            throw IncompatDtypesForOp(O::s_opname, l_dtype->str(), r_dtype->str());
        }

        if (l_device != r_device) {
            throw IncompatDevicesForOp(O::s_opname, l_device->str(), r_device->str());
        }

        OpPtr broadcasted_l_op = broadcast(l_op, r_view);
        OpPtr broadcasted_r_op = broadcast(r_op, l_view);
        const ArrayData out_data(Shape(broadcasted_l_op->get_data().get_view()), &b8, l_device);
        return std::make_shared<O>(out_data, broadcasted_l_op, broadcasted_r_op);
    }

    template <class O>
    OpPtr reduce(OpPtr in_op, const ShapeDims &dims, DtypePtr out_dtype, DtypeCategory dtype_category) {
        const ArrayData &in_data = in_op->get_data();
        const Shape &in_shape = in_data.get_shape();
        DtypePtr in_dtype = in_data.get_dtype();
        DevicePtr in_device = in_data.get_device();

        if (!in_dtype->has_category(dtype_category)) {
            throw IncompatDtypeForOp(O::s_opname, in_dtype->str());
        }

        OpPtr out_op;
        ShapeDims reduce_dims;
        ShapeDims remaining_dims(in_shape.get_ndim());
        std::iota(remaining_dims.begin(), remaining_dims.end(), 0);

        if (dims.size() == 0) {
            // Reduce to one element
            reduce_dims = remaining_dims;
            remaining_dims.clear();
            ArrayData out_data(Shape({1}), out_dtype, in_device);
            out_op = std::make_shared<O>(out_data, in_op, remaining_dims, reduce_dims);
            return out_op;
        }

        // Remove the dimensions to be reduced from remaining dims
        for (auto &dim : dims) {
            auto iter = std::find(remaining_dims.begin(), remaining_dims.end(), dim);

            if (iter == remaining_dims.end()) {
                throw std::invalid_argument(std::format("Invalid reduction dimension {} on array {}, either it does not exist or is duplicated.", dim, in_data.get_id()));
            } else {
                remaining_dims.erase(iter);
                reduce_dims.push_back(dim);
            }
        }

        ShapeView out_view(remaining_dims.size() + 1, 1);
        std::transform(remaining_dims.begin(), remaining_dims.end(), out_view.begin(), [&](isize dim) { return in_shape[dim]; });
        ArrayData out_data(Shape(out_view), out_dtype, in_device);
        out_op = std::make_shared<O>(out_data, in_op, remaining_dims, reduce_dims);
        return out_op;
    }
} // namespace nx::primitive