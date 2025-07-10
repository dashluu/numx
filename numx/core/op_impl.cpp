#include "op_impl.h"

namespace nx::core {
    OpPtr empty(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return std::make_shared<EmptyOp>(ArrayDescriptor(Shape(view), dtype, device)); }
    OpPtr empty_like(OpPtr op, DtypePtr dtype, DevicePtr device) { return empty(op->get_descriptor().get_view(), dtype, device); }

    OpPtr empty_like(OpPtr op) {
        const ArrayDescriptor &descriptor = op->get_descriptor();
        return empty(descriptor.get_view(), descriptor.get_dtype(), descriptor.get_device());
    }

    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 0, dtype, device); }
    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 0, dtype, device); }
    OpPtr zeros_like(OpPtr in_op) { return zeros_like(in_op, in_op->get_descriptor().get_dtype(), in_op->get_descriptor().get_device()); }
    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 1, dtype, device); }
    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 1, dtype, device); }
    OpPtr ones_like(OpPtr in_op) { return ones_like(in_op, in_op->get_descriptor().get_dtype(), in_op->get_descriptor().get_device()); }

    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device) {
        return std::make_shared<ArangeOp>(ArrayDescriptor(Shape(view), dtype, device), view, start, step);
    }

    OpPtr broadcast(OpPtr op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = op->get_descriptor();
        const Shape &in_shape = in_descriptor.get_shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast(view);

        if (in_shape == broadcast_shape) {
            return op;
        }

        const ArrayDescriptor out_descriptor(broadcast_shape, in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<BroadcastOp>(out_descriptor, op, in_view, view, broadcast_dims);
    }

    OpPtr broadcast_to(OpPtr op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = op->get_descriptor();
        const Shape &in_shape = in_descriptor.get_shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast_to(view);

        if (in_shape == broadcast_shape) {
            return op;
        }

        const ArrayDescriptor out_descriptor(broadcast_shape, in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<BroadcastOp>(out_descriptor, op, in_view, view, broadcast_dims);
    }

    OpPtr slice(OpPtr in_op, const RangeVec &ranges) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ArrayDescriptor out_descriptor(in_descriptor.get_shape().slice(ranges), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<SliceOp>(out_descriptor, in_op, ranges);
    }

    OpPtr astype(OpPtr in_op, DtypePtr dtype) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();

        if (in_descriptor.get_dtype() == dtype) {
            return in_op;
        }

        const ArrayDescriptor out_descriptor(in_descriptor.get_shape(), dtype, in_descriptor.get_device());
        return std::make_shared<AstypeOp>(out_descriptor, in_op, dtype);
    }

    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ArrayDescriptor out_descriptor(in_descriptor.get_shape().unsqueeze(dims), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<UnsqueezeOp>(out_descriptor, in_op, dims);
    }

    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ArrayDescriptor out_descriptor(in_descriptor.get_shape().squeeze(dims), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<SqueezeOp>(out_descriptor, in_op, dims);
    }

    OpPtr add(OpPtr l_op, OpPtr r_op) { return elmwise_binary<AddOp>(l_op, r_op); }
    OpPtr sub(OpPtr l_op, OpPtr r_op) { return elmwise_binary<SubOp>(l_op, r_op); }
    OpPtr mul(OpPtr l_op, OpPtr r_op) { return elmwise_binary<MulOp>(l_op, r_op); }
    OpPtr div(OpPtr l_op, OpPtr r_op) { return elmwise_binary<DivOp>(l_op, r_op); }

    OpPtr matmul(OpPtr l_op, OpPtr r_op) {
        const ArrayDescriptor &l_descriptor = l_op->get_descriptor(), r_descriptor = r_op->get_descriptor();
        const ShapeView &l_view = l_descriptor.get_view();
        const ShapeView &r_view = r_descriptor.get_view();
        DtypePtr l_dtype = l_descriptor.get_dtype(), r_dtype = r_descriptor.get_dtype();
        DevicePtr l_device = l_descriptor.get_device(), r_device = r_descriptor.get_device();

        if (!l_descriptor.get_shape().matmul_broadcastable(r_view)) {
            throw IncompatShapesForOp(MatmulOp::s_opname, join_nums(l_view), join_nums(r_view));
        }
        if (!l_dtype->is_numeric() || *l_dtype != *r_dtype) {
            throw IncompatDtypesForOp(MatmulOp::s_opname, l_dtype->str(), r_dtype->str());
        }
        if (l_device != r_device) {
            throw IncompatDevicesForOp(MatmulOp::s_opname, l_device->str(), r_device->str());
        }

        ShapeView broadcasted_l_view = l_view;
        ShapeView broadcasted_r_view = r_view;
        size_t ndim = std::max(broadcasted_l_view.size(), broadcasted_r_view.size());
        broadcasted_l_view.insert(broadcasted_l_view.begin(), ndim - broadcasted_l_view.size(), 1);
        broadcasted_r_view.insert(broadcasted_r_view.begin(), ndim - broadcasted_r_view.size(), 1);

        for (size_t i = 0; i < ndim - 2; i++) {
            isize shared_dim = std::max(broadcasted_l_view[i], broadcasted_r_view[i]);
            broadcasted_l_view[i] = shared_dim;
            broadcasted_r_view[i] = shared_dim;
        }

        OpPtr broadcasted_l_op = broadcast(l_op, broadcasted_l_view);
        OpPtr broadcasted_r_op = broadcast(r_op, broadcasted_r_view);
        ShapeView out_view = broadcasted_l_view;
        out_view[out_view.size() - 1] = r_view[r_view.size() - 1];
        const ArrayDescriptor out_descriptor(Shape(out_view), l_dtype, l_device);
        return std::make_shared<MatmulOp>(out_descriptor, broadcasted_l_op, broadcasted_r_op);
    }

    OpPtr iadd(OpPtr l_op, OpPtr r_op) { return inplace_binary<AddOp>(l_op, r_op); }
    OpPtr isub(OpPtr l_op, OpPtr r_op) { return inplace_binary<SubOp>(l_op, r_op); }
    OpPtr imul(OpPtr l_op, OpPtr r_op) { return inplace_binary<MulOp>(l_op, r_op); }
    OpPtr idiv(OpPtr l_op, OpPtr r_op) { return inplace_binary<DivOp>(l_op, r_op); }
    OpPtr eq(OpPtr l_op, OpPtr r_op) { return cmp<EqOp>(l_op, r_op, DtypeCategory::ALL); }
    OpPtr neq(OpPtr l_op, OpPtr r_op) { return cmp<NeqOp>(l_op, r_op, DtypeCategory::ALL); }
    OpPtr lt(OpPtr l_op, OpPtr r_op) { return cmp<LtOp>(l_op, r_op, DtypeCategory::NUMERIC); }
    OpPtr gt(OpPtr l_op, OpPtr r_op) { return cmp<GtOp>(l_op, r_op, DtypeCategory::NUMERIC); }
    OpPtr leq(OpPtr l_op, OpPtr r_op) { return cmp<LeqOp>(l_op, r_op, DtypeCategory::NUMERIC); }
    OpPtr geq(OpPtr l_op, OpPtr r_op) { return cmp<GeqOp>(l_op, r_op, DtypeCategory::NUMERIC); }
    OpPtr minimum(OpPtr l_op, OpPtr r_op) { return elmwise_binary<MinimumOp>(l_op, r_op); }
    OpPtr maximum(OpPtr l_op, OpPtr r_op) { return elmwise_binary<MaximumOp>(l_op, r_op); }
    OpPtr sq(OpPtr in_op, bool in_place) { return unary<SqOp>(in_op, in_place); }
    OpPtr sqrt(OpPtr in_op, bool in_place) { return unary_float<SqrtOp>(in_op, in_place); }
    OpPtr neg(OpPtr in_op, bool in_place) { return unary<NegOp>(in_op, in_place); }

    OpPtr copy(OpPtr in_op) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ArrayDescriptor out_descriptor(Shape(in_descriptor.get_view()), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<CopyOp>(out_descriptor, in_op);
    }

    OpPtr exp(OpPtr in_op, bool in_place) { return unary_float<ExpOp>(in_op, in_place); }
    OpPtr log(OpPtr in_op, bool in_place) { return unary_float<LogOp>(in_op, in_place); }
    OpPtr recip(OpPtr in_op, bool in_place) { return unary_float<RecipOp>(in_op, in_place); }

    OpPtr reshape(OpPtr in_op, const ShapeView &view) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();

        if (in_descriptor.get_view() == view) {
            return in_op;
        }

        const ArrayDescriptor out_descriptor(in_descriptor.get_shape().reshape(view), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<ReshapeOp>(out_descriptor, in_op, view);
    }

    OpPtr permute(OpPtr in_op, const ShapeDims &dims) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ArrayDescriptor out_descriptor(in_descriptor.get_shape().permute(dims), in_descriptor.get_dtype(), in_descriptor.get_device());
        return std::make_shared<PermuteOp>(out_descriptor, in_op, dims);
    }

    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ShapeDims &transpose_dims = in_descriptor.get_shape().transpose(start_dim, end_dim);
        return permute(in_op, transpose_dims);
    }

    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim) {
        const ArrayDescriptor &in_descriptor = in_op->get_descriptor();
        const ShapeView &flattened_view = in_descriptor.get_shape().flatten(start_dim, end_dim);
        return reshape(in_op, flattened_view);
    }

    OpPtr sum(OpPtr in_op, const ShapeDims &dims) { return reduce<SumOp>(in_op, dims, in_op->get_descriptor().get_dtype(), DtypeCategory::NUMERIC); }

    OpPtr mean(OpPtr in_op, const ShapeDims &dims) {
        OpPtr sum_op = sum(in_op, dims);
        isize numel;

        if (dims.empty()) {
            numel = in_op->get_descriptor().get_numel();
        } else {
            const ShapeView &in_view = in_op->get_descriptor().get_view();
            numel = std::accumulate(dims.begin(), dims.end(), 1ll, [&](isize acc, isize dim) { return acc * in_view[dim]; });
        }

        return div(sum_op, numel);
    }

    OpPtr max(OpPtr in_op, const ShapeDims &dims) { return reduce<MaxOp>(in_op, dims, in_op->get_descriptor().get_dtype(), DtypeCategory::NUMERIC); }
    OpPtr min(OpPtr in_op, const ShapeDims &dims) { return reduce<MinOp>(in_op, dims, in_op->get_descriptor().get_dtype(), DtypeCategory::NUMERIC); }
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgmaxOp>(in_op, dims, &i32, DtypeCategory::NUMERIC); }
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgminOp>(in_op, dims, &i32, DtypeCategory::NUMERIC); }
} // namespace nx::core