#include "functional.h"

namespace nx::graph {
    isize item(OpPtr op) {
        const ArrayData &data = op->get_data();

        if (data.get_numel() != 1) {
            throw std::runtime_error(std::format("Array {} must have exactly one element but has {} elements.", data.get_id(), data.get_numel()));
        }

        ArrayIterator iter(data);
        iter.start();
        uint8_t *ptr = iter.next();
        return data.get_dtype()->bit_cast(ptr);
    }

    OpPtr detach(OpPtr op) {
        const ArrayData &data = op->get_data();
        return from_buffer(data.m_buffer.get_ptr(), data.m_buffer.get_size(), data.get_shape(), data.get_dtype(), data.get_device());
    }

    OpPtr from_buffer(uint8_t *ptr, isize size, const Shape &shape, DtypePtr dtype, DevicePtr device) {
        ArrayData data = ArrayData::from_buffer(ptr, size, shape, dtype, device);
        return std::make_shared<Nop>(data);
    }

    OpPtr empty(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return std::make_shared<EmptyOp>(ArrayData(Shape(view), dtype, device)); }
    OpPtr empty_like(OpPtr op, DtypePtr dtype, DevicePtr device) { return empty(op->get_data().get_view(), dtype, device); }

    OpPtr empty_like(OpPtr op) {
        const ArrayData &data = op->get_data();
        return empty(data.get_view(), data.get_dtype(), data.get_device());
    }

    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 0, dtype, device); }
    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 0, dtype, device); }
    OpPtr zeros_like(OpPtr in_op) { return full_like(in_op, 0); }
    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 1, dtype, device); }
    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 1, dtype, device); }
    OpPtr ones_like(OpPtr in_op) { return full_like(in_op, 1); }

    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device) {
        return std::make_shared<ArangeOp>(ArrayData(Shape(view), dtype, device), start, step);
    }

    OpPtr uniform(const ShapeView &view, const RandomState &state, isize low, isize high, DtypePtr dtype, DevicePtr device) {
        return std::make_shared<UniformOp>(ArrayData(Shape(view), dtype, device), state, low, high);
    }

    OpPtr broadcast(OpPtr in_op, const ShapeView &view) {
        const ArrayData &in_data = in_op->get_data();
        const Shape &in_shape = in_data.get_shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return in_op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast(view);

        if (in_shape == broadcast_shape) {
            return in_op;
        }

        const ArrayData out_data(broadcast_shape, in_data.get_dtype(), in_data.get_device());
        return std::make_shared<BroadcastOp>(out_data, in_op, in_view, broadcast_dims);
    }

    OpPtr broadcast_to(OpPtr in_op, const ShapeView &view) {
        const ArrayData &in_data = in_op->get_data();
        const Shape &in_shape = in_data.get_shape();
        const ShapeView &in_view = in_shape.get_view();

        if (in_view == view) {
            return in_op;
        }

        auto [broadcast_shape, broadcast_dims] = in_shape.broadcast_to(view);

        if (in_shape == broadcast_shape) {
            return in_op;
        }

        const ArrayData out_data(broadcast_shape, in_data.get_dtype(), in_data.get_device());
        return std::make_shared<BroadcastOp>(out_data, in_op, in_view, broadcast_dims);
    }

    OpPtr slice(OpPtr in_op, const RangeVec &ranges) {
        const ArrayData &in_data = in_op->get_data();
        const ArrayData out_data(in_data.get_shape().slice(ranges), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<SliceOp>(out_data, in_op, ranges);
    }

    OpPtr astype(OpPtr in_op, DtypePtr dtype) {
        const ArrayData &in_data = in_op->get_data();

        if (in_data.get_dtype() == dtype) {
            return in_op;
        }

        const ArrayData out_data(in_data.get_shape(), dtype, in_data.get_device());
        return std::make_shared<AstypeOp>(out_data, in_op, dtype);
    }

    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayData &in_data = in_op->get_data();
        const ArrayData out_data(in_data.get_shape().unsqueeze(dims), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<UnsqueezeOp>(out_data, in_op, dims);
    }

    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims) {
        const ArrayData &in_data = in_op->get_data();
        const ArrayData out_data(in_data.get_shape().squeeze(dims), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<SqueezeOp>(out_data, in_op, dims);
    }

    OpPtr add(OpPtr l_op, OpPtr r_op) { return elmwise_binary<AddOp>(l_op, r_op); }
    OpPtr sub(OpPtr l_op, OpPtr r_op) { return elmwise_binary<SubOp>(l_op, r_op); }
    OpPtr mul(OpPtr l_op, OpPtr r_op) { return elmwise_binary<MulOp>(l_op, r_op); }
    OpPtr div(OpPtr l_op, OpPtr r_op) { return elmwise_binary<DivOp>(l_op, r_op); }

    OpPtr matmul(OpPtr l_op, OpPtr r_op) {
        const ArrayData &l_data = l_op->get_data();
        const ArrayData &r_data = r_op->get_data();
        const ShapeView &l_view = l_data.get_view();
        const ShapeView &r_view = r_data.get_view();
        DtypePtr l_dtype = l_data.get_dtype(), r_dtype = r_data.get_dtype();
        DevicePtr l_device = l_data.get_device(), r_device = r_data.get_device();

        if (!l_data.get_shape().matmul_broadcastable(r_view)) {
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
        const ArrayData out_data(Shape(out_view), l_dtype, l_device);
        return std::make_shared<MatmulOp>(out_data, broadcasted_l_op, broadcasted_r_op);
    }

    OpPtr iadd(OpPtr l_op, OpPtr r_op) { return in_place_binary<AddOp>(l_op, r_op); }
    OpPtr isub(OpPtr l_op, OpPtr r_op) { return in_place_binary<SubOp>(l_op, r_op); }
    OpPtr imul(OpPtr l_op, OpPtr r_op) { return in_place_binary<MulOp>(l_op, r_op); }
    OpPtr idiv(OpPtr l_op, OpPtr r_op) { return in_place_binary<DivOp>(l_op, r_op); }
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
        const ArrayData &in_data = in_op->get_data();
        const ArrayData out_data(Shape(in_data.get_view()), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<CopyOp>(out_data, in_op);
    }

    OpPtr exp(OpPtr in_op, bool in_place) { return unary_float<ExpOp>(in_op, in_place); }
    OpPtr log(OpPtr in_op, bool in_place) { return unary_float<LogOp>(in_op, in_place); }
    OpPtr recip(OpPtr in_op, bool in_place) { return unary_float<RecipOp>(in_op, in_place); }

    OpPtr reshape(OpPtr in_op, const ShapeView &view) {
        const ArrayData &in_data = in_op->get_data();

        if (in_data.get_view() == view) {
            return in_op;
        }

        const ArrayData out_data(in_data.get_shape().reshape(view), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<ReshapeOp>(out_data, in_op);
    }

    OpPtr permute(OpPtr in_op, const ShapeDims &dims) {
        const ArrayData &in_data = in_op->get_data();
        const ArrayData out_data(in_data.get_shape().permute(dims), in_data.get_dtype(), in_data.get_device());
        return std::make_shared<PermuteOp>(out_data, in_op, dims);
    }

    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim) {
        const ArrayData &in_data = in_op->get_data();
        const ShapeDims &transpose_dims = in_data.get_shape().transpose(start_dim, end_dim);
        return permute(in_op, transpose_dims);
    }

    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim) {
        const ArrayData &in_data = in_op->get_data();
        const ShapeView &flattened_view = in_data.get_shape().flatten(start_dim, end_dim);
        return reshape(in_op, flattened_view);
    }

    OpPtr sum(OpPtr in_op, const ShapeDims &dims) { return reduce<SumOp>(in_op, dims, in_op->get_data().get_dtype(), DtypeCategory::NUMERIC); }

    OpPtr mean(OpPtr in_op, const ShapeDims &dims) {
        OpPtr sum_op = sum(in_op, dims);
        isize numel;

        if (dims.empty()) {
            numel = in_op->get_data().get_numel();
        } else {
            const ShapeView &in_view = in_op->get_data().get_view();
            numel = std::accumulate(dims.begin(), dims.end(), 1ll, [&](isize acc, isize dim) { return acc * in_view[dim]; });
        }

        return div(sum_op, numel);
    }

    OpPtr max(OpPtr in_op, const ShapeDims &dims) { return reduce<MaxOp>(in_op, dims, in_op->get_data().get_dtype(), DtypeCategory::NUMERIC); }
    OpPtr min(OpPtr in_op, const ShapeDims &dims) { return reduce<MinOp>(in_op, dims, in_op->get_data().get_dtype(), DtypeCategory::NUMERIC); }
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgmaxOp>(in_op, dims, &i32, DtypeCategory::NUMERIC); }
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgminOp>(in_op, dims, &i32, DtypeCategory::NUMERIC); }

    OpPtr expand(OpPtr in_op, const ShapeView &reduce_operand_view, const ShapeDims &remaining_dims, const ShapeDims &reduce_dims) {
        // TODO: check if remaining_dims and reduce_dims are valid?
        isize reduce_numel = std::accumulate(reduce_dims.begin(), reduce_dims.end(), 1ll, [&](isize acc, isize dim) { return acc * reduce_operand_view[dim]; });

        // Broadcast the last dimension so the last dimension is the number of reduced elements
        ShapeView broadcasted_view(remaining_dims.size() + 1, reduce_numel);
        std::transform(remaining_dims.begin(), remaining_dims.end(), broadcasted_view.begin(), [&](isize dim) { return reduce_operand_view[dim]; });
        OpPtr out_op = broadcast(in_op, broadcasted_view);

        // Reshape so the last k dimensions are the reduced dimensions
        ShapeView reshaped_view(remaining_dims.size() + reduce_dims.size());
        std::transform(remaining_dims.begin(), remaining_dims.end(), reshaped_view.begin(), [&](isize dim) { return reduce_operand_view[dim]; });
        std::transform(reduce_dims.begin(), reduce_dims.end(), reshaped_view.begin() + remaining_dims.size(), [&](isize dim) { return reduce_operand_view[dim]; });
        out_op = reshape(out_op, reshaped_view);

        // Permute to restore the shape before reduction
        ShapeDims permuted_view;
        permuted_view.reserve(remaining_dims.size() + reduce_dims.size());
        permuted_view.insert(permuted_view.end(), remaining_dims.begin(), remaining_dims.end());
        permuted_view.insert(permuted_view.end(), reduce_dims.begin(), reduce_dims.end());
        const ShapeView &out_view = out_op->get_data().get_shape().undo_permute_view(permuted_view);
        out_op = permute(out_op, out_view);
        return out_op;
    }
} // namespace nx::graph