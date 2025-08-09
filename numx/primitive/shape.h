#pragma once

#include "range.h"

namespace nx::primitive {
    using ShapeView = std::vector<isize>;
    using ShapeStride = std::vector<isize>;
    using ShapeDims = std::vector<isize>;

    class Shape {
    private:
        isize m_offset;
        ShapeView m_view;
        ShapeStride m_stride;

        void if_ranges_are_valid(const RangeVector &ranges) const {
            if (ranges.size() != get_ndim()) {
                throw std::invalid_argument(std::format("The number of ranges does not match the number of dimensions: {} and {}.", ranges.size(), get_ndim()));
            }

            for (size_t i = 0; i < ranges.size(); i++) {
                const Range range = ranges[i];
                if (range.get_start() < 0 || range.get_start() >= static_cast<isize>(m_view[i])) {
                    throw std::invalid_argument(std::format("Start {} is not in the range [0, {}).", range.get_start(), m_view[i]));
                }
                if (range.get_stop() < -1 || range.get_stop() > static_cast<isize>(m_view[i])) {
                    throw std::invalid_argument(std::format("Stop {} is not in the range [-1, {}].", range.get_stop(), m_view[i]));
                }
                if (range.get_step() == 0) {
                    throw std::invalid_argument("Step cannot be zero.");
                }
                if (range.get_start() < range.get_stop() && range.get_step() < 0) {
                    throw std::invalid_argument(std::format("Step {} is not positive when start {} < stop {}.", range.get_step(), range.get_start(), range.get_stop()));
                }
                if (range.get_start() > range.get_stop() && range.get_step() > 0) {
                    throw std::invalid_argument(std::format("Step {} is not negative when start {} > stop {}.", range.get_step(), range.get_start(), range.get_stop()));
                }
            }
        }

        void if_dims_make_valid_permutation(const ShapeDims &dims) const {
            isize ndim = get_ndim();

            if (dims.size() != ndim) {
                throw std::invalid_argument(std::format("The number of dimensions in the specified order does not match the number of dimensions in the shape: {} and {}.", dims.size(), ndim));
            }

            std::vector<bool> dims_used(ndim, false);

            for (auto &dim : dims) {
                if (dim < 0 || dim >= ndim) {
                    throw std::invalid_argument(std::format("The dimension must be in the range [0, {}) but got {}.", ndim, dim));
                }
                dims_used[dim] = true;
            }

            for (auto dim_used : dims_used) {
                if (!dim_used) {
                    throw std::invalid_argument(std::format("The specified order must be a permutation of the dimensions but got {}.", join_nums(dims)));
                }
            }
        }

        void if_start_end_dim_are_valid(isize start_dim, isize end_dim) const {
            isize ndim = get_ndim();

            if (start_dim > end_dim) {
                throw std::invalid_argument("The start dimension must be smaller than the end dimension.");
            }

            if (start_dim < 0 || start_dim >= ndim) {
                throw std::invalid_argument(std::format("The start dimension must be in the range [0, {}) but got {}.", ndim, start_dim));
            }

            if (end_dim < 0 || end_dim >= ndim) {
                throw std::invalid_argument(std::format("The end dimension must be in the range [0, {}) but got {}.", ndim, end_dim));
            }
        }

    public:
        Shape() : Shape(0, {1}, {1}) {}

        Shape(isize offset, const ShapeView &view, const ShapeStride &stride) {
            if_view_is_valid(view);

            if (view.size() != stride.size()) {
                throw std::invalid_argument(std::format("View and stride do not have the same number of dimensions: {} and {}.", view.size(), stride.size()));
            }

            m_offset = offset;
            m_view = view;
            m_stride = stride;
        }

        Shape(isize offset, const ShapeView &view) {
            if_view_is_valid(view);
            m_offset = offset;
            m_view = view;
            m_stride.resize(view.size());
            isize s = 1;

            for (ssize_t i = view.size() - 1; i >= 0; i--) {
                m_stride[i] = s;
                s *= view[i];
            }
        }

        Shape(const ShapeView &view) : Shape(0, view) {}
        Shape(const Shape &shape) : Shape(shape.m_offset, shape.m_view, shape.m_stride) {}

        Shape &operator=(const Shape &shape) {
            m_offset = shape.m_offset;
            m_view = shape.m_view;
            m_stride = shape.m_stride;
            return *this;
        }

        isize get_offset() const { return m_offset; }
        const ShapeView &get_view() const { return m_view; }
        const ShapeStride &get_stride() const { return m_stride; }
        bool is_contiguous() const { return m_stride == get_contiguous_stride(); }

        static void if_view_is_valid(const ShapeView &view) {
            if (view.size() == 0) {
                throw std::invalid_argument("Shape must have at least one dimension.");
            }

            if (std::any_of(view.begin(), view.end(), [](isize v) { return v == 0; })) {
                throw std::invalid_argument("Dimension cannot be zero.");
            }
        }

        ShapeStride get_contiguous_stride() const {
            isize ndim = get_ndim();
            ShapeStride contiguous_stride(ndim, 0);
            isize s = 1;

            for (isize i = ndim - 1; i >= 0; i--) {
                contiguous_stride[i] = s;
                s *= m_view[i];
            }

            return contiguous_stride;
        }

        std::vector<isize> get_elms_per_dim() const {
            isize ndim = get_ndim();
            std::vector<isize> elms_per_dim(ndim, 0);
            isize n = 1;

            for (isize i = ndim - 1; i >= 0; i--) {
                n *= m_view[i];
                elms_per_dim[i] = n;
            }

            return elms_per_dim;
        }

        isize get_ndim() const { return m_view.size(); }
        isize get_numel() const { return std::accumulate(m_view.begin(), m_view.end(), 1ll, std::multiplies<isize>()); }

        bool broadcastable(const ShapeView &view) const {
            if (m_view == view) {
                return true;
            }

            for (auto l_iter = m_view.rbegin(), r_iter = view.rbegin(); l_iter != m_view.rend() && r_iter != view.rend(); l_iter++, r_iter++) {
                if (*l_iter != *r_iter && *l_iter != 1 && *r_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        // One-direction broadcast check
        bool broadcastable_to(const ShapeView &view) const {
            if (m_view == view) {
                return true;
            }

            if (get_ndim() > view.size()) {
                return false;
            }

            for (auto l_iter = m_view.rbegin(), r_iter = view.rbegin(); l_iter != m_view.rend(); l_iter++, r_iter++) {
                if (*l_iter != *r_iter && *l_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        bool matmul_broadcastable(const ShapeView &view) const {
            isize ndim = get_ndim();

            if (ndim < 2 || m_view[ndim - 1] != view[view.size() - 2]) {
                return false;
            }

            for (auto l_iter = m_view.begin(), r_iter = view.begin(); l_iter != m_view.end() - 2 && r_iter != view.end() - 2; l_iter++, r_iter++) {
                if (*l_iter != *r_iter && *l_iter != 1 && *r_iter != 1) {
                    return false;
                }
            }

            return true;
        }

        // One-direction broadcast
        std::pair<Shape, ShapeDims> broadcast_to(const ShapeView &view) const {
            ShapeDims broadcast_dims;

            if (m_view == view) {
                return std::make_pair(*this, broadcast_dims);
            }

            if (!broadcastable_to(view)) {
                throw std::invalid_argument(std::format("Cannot broadcast shape ({}) to ({}).", join_nums(m_view), join_nums(view)));
            }

            ShapeView broadcast_view = m_view;
            size_t ndim_diff = view.size() - broadcast_view.size();
            broadcast_view.insert(broadcast_view.begin(), ndim_diff, 1);
            Shape broadcast_shape(m_offset, broadcast_view);
            std::fill_n(broadcast_shape.m_stride.begin(), ndim_diff, 0);

            for (size_t i = 0; i < view.size(); i++) {
                if (broadcast_view[i] < view[i]) {
                    broadcast_dims.emplace_back(i);
                    broadcast_shape.m_view[i] = view[i];
                    broadcast_shape.m_stride[i] = 0;
                }
            }

            return {broadcast_shape, broadcast_dims};
        }

        // ShapeDims specifies which dimensions are broadcasted
        std::pair<Shape, ShapeDims> broadcast(const ShapeView &view) const {
            ShapeDims broadcast_dims;

            if (m_view == view) {
                return std::make_pair(*this, broadcast_dims);
            }

            if (!broadcastable(view)) {
                throw std::invalid_argument(std::format("Cannot broadcast shape ({}) and ({}).", join_nums(m_view), join_nums(view)));
            }

            ShapeView l_view = m_view;
            ShapeView r_view = view;
            size_t ndim = std::max(l_view.size(), r_view.size());
            size_t l_diff = ndim - l_view.size();
            size_t r_diff = ndim - r_view.size();
            l_view.insert(l_view.begin(), l_diff, 1);
            r_view.insert(r_view.begin(), r_diff, 1);
            Shape broadcast_shape(m_offset, l_view);
            std::fill_n(broadcast_shape.m_stride.begin(), l_diff, 0);

            for (size_t i = 0; i < ndim; i++) {
                if (l_view[i] < r_view[i]) {
                    broadcast_dims.emplace_back(i);
                    broadcast_shape.m_view[i] = r_view[i];
                    broadcast_shape.m_stride[i] = 0;
                }
            }

            return {broadcast_shape, broadcast_dims};
        }

        Shape reshape(const ShapeView &view) const {
            // TODO: fix this
            if_view_is_valid(view);
            isize l_numel = get_numel();
            isize r_numel = std::accumulate(view.begin(), view.end(), 1ll, std::multiplies<isize>());

            if (l_numel != r_numel) {
                throw std::invalid_argument(std::format("Cannot reshape array of {} elements to {} elements.", l_numel, r_numel));
            }

            return Shape(m_offset, view);
        }

        ShapeDims transpose(isize start_dim, isize end_dim) const {
            if_start_end_dim_are_valid(start_dim, end_dim);
            ShapeDims transpose_dims(get_ndim());
            std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
            std::reverse(transpose_dims.begin() + start_dim, transpose_dims.begin() + end_dim + 1);
            return transpose_dims;
        }

        ShapeView flatten(isize start_dim, isize end_dim) const {
            if_start_end_dim_are_valid(start_dim, end_dim);
            ShapeView flattened_view = m_view;
            isize prod = std::accumulate(flattened_view.begin() + start_dim, flattened_view.begin() + end_dim + 1, 1ll, std::multiplies<isize>());
            // Erase from start_dim + 1 to end_dim + 1
            flattened_view.erase(flattened_view.begin() + start_dim + 1, flattened_view.begin() + end_dim + 1);
            // Update view at start_dim
            flattened_view[start_dim] = prod;
            return flattened_view;
        }

        Shape permute(const ShapeDims &dims) const {
            if_dims_make_valid_permutation(dims);
            isize ndim = get_ndim();
            ShapeView view(ndim, 0);
            ShapeStride stride(ndim, 0);

            for (isize i = 0; i < ndim; i++) {
                view[i] = m_view[dims[i]];
                stride[i] = m_stride[dims[i]];
            }

            return Shape(m_offset, view, stride);
        }

        ShapeView undo_permute_view(const ShapeDims &dims) const {
            if_dims_make_valid_permutation(dims);
            ShapeView reverse_dims(dims.size());

            for (size_t i = 0; i < dims.size(); i++) {
                reverse_dims[dims[i]] = i;
            }

            return reverse_dims;
        }

        Shape undo_permute(const ShapeDims &dims) const {
            return permute(undo_permute_view(dims));
        }

        Shape slice(const RangeVector &ranges) const {
            if_ranges_are_valid(ranges);
            isize offset = m_offset;

            for (size_t i = 0; i < ranges.size(); i++) {
                offset += ranges[i].get_start() * m_stride[i];
            }

            isize ndim = get_ndim();
            ShapeView view(ndim);
            ShapeStride stride(ndim);

            for (size_t i = 0; i < ranges.size(); i++) {
                const Range &range = ranges[i];
                isize diff = std::abs(range.get_stop() - range.get_start());
                view[i] = static_cast<isize>(ceil((static_cast<double>(diff)) / std::abs(range.get_step())));
                stride[i] = m_stride[i] * range.get_step();
            }

            return Shape(offset, view, stride);
        }

        Shape unsqueeze(const ShapeDims &dims) const {
            ShapeView view = m_view;
            ShapeStride stride = m_stride;

            if (dims.empty()) {
                view.push_back(1);
                stride.push_back(1);
                return Shape(m_offset, view, stride);
            }

            // Check for duplicates
            // Set is red black tree under the hood, which can be used for both sorting and checking for uniqueness
            std::set<isize> unique_dims(dims.begin(), dims.end());

            if (unique_dims.size() != dims.size()) {
                throw std::invalid_argument("Duplicate dimensions in unsqueeze.");
            }

            isize ndim = get_ndim();

            // Check for invalid dimension
            for (auto &dim : unique_dims) {
                if (dim < 0 || dim > ndim) {
                    throw std::invalid_argument(std::format("Dimension {} is out of range [0, {}] during unsqueeze.", dim, ndim));
                }
            }

            // Process in descending order using reverse iterator
            for (auto iter = unique_dims.rbegin(); iter != unique_dims.rend(); ++iter) {
                view.insert(view.begin() + *iter, 1);
                stride.insert(stride.begin() + *iter, 0);
            }

            return Shape(m_offset, view, stride);
        }

        Shape squeeze(const ShapeDims &dims) const {
            ShapeView view = m_view;
            ShapeStride stride = m_stride;

            if (dims.empty()) {
                for (ssize_t i = m_view.size() - 1; i >= 0; i--) {
                    if (m_view[i] == 1) {
                        view.erase(view.begin() + i);
                        stride.erase(stride.begin() + i);
                    }
                }

                return Shape(m_offset, view, stride);
            }

            // Check for duplicates
            std::set<isize> unique_dims(dims.begin(), dims.end());

            if (unique_dims.size() != dims.size()) {
                throw std::invalid_argument("Duplicate dimensions in squeeze.");
            }

            isize ndim = get_ndim();

            // Check for invalid dimension
            for (auto &dim : unique_dims) {
                if (dim < 0 || dim >= ndim) {
                    throw std::invalid_argument(std::format("Dimension {} is out of range [0, {}) during squeeze.", dim, ndim));
                }
                if (m_view[dim] != 1) {
                    throw std::invalid_argument(std::format("Dimension {} is not a singleton during squeeze.", dim));
                }
            }

            // Process in descending order using reverse iterator
            for (auto iter = unique_dims.rbegin(); iter != unique_dims.rend(); ++iter) {
                view.erase(view.begin() + *iter);
                stride.erase(stride.begin() + *iter);
            }

            return Shape(m_offset, view, stride);
        }

        bool operator==(const Shape &shape) const { return m_view == shape.m_view; }
        isize operator[](isize dim) const { return m_view[dim]; }
        ShapeView::const_iterator begin() const { return m_view.cbegin(); }
        ShapeView::const_iterator end() const { return m_view.cend(); }
        const std::string str() const { return std::format("offset: {}, view: ({}), stride: ({})", m_offset, join_nums(m_view), join_nums(m_stride)); }
        friend std::ostream &operator<<(std::ostream &os, const Shape &shape) { return os << shape.str(); }
    };
} // namespace nx::primitive

namespace std {
    template <>
    struct formatter<nx::primitive::Shape> : formatter<string> {
        auto format(const nx::primitive::Shape &shape, format_context &ctx) const {
            return formatter<string>::format(shape.str(), ctx);
        }
    };
} // namespace std