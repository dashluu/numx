#include "op.h"
#include "functional.h"

namespace nx::primitive {
    void Op::zero_grad() {
        if (!m_grad) {
            m_grad = zeros_like(nonconst_from_this());
            m_partial_grad = m_grad;
        }
    }

    void Op::one_grad() {
        if (!m_grad) {
            m_grad = ones_like(nonconst_from_this());
            m_partial_grad = m_grad;
        }
    }

    OpPtr Op::detach_this() const { return detach(nonconst_from_this()); }
    void Op::iadd_grad(OpPtr grad) { m_partial_grad = iadd(m_partial_grad, grad); }
    void Op::isub_grad(OpPtr grad) { m_partial_grad = isub(m_partial_grad, grad); }
    void Op::slice_grad(OpPtr grad, const RangeVector &ranges) { m_partial_grad = slice(grad, ranges); }

    void AddOp::grad_fn() const {
        // In-place or not, gradient should be computed properly
        // z = x + y
        // dx += dz
        // dy += dz
        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            m_lhs->iadd_grad(m_grad);
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            m_rhs->iadd_grad(m_grad);
        }
    }

    void SubOp::grad_fn() const {
        // z = x - y
        // dx += dz
        // dy -= dz
        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            m_lhs->iadd_grad(m_grad);
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            m_rhs->isub_grad(m_grad);
        }
    }

    void MulOp::grad_fn() const {
        // z = x * y
        // dx += dz * y
        // dy += dz * x
        // Use detach to prevent circular dependencies
        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            m_lhs->iadd_grad(mul(m_grad, detach(m_rhs)));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            m_rhs->iadd_grad(mul(m_grad, detach(m_lhs)));
        }
    }

    void DivOp::grad_fn() const {
        // z = x / y
        // dx += dz * (1/y)
        // dy += dz * (-x/y**2)
        // dy -= dz * (z/y)
        // Use detach to prevent circular dependencies
        OpPtr d_rhs = detach(m_rhs);

        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            m_lhs->iadd_grad(div(m_grad, d_rhs));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            m_rhs->isub_grad(mul(m_grad, div(detach_this(), d_rhs)));
        }
    }

    void MinimumOp::grad_fn() const {
        // z = min(x, y)
        // dx += dz * (1 where x is min and 0 otherwise)
        // dy += dz * (1 where y is min and 0 otherwise)
        OpPtr d_out = detach_this();

        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            OpPtr l_minimum = astype(eq(detach(m_lhs), d_out), d_out->get_data().get_dtype());
            m_lhs->iadd_grad(mul(m_grad, l_minimum));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            OpPtr r_minimum = astype(eq(detach(m_rhs), d_out), d_out->get_data().get_dtype());
            m_rhs->iadd_grad(mul(m_grad, r_minimum));
        }
    }

    void MaximumOp::grad_fn() const {
        // z = max(x, y)
        // dx += dz * (1 where x is max and 0 otherwise)
        // dy += dz * (1 where y is max and 0 otherwise)
        OpPtr d_out = detach_this();

        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            OpPtr l_maximum = astype(eq(detach(m_lhs), d_out), d_out->get_data().get_dtype());
            m_lhs->iadd_grad(mul(m_grad, l_maximum));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            OpPtr r_maximum = astype(eq(detach(m_rhs), d_out), d_out->get_data().get_dtype());
            m_rhs->iadd_grad(mul(m_grad, r_maximum));
        }
    }

    void MatmulOp::grad_fn() const {
        // Transpose the last two dimensions of m_lhs and m_rhs
        // z = x @ y
        // dx += dz @ y^T
        // dy += x^T @ dz
        isize ndim = m_lhs->get_data().get_ndim();

        if (m_lhs->is_grad_enabled()) {
            m_lhs->zero_grad();
            m_lhs->iadd_grad(matmul(m_grad, transpose(detach(m_rhs), ndim - 2, ndim - 1)));
        }

        if (m_rhs->is_grad_enabled()) {
            m_rhs->zero_grad();
            m_rhs->iadd_grad(matmul(transpose(detach(m_lhs), ndim - 2, ndim - 1), m_grad));
        }
    }

    void SqOp::grad_fn() const {
        // z = x**2
        // dx += dz * (2*x)
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(mul(m_grad, mul(detach(m_operand), 2.0f)));
        }
    }

    void SqrtOp::grad_fn() const {
        // z = sqrt(x)
        // dx += dz / (2*sqrt(x))
        // dx += dz / (2*z)
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(div(m_grad, mul(detach_this(), 2.0f)));
        }
    }

    void NegOp::grad_fn() const {
        // z = -x
        // dx += dz * -1
        // dx -= dz
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->isub_grad(m_grad);
        }
    }

    void CopyOp::grad_fn() const {
        // z = x
        // dx += dz
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(m_grad);
        }
    }

    void ExpOp::grad_fn() const {
        // z = exp(x)
        // dx += dz * exp(x)
        // dx += dz * z
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(mul(m_grad, detach_this()));
        }
    }

    void LogOp::grad_fn() const {
        // z = log(x)
        // dx += dz / x
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(div(m_grad, detach(m_operand)));
        }
    }

    void RecipOp::grad_fn() const {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->isub_grad(mul(m_grad, sq(detach_this())));
        }
    }

    void SinOp::grad_fn() const {
        // z = sin(x)
        // dx += dz * cos(x)
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(mul(m_grad, cos(detach(m_operand))));
        }
    }

    void CosOp::grad_fn() const {
        // z = cos(x)
        // dx -= dz * sin(x)
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->isub_grad(mul(m_grad, sin(detach(m_operand))));
        }
    }

    void SliceOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->slice_grad(m_operand->get_grad(), m_ranges);
            m_operand->iadd_grad(m_grad);
        }
    }

    void ReshapeOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(reshape(m_grad, m_operand->get_data().get_view()));
        }
    }

    void PermuteOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            const ShapeView &reverse_dims = m_grad->get_data().get_shape().undo_permute_view(m_dims);
            m_operand->iadd_grad(permute(m_grad, reverse_dims));
        }
    }

    void BroadcastOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(reshape(sum(m_grad, m_dims), m_input_view));
        }
    }

    void SqueezeOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(unsqueeze(m_grad, m_dims));
        }
    }

    void UnsqueezeOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(squeeze(m_grad, m_dims));
        }
    }

    void SumOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            m_operand->iadd_grad(expand(m_grad, m_operand->get_data().get_view(), m_remaining_dims, m_reduce_dims));
        }
    }

    void MaxOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            const ShapeView &operand_view = m_operand->get_data().get_view();
            OpPtr mask = eq(detach(m_operand), expand(detach_this(), operand_view, m_remaining_dims, m_reduce_dims));
            m_operand->iadd_grad(mul(astype(mask, m_operand->get_data().get_dtype()), expand(m_grad, operand_view, m_remaining_dims, m_reduce_dims)));
        }
    }

    void MinOp::grad_fn() const {
        if (m_operand->is_grad_enabled()) {
            m_operand->zero_grad();
            const ShapeView &operand_view = m_operand->get_data().get_view();
            OpPtr mask = eq(detach(m_operand), expand(detach_this(), operand_view, m_remaining_dims, m_reduce_dims));
            m_operand->iadd_grad(mul(astype(mask, m_operand->get_data().get_dtype()), expand(m_grad, operand_view, m_remaining_dims, m_reduce_dims)));
        }
    }
} // namespace nx::primitive