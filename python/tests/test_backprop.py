from numx.core import Array, Shape, from_numpy
from numx.profiler import enable_memory_profile
import numpy as np
import torch


def assert_array(array: Array, tensor: torch.Tensor):
    assert torch.allclose(array.torch(), tensor, atol=1e-3, rtol=0)


class TestBackprop:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    @staticmethod
    def make_shape() -> Shape:
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        return shape

    def test_backprop_v1(self):
        print("backprop 1:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.randn(*shape).astype(np.float32)
        np_a2 = np.random.randn(*shape).astype(np.float32)
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3 = nx_a1 + nx_a2
        nx_a4 = nx_a1 * nx_a2
        nx_a5 = nx_a3 + nx_a4
        nx_a6 = nx_a3 * nx_a4
        nx_a7 = nx_a5 + nx_a6
        nx_a8 = nx_a7.sum()
        nx_a8.backward()
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = torch.from_numpy(np_a2).requires_grad_(True)
        t3 = t1 + t2
        t3.retain_grad()
        t4 = t1 * t2
        t4.retain_grad()
        t5 = t3 + t4
        t5.retain_grad()
        t6 = t3 * t4
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)
        assert_array(nx_a6.grad, t6.grad)
        assert_array(nx_a7.grad, t7.grad)

    def test_backprop_v2(self):
        print("Testing complex unary(and one binary) operations chain:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)  # Positive values for log

        # Implementation: log(exp(x) * x) / x
        nx_a1 = from_numpy(np_a1)
        nx_a2 = nx_a1.exp()
        nx_a3 = nx_a2 * nx_a1
        nx_a4 = nx_a3.log()
        nx_a5 = nx_a4 / nx_a1
        nx_a6 = nx_a5.sum()
        nx_a6.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = t1.exp()
        t2.retain_grad()
        t3 = t2 * t1
        t3.retain_grad()
        t4 = t3.log()
        t4.retain_grad()
        t5 = t4 / t1
        t5.retain_grad()
        t6 = t5.sum()
        t6.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)

    def test_backprop_v3(self):
        print("Testing branched operations:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np_a2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3 = nx_a1.log()
        nx_a4 = nx_a2.exp()
        nx_a5 = nx_a3 * nx_a4
        nx_a6 = nx_a1 / nx_a2
        nx_a7 = nx_a5 + nx_a6
        nx_a8 = nx_a7.maximum(nx_a6)
        nx_a9 = nx_a8.sum()
        nx_a9.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = torch.from_numpy(np_a2).requires_grad_(True)
        t3 = t1.log()
        t3.retain_grad()
        t4 = t2.exp()
        t4.retain_grad()
        t5 = t3 * t4
        t5.retain_grad()
        t6 = t1 / t2
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.maximum(t6)
        t8.retain_grad()
        t9 = t8.sum()
        t9.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)
        assert_array(nx_a6.grad, t6.grad)
        assert_array(nx_a7.grad, t7.grad)
        assert_array(nx_a8.grad, t8.grad)

    def test_backprop_v4(self):
        print("Testing nested operations:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np_a2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation: log(exp(x1/x2) * recip(x1))
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3 = nx_a1 / nx_a2
        nx_a4 = nx_a3.exp()
        nx_a5 = nx_a1.recip()
        nx_a6 = nx_a4 * nx_a5
        nx_a7 = nx_a6.log()
        nx_a8 = nx_a7.sum()
        nx_a8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = torch.from_numpy(np_a2).requires_grad_(True)
        t3 = t1 / t2
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = 1.0 / t1
        t5.retain_grad()
        t6 = t4 * t5
        t6.retain_grad()
        t7 = t6.log()
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)
        assert_array(nx_a6.grad, t6.grad)
        assert_array(nx_a7.grad, t7.grad)

    def test_backprop_v5(self):
        print("Testing square and sqrt operations:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation: sqrt(x^2) + x^2/sqrt(x)
        nx_a1 = from_numpy(np_a1)
        nx_a2 = nx_a1.sq()
        nx_a3 = nx_a2.sqrt()
        nx_a4 = nx_a1.sq()
        nx_a5 = nx_a1.sqrt()
        nx_a6 = nx_a4 / nx_a5
        nx_a7 = nx_a3 + nx_a6
        nx_a8 = nx_a7.sum()
        nx_a8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = t1 * t1
        t2.retain_grad()
        t3 = t2.sqrt()
        t3.retain_grad()
        t4 = t1 * t1
        t4.retain_grad()
        t5 = t1.sqrt()
        t5.retain_grad()
        t6 = t4 / t5
        t6.retain_grad()
        t7 = t3 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)
        assert_array(nx_a6.grad, t6.grad)
        assert_array(nx_a7.grad, t7.grad)

    def test_backprop_v8(self):
        print("Testing double backpropagation with complex operations:")
        shape = TestBackprop.make_shape()
        np_a1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np_a2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # numx implementation
        # f(x1, x2) = log(sqrt(x1^2) * exp(x2/x1)) + (x1 * sqrt(x2))^2
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3 = nx_a1.sq()
        nx_a4 = nx_a3.sqrt()
        nx_a5 = nx_a2 / nx_a1
        nx_a6 = nx_a5.exp()
        nx_a7 = nx_a4 * nx_a6
        nx_a8 = nx_a7.minimum(nx_a4)
        nx_a9 = nx_a8.log()
        nx_a10 = nx_a2.sqrt()
        nx_a11 = nx_a1 * nx_a10
        nx_a12 = nx_a11.sq()
        nx_a13 = nx_a9 + nx_a12
        nx_a14 = nx_a13.sum()
        nx_a14.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = torch.from_numpy(np_a2).requires_grad_(True)
        t3 = t1 * t1
        t3.retain_grad()
        t4 = t3.sqrt()
        t4.retain_grad()
        t5 = t2 / t1
        t5.retain_grad()
        t6 = t5.exp()
        t6.retain_grad()
        t7 = t4 * t6
        t7.retain_grad()
        t8 = t7.minimum(t4)
        t8.retain_grad()
        t9 = t8.log()
        t9.retain_grad()
        t10 = t2.sqrt()
        t10.retain_grad()
        t11 = t1 * t10
        t11.retain_grad()
        t12 = t11.square()
        t12.retain_grad()
        t13 = t9 + t12
        t13.retain_grad()
        t14 = t13.sum()
        t14.backward(retain_graph=True)

        # Compare backward pass gradients
        print("\nChecking backward pass:")
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
        assert_array(nx_a4.grad, t4.grad)
        assert_array(nx_a5.grad, t5.grad)
        assert_array(nx_a6.grad, t6.grad)
        assert_array(nx_a7.grad, t7.grad)
        assert_array(nx_a8.grad, t8.grad)
        assert_array(nx_a9.grad, t9.grad)
        assert_array(nx_a10.grad, t10.grad)
        assert_array(nx_a11.grad, t11.grad)
        assert_array(nx_a12.grad, t12.grad)

    def test_permute_binary_backprop(self):
        """Test backprop through permute and binary op"""
        # Forward: (2,3,4) -> (4,2,3) * (4,2,3)
        x = torch.randn(2, 3, 4, dtype=torch.float32)
        y = torch.randn(4, 2, 3, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = from_numpy(y.numpy())
        nx_a3 = nx_a1.permute([2, 0, 1]) * nx_a2
        nx_a4 = nx_a3.sum()
        nx_a4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1.permute(2, 0, 1) * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)

    def test_backprop_v6(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = nx_a1.permute([0, 2, 1, 3]).reshape([8, 3, 5])  # (2,4,3,5)  # (8,3,5)
        nx_a3 = nx_a2.exp()
        nx_a4 = nx_a3.sum()
        nx_a4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1.permute(0, 2, 1, 3).reshape(8, 3, 5)  # (2,4,3,5)  # (8,3,5)
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)

    def test_backprop_v7(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> matmul -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
        # TODO: can try doing matmul with broadcast
        y = torch.randn(8, 5, 2, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = from_numpy(y.numpy())
        nx_a3 = nx_a1.permute([0, 2, 1, 3]).reshape([8, 3, 5]) @ nx_a2  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        nx_a4 = nx_a3.exp()
        nx_a5 = nx_a4.sum()
        nx_a5.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1.permute(0, 2, 1, 3).reshape(8, 3, 5) @ t2  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)

    def test_slice_basic_backprop(self):
        """Test basic slicing backpropagation"""
        print("\nTesting basic slice backprop:")
        x = torch.randn(4, 6, 8, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = nx_a1[1:3, ::2, ::1]  # Basic slicing
        nx_a3 = nx_a2.sum()
        nx_a3.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:3, ::2, ::1]
        t2.retain_grad()
        t3 = t2.sum()
        t3.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)

    def test_slice_with_unary_backprop(self):
        """Test slicing combined with unary operations"""
        print("\nTesting slice with unary ops backprop:")
        x = torch.randn(3, 4, 5, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = nx_a1[::2, 1:3]  # Slice first
        nx_a3 = nx_a2.exp()  # Then unary op
        nx_a4 = nx_a3.sum()
        nx_a4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[::2, 1:3]
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)

    def test_slice_with_binary_backprop(self):
        """Test slicing combined with binary operations"""
        print("\nTesting slice with binary ops backprop:")

        x = torch.randn(4, 6, 8, dtype=torch.float32)
        y = torch.randn(2, 6, 8, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = from_numpy(y.numpy())
        nx_a3 = nx_a1[::2] * nx_a2  # Slice and multiply
        nx_a4 = nx_a3.sum()
        nx_a4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2] * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)

    def test_slice_chain_backprop(self):
        """Test chain of slice operations"""
        print("\nTesting slice chain backprop:")
        x = torch.randn(5, 6, 7, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = nx_a1[1:4, ::2]  # First slice
        nx_a3 = nx_a2[:, 1::2]  # Second slice
        nx_a4 = nx_a3.sum()
        nx_a4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:4, ::2]
        t2.retain_grad()
        t3 = t2[:, 1::2]
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)

    def test_slice_complex_chain_backprop(self):
        """Test complex chain with slicing, unary and binary operations"""
        print("\nTesting complex slice chain backprop:")

        x = torch.randn(4, 5, 6, dtype=torch.float32)
        y = torch.randn(2, 5, 3, dtype=torch.float32)

        # numx implementation
        nx_a1 = from_numpy(x.numpy())
        nx_a2 = from_numpy(y.numpy())
        nx_a3 = nx_a1[::2, :, ::2]  # Initial slice
        nx_a4 = nx_a3.exp()  # Unary op
        nx_a5 = nx_a4 * nx_a2  # Binary op
        nx_a6 = nx_a5[:, 1:4]  # Another slice
        nx_a7 = nx_a6.sum()
        nx_a7.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2, :, ::2]
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4 * t2
        t5.retain_grad()
        t6 = t5[:, 1:4]
        t6.retain_grad()
        t7 = t6.sum()
        t7.backward()

        # Compare gradients
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)

    def test_linear_backprop(self):
        """Test backprop through matmul"""
        print("\nTesting matmul backprop:")
        np_a1 = np.random.randn(64, 784).astype(np.float32)
        np_a2 = np.random.randn(10, 784).astype(np.float32)
        np_a3 = np.random.randn(10).astype(np.float32)
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3 = from_numpy(np_a3)
        nx_a4 = nx_a1 @ nx_a2.transpose(-2, -1) + nx_a3
        nx_a5 = nx_a4.sum()
        nx_a5.backward()
        t1 = torch.from_numpy(np_a1).requires_grad_(True)
        t2 = torch.from_numpy(np_a2).requires_grad_(True)
        t3 = torch.from_numpy(np_a3).requires_grad_(True)
        t4 = t1 @ t2.T + t3
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()
        assert_array(nx_a4, t4)
        assert_array(nx_a1.grad, t1.grad)
        assert_array(nx_a2.grad, t2.grad)
        assert_array(nx_a3.grad, t3.grad)
