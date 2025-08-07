from __future__ import annotations
from numx.core import Array, from_numpy
from numx.profiler import enable_memory_profile
import numpy as np
import operator


def randn(shape) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32)


def nonzero_randn(shape) -> np.ndarray:
    array = randn(shape)
    # Replace zeros with small random values
    zero_mask = array == 0
    array[zero_mask] = np.random.uniform(0.1, 1.0, size=np.count_nonzero(zero_mask))
    return array


def positive_randn(shape) -> np.ndarray:
    return np.abs(nonzero_randn(shape))


class TestBinary:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    def binary_no_broadcast(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np_a1 = gen_fn(shape)
        np_a2 = gen_fn(shape)
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)
        nx_a3: Array = op1(nx_a1, nx_a2)
        np_a3: np.ndarray = nx_a3.numpy()
        np_a4: np.ndarray = op2(np_a1, np_a2)
        assert tuple(nx_a3.view) == np_a4.shape
        assert np.allclose(np_a3, np_a4, atol=1e-3, rtol=0)

    def binary_with_broadcast(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name} with broadcast:")
        # Test cases with different broadcasting scenarios
        test_cases = [
            # [shape1, shape2, result_shape]
            ([2, 1, 4], [3, 4], [2, 3, 4]),  # Left broadcast
            ([1, 5], [2, 1, 5], [2, 1, 5]),  # Right broadcast
            ([3, 1, 1], [1, 4, 5], [3, 4, 5]),  # Both broadcast
            ([1], [2, 3, 4], [2, 3, 4]),  # Scalar to array
            ([2, 3, 4], [1], [2, 3, 4]),  # Array to scalar
            ([3, 1, 19, 1, 1], [1, 47, 19, 63, 1], [3, 47, 19, 63, 1]),
            ([1, 2], [1, 47, 19, 63, 1], [1, 47, 19, 63, 2]),
        ]
        for shape1, shape2, expected_shape in test_cases:
            print(f"\nTesting shapes: {shape1}, {shape2} -> {expected_shape}")
            np_a1 = gen_fn(shape1)
            np_a2 = gen_fn(shape2)
            nx_a1 = from_numpy(np_a1)
            nx_a2 = from_numpy(np_a2)
            nx_a3: Array = op1(nx_a1, nx_a2)
            np_a3: np.ndarray = nx_a3.numpy()
            np_a4: np.ndarray = op2(np_a1, np_a2)
            assert tuple(nx_a3.view) == np_a4.shape
            assert np.allclose(np_a3, np_a4, atol=1e-3, rtol=0)

    def binary_inplace(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name} inplace:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]

        # Generate inputs
        np_a1: np.ndarray = gen_fn(shape)
        np_a2: np.ndarray = gen_fn(shape)
        np_a3 = np_a1.copy()  # Keep copy for numpy comparison

        # Create arrays
        nx_a1 = from_numpy(np_a1)
        nx_a2 = from_numpy(np_a2)

        # Apply inplace operation
        nx_a1: Array = op1(nx_a1, nx_a2)  # nx_a1 += nx_a2, etc.
        nx_a1: Array = op1(nx_a1, nx_a2)  # Second time to make sure it's updated.

        # Compare with NumPy
        np_a3: np.ndarray = op2(np_a3, np_a2)  # np_a1_copy += np_a2, etc.
        np_a3: np.ndarray = op2(np_a3, np_a2)  # Second time
        assert tuple(nx_a1.view) == np_a3.shape
        assert np.allclose(nx_a1.numpy(), np_a3, atol=1e-3, rtol=0)

    def binary_inplace_broadcast(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name} inplace broadcast:")

        test_cases = [
            # [lshape, rshape] -> result shape will be lshape
            ([2, 3, 4], [4]),  # Broadcast scalar to 3D
            ([3, 4, 5], [1, 5]),  # Broadcast from 2D to 3D
            ([2, 4, 6], [4, 1]),  # Broadcast with ones
            ([5, 5, 5], [1, 5, 1]),  # Broadcast with ones in multiple dims
            ([4, 3, 2], [3, 1]),  # Partial broadcast with ones
            ([3, 47, 19, 63, 1], [1, 1, 19, 63, 1]),
            ([1, 47, 19, 63, 2], [1, 2]),
        ]

        for lshape, rshape in test_cases:
            print(f"\nTesting: {lshape} @= {rshape}")

            # Generate inputs
            np_a1: np.ndarray = gen_fn(lshape)
            np_a2: np.ndarray = gen_fn(rshape)
            np_a3 = np_a1.copy()

            # Create arrays
            nx_a1 = from_numpy(np_a1)
            nx_a2 = from_numpy(np_a2)

            # Apply inplace operation
            nx_a1: Array = op1(nx_a1, nx_a2)
            # Compare with NumPy
            np_a3: np.ndarray = op2(np_a3, np_a2)
            assert tuple(nx_a1.view) == np_a3.shape
            assert np.allclose(nx_a1.numpy(), np_a3, atol=1e-3, rtol=0)

    def test_add(self):
        self.binary_no_broadcast("add", operator.add, operator.add)

    def test_sub(self):
        self.binary_no_broadcast("sub", operator.sub, operator.sub)

    def test_mul(self):
        self.binary_no_broadcast("mul", operator.mul, operator.mul)

    def test_div(self):
        self.binary_no_broadcast("div", operator.truediv, operator.truediv)

    def test_minimum(self):
        self.binary_no_broadcast("minimum", lambda x, y: x.minimum(y), lambda x, y: np.minimum(x, y))

    def test_maximum(self):
        self.binary_no_broadcast("maximum", lambda x, y: x.maximum(y), lambda x, y: np.maximum(x, y))

    def test_add_broadcast(self):
        self.binary_with_broadcast("add", operator.add, operator.add)

    def test_sub_broadcast(self):
        self.binary_with_broadcast("sub", operator.sub, operator.sub)

    def test_mul_broadcast(self):
        self.binary_with_broadcast("mul", operator.mul, operator.mul)

    def test_div_broadcast(self):
        self.binary_with_broadcast("div", operator.truediv, operator.truediv)

    def test_add_inplace(self):
        self.binary_inplace("add", operator.iadd, operator.iadd)

    def test_sub_inplace(self):
        self.binary_inplace("sub", operator.isub, operator.isub)

    def test_mul_inplace(self):
        self.binary_inplace("mul", operator.imul, operator.imul)

    def test_div_inplace(self):
        self.binary_inplace("div", operator.itruediv, operator.itruediv)

    def test_add_inplace_broadcast(self):
        self.binary_inplace_broadcast("add", operator.iadd, operator.iadd)

    def test_sub_inplace_broadcast(self):
        self.binary_inplace_broadcast("sub", operator.isub, operator.isub)

    def test_mul_inplace_broadcast(self):
        self.binary_inplace_broadcast("mul", operator.imul, operator.imul)

    def test_div_inplace_broadcast(self):
        self.binary_inplace_broadcast("div", operator.itruediv, operator.itruediv)
