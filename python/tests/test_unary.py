from __future__ import annotations
from numx.core import Array, from_numpy
from numx.profiler import enable_memory_profile
import numpy as np


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


class TestUnary:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    def unary_no_broadcast(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np_a1 = gen_fn(shape)
        nx_a1 = from_numpy(np_a1)
        nx_a2: Array = op1(nx_a1)
        np_a2 = nx_a2.numpy()
        np_a3: np.ndarray = op2(np_a1)
        assert tuple(nx_a2.view) == np_a3.shape
        assert np.allclose(np_a2, np_a3, atol=1e-3, rtol=0)

    def unary_with_slicing(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name} with slicing:")

        # Test cases with different slicing patterns
        test_cases = [
            # [shape, slices] -> creates non-contiguous tensors
            ([4, 4], (slice(None, None, 2), slice(None))),  # Skip every other row
            ([4, 6], (slice(None), slice(None, None, 2))),  # Skip every other column
            ([4, 4, 4], (slice(None), slice(1, 3), slice(None))),  # Middle slice
            ([6, 6], (slice(None, None, 3), slice(1, None, 2))),  # Complex slicing
        ]

        for shape, slices in test_cases:
            print(f"\nTesting shape: {shape}, slices: {slices}")
            np_a1 = gen_fn(shape)
            nx_a1 = from_numpy(np_a1)
            # Create non-contiguous array using slicing
            nx_a2 = nx_a1[slices]
            nx_a3: Array = op1(nx_a2)  # Apply unary operation
            # Compare with NumPy
            np_a2 = np_a1[slices]  # Apply same slicing
            np_a3: np.ndarray = op2(np_a2)  # Apply same operation
            np_a4 = nx_a3.numpy().reshape(np_a3.shape)
            assert np.allclose(np_a4, np_a3, atol=1e-3, rtol=0)
            assert tuple(nx_a3.view) == np_a3.shape

    def unary_inplace(self, name: str, op1, op2, gen_fn=randn):
        print(f"{name} inplace:")

        # Test different shapes
        test_cases = [
            [5],  # 1D
            [2, 3],  # 2D
            [2, 3, 4],  # 3D
            [1, 2, 3, 4],  # 4D with leading 1
            [5, 1, 4],  # 3D with middle 1
        ]

        for shape in test_cases:
            print(f"\nTesting shape: {shape}")
            # Generate input
            np_a1: np.ndarray = gen_fn(shape)
            np_a2 = np_a1.copy()
            # Create array
            nx_a1 = from_numpy(np_a1)
            # Apply inplace operation
            nx_a2: Array = op1(nx_a1)
            # Compare with NumPy
            np_a3 = op2(np_a2)
            nx_a3 = nx_a2.numpy().reshape(shape)
            assert np.allclose(nx_a3, np_a3, atol=1e-3, rtol=0)

    def test_exp(self):
        self.unary_no_broadcast("exp", Array.exp, np.exp)

    def test_neg(self):
        self.unary_no_broadcast("neg", Array.neg, np.negative)

    def test_log(self):
        self.unary_no_broadcast("log", Array.log, np.log, gen_fn=positive_randn)

    def test_recip(self):
        self.unary_no_broadcast("recip", Array.recip, np.reciprocal, gen_fn=nonzero_randn)

    def test_exp_with_slicing(self):
        self.unary_with_slicing("exp", Array.exp, np.exp)

    def test_neg_with_slicing(self):
        self.unary_with_slicing("neg", Array.neg, np.negative)

    def test_log_with_slicing(self):
        self.unary_with_slicing("log", Array.log, np.log, gen_fn=positive_randn)

    def test_exp_inplace(self):
        def exp_inplace(x: Array):
            return x.exp(in_place=True)

        self.unary_inplace("exp", exp_inplace, np.exp)

    def test_sqrt_inplace(self):
        def sqrt_inplace(x: Array):
            return x.sqrt(in_place=True)

        self.unary_inplace("sqrt", sqrt_inplace, np.sqrt, gen_fn=positive_randn)

    def test_neg_inplace(self):
        def neg_inplace(x: Array):
            return x.neg(in_place=True)

        self.unary_inplace("neg", neg_inplace, np.negative)

    def test_recip_inplace(self):
        def recip_inplace(x: Array):
            return x.recip(in_place=True)

        self.unary_inplace("recip", recip_inplace, np.reciprocal, gen_fn=nonzero_randn)

    def test_log_inplace(self):
        def log_inplace(x: Array):
            return x.log(in_place=True)

        self.unary_inplace("log", log_inplace, np.log, gen_fn=positive_randn)
