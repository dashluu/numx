from numx.core import Array, zeros, full, full_like, zeros_like, ones_like, ones, arange
from numx.profiler import enable_memory_profile
import numpy as np


class TestInitializers:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    def test_zeros(self):
        print("\nTesting zeros:")

        test_shapes = [
            [5],  # 1D
            [2, 3],  # 2D
            [2, 3, 4],  # 3D
            [1, 2, 3, 4],  # 4D with leading 1
            [5, 1, 4],  # 3D with middle 1
        ]

        for shape in test_shapes:
            print(f"Testing shape: {shape}")
            nx_a1 = zeros(shape)
            np_a1 = np.zeros(shape, dtype=np.float32)
            assert tuple(nx_a1.view) == np_a1.shape
            assert np.allclose(nx_a1.numpy(), np_a1)

    def test_full(self):
        print("\nTesting full:")

        test_cases = [
            ([2, 3], 5),  # Integer fill
            ([3, 4], -2),  # Negative integer
            ([2, 2], 3.14),  # Float fill
            ([2, 3], -0.5),  # Negative float
            ([3, 3], 0),  # Zero
        ]

        for shape, value in test_cases:
            print(f"Testing shape: {shape}, value: {value}")
            nx_a1 = full(shape, value)
            np_a1 = np.full(shape, value, dtype=np.float32)
            assert tuple(nx_a1.view) == np_a1.shape
            assert np.allclose(nx_a1.numpy(), np_a1)

    def test_like_methods(self):
        print("\nTesting *_like methods:")

        # Create a template array
        shape = [2, 3, 4]
        nx_a1 = ones(shape)
        np_a1 = np.ones(shape, dtype=np.float32)

        # Test cases: (method, numpy_equivalent, fill_value)
        test_cases = [
            (zeros_like, np.zeros_like, 0),
            (ones_like, np.ones_like, 1),
            (lambda x: full_like(x, 5), lambda x: np.full_like(x, 5), 5),
            (lambda x: full_like(x, -2.5), lambda x: np.full_like(x, -2.5), -2.5),
        ]

        for nx_method, np_method, value in test_cases:
            print(f"Testing {nx_method.__name__} with value {value}")
            nx_a2: Array = nx_method(nx_a1)
            np_a2: np.ndarray = np_method(np_a1)
            assert tuple(nx_a2.view) == np_a2.shape
            assert np.allclose(nx_a2.numpy(), np_a2)

    def test_arange(self):
        print("\nTesting arange:")

        test_cases = [
            # [shape, start, step]
            ([5], 0, 1),  # Basic range
            ([5], 1, 2),  # Custom step
            ([5], -2, 1),  # Negative start
            ([5], 0, -1),  # Negative step
            ([10], 5, -2),  # Float step
            ([8], -4, 3),  # Float step with negative start
            ([1], 0, 1),  # Single element
            ([2, 4], 1, 3),  # Multidimensional shape
            ([5, 11, 7], -5, 5),  # Multidimensional shape with negative start
            ([5, 11, 7], 10, -7),  # Multidimensional shape with negative step
            ([6, 13, 17], -11, -13),  # Multidimensional shape with negative start and step
        ]

        for shape, start, step in test_cases:
            print(f"Testing shape: {shape}, start: {start}, step: {step}")
            nx_a1 = arange(shape, start, step)
            # Calculate total size from all dimensions
            size = np.prod(shape)
            # Create base numpy array and reshape to match target shape
            np_a1 = np.arange(start, start + size * step, step, dtype=np.float32).reshape(shape)
            assert tuple(nx_a1.view) == np_a1.shape
            assert np.allclose(nx_a1.numpy(), np_a1, atol=1e-6)
