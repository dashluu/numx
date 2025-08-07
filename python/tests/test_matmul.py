import numpy as np
from numx.core import Array, from_numpy
from numx.profiler import enable_memory_profile


class TestMatmul:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    def test_2d_matmul(self):
        """Test matrix multiplication for 2D arrays"""
        print("matmul 2d:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([2, 3], [3, 4]),  # Basic matrix multiplication
            ([1, 4], [4, 5]),  # Single row matrix
            ([3, 2], [2, 1]),  # Result is a column matrix
            ([5, 5], [5, 5]),  # Square matrices
            ([1, 1], [1, 1]),  # 1x1 matrices
        ]

        for shape1, shape2 in test_cases:
            print(f"\nTesting shapes: {shape1} @ {shape2}")
            np_a1 = np.random.randn(*shape1).astype(np.float32)
            np_a2 = np.random.randn(*shape2).astype(np.float32)
            nx_a1 = from_numpy(np_a1)
            nx_a2 = from_numpy(np_a2)
            nx_a3 = nx_a1 @ nx_a2
            np_a3 = np_a1 @ np_a2
            assert tuple(nx_a3.view) == np_a3.shape
            assert np.allclose(nx_a3.numpy(), np_a3, atol=1e-3, rtol=0)

    def test_3d_matmul(self):
        """Test matrix multiplication for 3D arrays (batched matmul)"""
        print("\nTesting 3D matrix multiplication:")

        # Test cases: [(shape1, shape2, description)]
        test_cases = [
            # Basic batch matmul
            ([4, 2, 3], [4, 3, 4], "Standard batch size"),
            ([1, 2, 3], [1, 3, 4], "Single batch"),
            ([10, 3, 3], [10, 3, 3], "Square matrices batch"),
            # Broadcasting cases
            ([1, 2, 3], [5, 3, 4], "Broadcast first dim"),
            ([5, 2, 3], [1, 3, 4], "Broadcast second dim"),
            ([7, 1, 3], [7, 3, 5], "Batch with singular dimension"),
            # Edge cases
            ([3, 1, 4], [3, 4, 1], "Result has singular dimension"),
            ([2, 5, 1], [2, 1, 3], "Inner dimension is 1"),
            ([1, 1, 1], [1, 1, 1], "All dimensions are 1"),
        ]

        for shape1, shape2, desc in test_cases:
            print(f"\nTesting {desc}:")
            print(f"Shapes: {shape1} @ {shape2}")
            np_a1 = np.random.randn(*shape1).astype(np.float32)
            np_a2 = np.random.randn(*shape2).astype(np.float32)
            nx_a1 = from_numpy(np_a1)
            nx_a2 = from_numpy(np_a2)
            nx_a3 = nx_a1 @ nx_a2
            np_a3 = np_a1 @ np_a2
            assert tuple(nx_a3.view) == np_a3.shape
            assert np.allclose(nx_a3.numpy(), np_a3, atol=1e-3, rtol=0)

    def test_multidim_matmul(self):
        """Test multi-dimensional matrix multiplication"""
        print("\nTesting multi-dimensional matrix multiplication:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([1, 5, 4, 2, 3], [5, 1, 4, 3, 4]),
            ([1, 1, 2, 3], [5, 2, 3, 4]),
            ([1, 3, 7, 3, 17], [10, 3, 1, 17, 6]),
            ([13, 4, 2, 9, 1], [1, 1, 2, 1, 8]),
        ]

        for shape1, shape2 in test_cases:
            print(f"Shapes: {shape1} @ {shape2}")
            np_a1 = np.random.randn(*shape1).astype(np.float32)
            np_a2 = np.random.randn(*shape2).astype(np.float32)
            nx_a1 = from_numpy(np_a1)
            nx_a2 = from_numpy(np_a2)
            nx_a3 = nx_a1 @ nx_a2
            np_a3 = np_a1 @ np_a2
            assert tuple(nx_a3.view) == np_a3.shape
            assert np.allclose(nx_a3.numpy(), np_a3, atol=1e-3, rtol=0)
