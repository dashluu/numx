from collections.abc import Sequence

import numx.core


def uniform(view: Sequence[int], low: object = 0.0, high: object = 1.0, dtype: numx.core.Dtype = ..., device: str = 'mps:0') -> numx.core.Array:
    """Create a new array with random values from a uniform distribution"""

def normal(view: Sequence[int], mean: object = 0.0, std: object = 1.0, dtype: numx.core.Dtype = ..., device: str = 'mps:0') -> numx.core.Array:
    """Create a new array with random values from a normal distribution"""

def kaiming_uniform(view: Sequence[int], dtype: numx.core.Dtype = ..., device: str = 'mps:0') -> numx.core.Array:
    """
    Create a new array with random values from a Kaiming uniform distribution
    """
