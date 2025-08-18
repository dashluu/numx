from collections.abc import Sequence
from typing import overload

import numx.core


def linear(x: numx.core.Array, weight: numx.core.Array) -> numx.core.Array:
    """Functional linear without bias"""

def linear_with_bias(x: numx.core.Array, weight: numx.core.Array, bias: numx.core.Array) -> numx.core.Array:
    """Functional linear with bias"""

def relu(x: numx.core.Array) -> numx.core.Array:
    """ReLU activation function"""

def onehot(x: numx.core.Array, num_classes: int = -1) -> numx.core.Array:
    """One-hot encode input array"""

def softmax(x: numx.core.Array, dim: int = -1) -> numx.core.Array:
    """Compute softmax for input array"""

def cross_entropy_loss(x: numx.core.Array, y: numx.core.Array) -> numx.core.Array:
    """Compute cross-entropy loss between input x and target y"""

class Parameter(numx.core.Array):
    @overload
    def __init__(self, view: Sequence[int], dtype: numx.core.Dtype = ..., device: str = 'mps:0') -> None:
        """Module parameter"""

    @overload
    def __init__(self, array: numx.core.Array) -> None: ...

class Module:
    def __init__(self) -> None:
        """Base module"""

    def add_parameter(self, param: Parameter) -> None:
        """Add parameter to module"""

    def parameters(self) -> list[Parameter]:
        """Get module parameters"""

    def forward(self, x: numx.core.Array) -> numx.core.Array:
        """Forward pass through module"""

    def __call__(self, x: numx.core.Array) -> numx.core.Array:
        """Forward pass through module"""

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Linear layer"""

    @property
    def weight(self) -> Parameter:
        """Get linear layer weight"""

    @property
    def bias(self) -> Parameter:
        """Get linear layer bias"""
