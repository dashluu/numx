import numx.core


class Module:
    def __init__(self) -> None:
        """Base module"""

    def forward(self, x: numx.core.Array) -> numx.core.Array:
        """Forward module"""

    def __call__(self, x: numx.core.Array) -> numx.core.Array:
        """Forward module"""

    def parameters(self) -> list[numx.core.Array]:
        """Get module parameters"""

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Linear layer"""

    @property
    def weight(self) -> numx.core.Array:
        """Get linear layer's weight"""

    @property
    def bias(self) -> numx.core.Array | None:
        """Get linear layer's bias"""

def linear(x: numx.core.Array, weight: numx.core.Array) -> numx.core.Array:
    """Functional linear without bias"""

def linear_with_bias(x: numx.core.Array, weight: numx.core.Array, bias: numx.core.Array) -> numx.core.Array:
    """Functional linear with bias"""

def relu(x: numx.core.Array) -> numx.core.Array:
    """ReLU activation function"""

def onehot(x: numx.core.Array, num_classes: int = -1) -> numx.core.Array:
    """One-hot encode input array"""

def cross_entropy_loss(x: numx.core.Array, y: numx.core.Array) -> numx.core.Array:
    """Compute cross-entropy loss between input x and target y"""
