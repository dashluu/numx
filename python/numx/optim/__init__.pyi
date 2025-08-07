from collections.abc import Sequence

import numx.core


class Optimizer:
    def __init__(self, lr: float = 0.001) -> None:
        """Base optimizer"""

    def forward(self) -> None:
        """Parameter update function"""

    def update(self, arrays: Sequence[numx.core.Array]) -> None:
        """Update module parameters"""

class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        """Gradient Descent optimizer"""
