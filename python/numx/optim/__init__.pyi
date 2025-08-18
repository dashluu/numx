from collections.abc import Sequence

import numx.nn


class Optimizer:
    def __init__(self, lr: float = 0.001) -> None:
        """Base optimizer"""

    def forward(self) -> None:
        """Parameters update function"""

    def update(self, params: Sequence[numx.nn.Parameter]) -> None:
        """Update module parameters"""

class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        """Gradient Descent optimizer"""
