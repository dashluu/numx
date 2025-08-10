from numx.core import Array


class Optimizer:
    def __init__(self, lr):
        self._lr = lr
        self._params: list[Array] = []
        self._grads: list[Array] = []

    def update(self, arrays: list[Array]):
        self._params.clear()
        self._grads.clear()
        for array in arrays:
            grad = array.grad
            if grad is None:
                pass
            self._params.append(array.detach())
            self._grads.append(grad.detach())
        self.forward()
        for param in self._params:
            param.eval()

    def forward(self):
        raise NotImplementedError()


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 1e-3):
        super().__init__(lr)

    def forward(self):
        for i in range(len(self._params)):
            self._params[i] -= self._lr * self._grads[i]
