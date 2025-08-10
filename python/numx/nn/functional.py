from numx.core import Array, i32, arange


def relu(x: Array):
    return x.maximum(0)


def linear(x: Array, weight: Array):
    return x.matmul(weight.transpose(-2, -1))


def linear_with_bias(x: Array, weight: Array, bias: Array):
    return linear(x, weight) + bias


def onehot(x: Array, num_classes: int):
    if not x.dtype.is_int:
        raise ValueError(f"Array {x.id} is not of type int.")
    if num_classes <= 0:
        num_classes = x.max().item() + 1  # type: ignore
    cls = arange({num_classes}, 0, 1, i32)
    return (x.unsqueeze() == cls).astype(i32)


def cross_entropy_loss(x: Array, y: Array):
    max = x.max([-1])
    exp = (x - max).exp()
    sum_exp = exp.sum([-1])
    log_sum_exp = sum_exp.log() + max
    target_onehot = onehot(y, 0).astype(x.dtype)
    loss = -(target_onehot * x).sum([-1]) + log_sum_exp
    return loss.mean()
