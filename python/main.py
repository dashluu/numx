from numx.core import Array, f32, b8, i32, from_numpy, normal, full
from numx.nn import Linear
from numx.optim import GradientDescent
from numx.profiler import enable_memory_profile, save_memory_profile
import numpy as np
import torch


def run_optimizer():
    nx_a1 = full([2, 3, 4], 5)
    nx_a2 = full([1, 3, 4], 0.2)
    nx_a3 = nx_a1 + nx_a2
    nx_a4 = nx_a3.sum()
    nx_a4.backward()
    print(nx_a4)
    print(nx_a1)
    print(nx_a1.grad)
    print(nx_a2)
    print(nx_a2.grad)
    nx_optimizer = GradientDescent(lr=1)
    nx_optimizer.update([nx_a1, nx_a2])
    print(nx_a1)
    print(nx_a2)


def run_linear():
    nx_a1 = normal([3, 10])
    nx_model = Linear(10, 4)
    nx_a2 = nx_model(nx_a1)
    print(nx_a1)
    print(nx_a2)


if __name__ == "__main__":
    enable_memory_profile()
    run_linear()
    save_memory_profile("memory_profile.json")
