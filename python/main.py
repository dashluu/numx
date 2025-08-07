from numx.core import Array, f32, b8, i32, from_numpy, normal
from numx.nn import Linear
from numx.profiler import enable_memory_profile, save_memory_profile
import numpy as np
import torch


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
