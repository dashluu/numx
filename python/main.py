from numx.core import Array, f32, b8, i32, from_numpy, full, zeros, ones
from numx.random import normal, randint, uniform, randbool
from numx.nn import Linear, cross_entropy_loss
from numx.optim import GradientDescent
from numx.profiler import enable_memory_profile, save_memory_profile
from mnist import MnistModel, load_mnist, test_mnist, train_mnist, plot_train_and_validation_losses
import numpy as np
import torch
import time
from tqdm import tqdm
import mlx.core as mx
import matplotlib.pyplot as plt


def run_random():
    a = randbool([2, 3, 4])
    print(a)
    b = randint([10000], low=10, high=100)
    plt.hist(b.numpy().flatten(), bins=90, edgecolor="black")
    plt.title("Histogram of Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def run_train_mnist():
    train_loader, validation_loader, test_loader = load_mnist()
    model = MnistModel()
    loss_fn = cross_entropy_loss
    optimizer = GradientDescent(lr=1e-3)
    start_time = time.perf_counter()
    train_losses, validation_losses = train_mnist(train_loader, validation_loader, model, loss_fn, optimizer)
    end_time = time.perf_counter()
    print(f"Training time taken: {end_time - start_time:0.4f} seconds")
    test_loss, test_accuracy = test_mnist(test_loader, model, loss_fn)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
    plot_train_and_validation_losses(train_losses, validation_losses)


if __name__ == "__main__":
    enable_memory_profile()
    # run_train_mnist()
    run_random()
    save_memory_profile("memory_profile.json")
