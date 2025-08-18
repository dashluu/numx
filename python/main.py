from numx.core import Array, f32, b8, i32, from_numpy, full
from numx.random import normal
from numx.nn import Linear, cross_entropy_loss
from numx.optim import GradientDescent
from numx.profiler import enable_memory_profile, save_memory_profile
from mnist import MnistModel, load_mnist, test_train_mnist, train_mnist, plot_train_and_validation_losses
import numpy as np
import torch
import time
from tqdm import tqdm


def run_basic():
    nx_a1 = full([2, 3, 4], 5)
    nx_a2 = full([1, 3, 4], 0.2)
    nx_a3 = nx_a1 + nx_a2
    print(nx_a3)


def run_basic_loop():
    for i in range(20):
        print(f"Epoch {i + 1}:")
        for _ in tqdm(range(1000)):
            nx_a1 = full([20, 30, 40], 5)
            nx_a2 = full([20, 30, 40], 0.2)
            nx_a3 = nx_a1 + nx_a2
            nx_a3.eval()


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


def run_advanced_linear():
    # Input data
    np_input = np.random.randn(64, 784).astype(np.float32)
    np_label = np.random.randint(0, 10, (64,), dtype=np.int32)
    # Array implementation
    nx_input = from_numpy(np_input)
    nx_label = from_numpy(np_label)
    nx_model = MnistModel()
    # Forward pass
    nx_logits = nx_model(nx_input)
    # Loss computation
    nx_loss = cross_entropy_loss(nx_logits, nx_label)
    # Backward pass
    nx_loss.backward()


def run_multipass_with_optimizer():
    nx_model = MnistModel()
    nx_loss_fn = cross_entropy_loss
    nx_optimizer = GradientDescent(lr=1)
    passes = 5

    for _ in range(passes):
        # Input data
        np_input = np.random.randn(64, 784).astype(np.float32)
        np_label = np.random.randint(0, 10, (64,), dtype=np.int32)
        # Array implementation
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        # Loss computation
        nx_logits = nx_model(nx_input)
        nx_loss = nx_loss_fn(nx_logits, nx_label)
        # Backward pass and parameters update
        nx_loss.backward()
        # Update parameters
        nx_optimizer.update(nx_model.parameters())


def run_train_mnist():
    train_loader, validation_loader, _ = load_mnist()
    start_time = time.perf_counter()
    train_losses, validation_losses = train_mnist(train_loader, validation_loader)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:0.4f} seconds")
    plot_train_and_validation_losses(train_losses, validation_losses)


if __name__ == "__main__":
    # enable_memory_profile()
    run_train_mnist()
    # save_memory_profile("memory_profile.json")
