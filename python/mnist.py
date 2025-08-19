from numx.core import Array, from_numpy, f32
import numx.nn as nn
import numx.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Callable
import torch
from tqdm import tqdm
from memory_profiler import profile
import matplotlib.pyplot as plt


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)
        for param in self.linear1.parameters():
            self.add_parameter(param)
        for param in self.linear2.parameters():
            self.add_parameter(param)

    def forward(self, x: Array) -> Array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def load_mnist() -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = 64
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_validation_dataset = datasets.MNIST("./mnist/train", train=True, transform=transform, download=True)
    train_dataset, validation_dataset = random_split(train_validation_dataset, lengths=(0.8, 0.2))
    test_dataset = datasets.MNIST("./mnist/test", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader


# @profile
def try_train_mnist(
    loader: DataLoader, model: MnistModel, loss_fn, optimizer: optim.Optimizer, iterations=5
) -> list[float]:
    losses = []
    for i, (torch_input, torch_label) in tqdm(enumerate(loader), total=iterations):
        if i == iterations:
            break
        np_input = torch_input.numpy()
        np_label = torch_label.numpy().astype(np.int32)
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        logits = model(nx_input)
        loss = loss_fn(logits, nx_label)
        losses.append(loss.item())
        loss.backward()
        optimizer.update(model.parameters())
    return losses


def train_mnist_epoch(train_loader: DataLoader, model: MnistModel, loss_fn, optimizer: optim.Optimizer) -> float:
    mean_loss = 0
    for torch_input, torch_label in tqdm(train_loader):
        np_input = torch_input.numpy()
        np_label = torch_label.numpy().astype(np.int32)
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        logits: Array = model(nx_input)
        loss: Array = loss_fn(logits, nx_label)
        mean_loss += loss.item()
        loss.backward()
        optimizer.update(model.parameters())
    mean_loss /= len(train_loader)
    return mean_loss


def test_mnist(test_loader: DataLoader, model: MnistModel, loss_fn) -> tuple[float, float]:
    mean_loss = 0
    accuracy = 0.0
    for torch_input, torch_label in tqdm(test_loader):
        np_input = torch_input.numpy()
        np_label = torch_label.numpy().astype(np.int32)
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        logits: Array = model(nx_input)
        probs = nn.softmax(logits).argmax([-1])
        loss: Array = loss_fn(logits, nx_label)
        mean_loss += loss.item()
        cmp = (probs == nx_label.unsqueeze()).astype(f32)
        accuracy += cmp.sum().item() / len(cmp)
    mean_loss /= len(test_loader)
    accuracy /= len(test_loader)
    return mean_loss, accuracy


def train_mnist(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model: MnistModel,
    loss_fn,
    optimizer: optim.Optimizer,
    epochs=3,
) -> tuple[list[float], list[float]]:
    train_losses, validation_losses = [], []
    for i in range(epochs):
        print(f"Epoch {i + 1}:")
        mean_loss = train_mnist_epoch(train_loader, model, loss_fn, optimizer)
        train_losses.append(mean_loss)
        print(f"Training loss: {mean_loss}")
        mean_loss, accuracy = test_mnist(validation_loader, model, loss_fn)
        validation_losses.append(mean_loss)
        print(f"Validation loss: {mean_loss}")
        print(f"Validation accuracy: {accuracy}")
    return train_losses, validation_losses


def plot_train_and_validation_losses(train_losses, validation_losses):
    plt.plot(np.arange(0, len(train_losses)), np.array(train_losses), "b-", label="Train loss")
    plt.plot(np.arange(0, len(validation_losses)), np.array(validation_losses), "r-", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
