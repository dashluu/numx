from numx.core import Array, from_numpy
import numx.nn as nn
import numx.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Callable
import torch
from tqdm import tqdm
from memory_profiler import profile


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)

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
    train_valid_dataset = datasets.MNIST("../mnist_ds/train", train=True, transform=transform, download=True)
    train_dataset, valid_dataset = random_split(train_valid_dataset, lengths=(0.8, 0.2))
    test_dataset = datasets.MNIST("../mnist_ds/test", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


# @profile
def test_train_mnist(loader: DataLoader, iterations=2) -> list[float]:
    model = MnistModel()
    loss_fn = nn.cross_entropy_loss
    optimizer = optim.GradientDescent(lr=1e-3)
    losses = []
    for i, (img, label) in enumerate(loader):
        if i == iterations:
            break
        img_array = from_numpy(img.numpy())
        label_array = from_numpy(label.numpy().astype(np.int32))
        logits = model(img_array)
        loss = loss_fn(logits, label_array)
        losses.append(loss.item())
        loss.backward()
        optimizer.update(model.parameters())
    return losses


def train_mnist(loader: DataLoader, epochs=1):
    model = MnistModel()
    loss_fn = nn.cross_entropy_loss
    optimizer = optim.GradientDescent(lr=1e-3)
    losses = []
    for i in range(epochs):
        for img, label in loader:
            img_array = Array.from_numpy(img.numpy())
            label_array = Array.from_numpy(label.numpy().astype(np.int32))
            loss = loss_fn(model(img_array), label_array)
            losses.append(loss.item())
            print(loss.item())
            loss.backward()
            optimizer.update(model.parameters())
        print(f"Epoch {i + 1} loss: {np.mean(losses)}")
