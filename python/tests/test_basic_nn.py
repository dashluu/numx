from __future__ import annotations
from numx.core import Array, from_numpy
import numx.nn as nn
import numx.optim as optim
from numx.profiler import enable_memory_profile
from mnist import MnistModel
import numpy as np
import torch


class TestBasicNN:
    @classmethod
    def setup_class(cls):
        enable_memory_profile()

    @staticmethod
    def elmwise_assert(array: torch.Tensor, tensor: torch.Tensor, atol=1e-3, rtol=0):
        try:
            torch.testing.assert_close(array, tensor, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"Tensors not close: {e}")
            # Find mismatched indices
            mask = torch.isclose(array, tensor, atol=atol, rtol=rtol)
            if array.ndim > 1:
                entry = torch.where(~mask)
                print("Mismatched values:")
                print("array:", array[entry].double())
                print("tensor:", tensor[entry].double())
            else:
                entry = torch.where(~mask)[0]
                print("Mismatched values:")
                print("array:", array[entry].double())
                print("tensor:", tensor[entry].double())
            raise  # Re-raise the assertion error

    def test_linear(self):
        np_a1 = np.random.randn(64, 784).astype(np.float32)
        nx_a1 = from_numpy(np_a1)
        t1 = torch.from_numpy(np_a1)
        linear = nn.Linear(784, 10)
        nx_weight = linear.weight
        nx_bias = linear.bias
        torch_weight: torch.Tensor = nx_weight.torch()
        torch_weight.requires_grad_(True)
        torch_weight.retain_grad()
        torch_bias: torch.Tensor = nx_bias.torch()
        torch_bias.requires_grad_(True)
        torch_bias.retain_grad()
        nx_a2 = linear(nx_a1).sum()
        t2 = (t1 @ torch_weight.T + torch_bias).sum()
        nx_a2.backward()
        t2.backward()
        assert torch.allclose(nx_a2.torch(), t2, atol=1e-3, rtol=0)

    def test_relu(self):
        np_a1 = np.random.randn(64, 10).astype(np.float32)
        nx_a1 = from_numpy(np_a1)
        t1 = torch.from_numpy(np_a1)
        t1.requires_grad_(True)
        nx_a2 = nn.relu(nx_a1)
        nx_a3 = nx_a2.sum()
        t2 = torch.relu(t1)
        t3 = t2.sum()
        nx_a3.backward()
        t3.backward()
        assert torch.allclose(nx_a2.torch(), t2, atol=1e-3, rtol=0)
        assert torch.allclose(nx_a1.grad.torch(), t1.grad, atol=1e-3, rtol=0)

    def test_onehot(self):
        np_a1 = np.random.randint(0, 10, (64,), dtype=np.int32)
        nx_a1 = from_numpy(np_a1)
        t1 = torch.from_numpy(np_a1).type(torch.int64)
        nx_a2 = nn.onehot(nx_a1)
        t2 = torch.nn.functional.one_hot(t1, num_classes=10).type(torch.int32)
        assert torch.allclose(nx_a2.torch(), t2, atol=1e-3, rtol=0)

    def test_cross_entropy(self):
        np_input = np.random.randn(64, 10).astype(np.float32)
        np_label = np.random.randint(0, 10, (64,), dtype=np.int32)
        nx_input = from_numpy(np_input)
        torch_input = torch.from_numpy(np_input)
        torch_input.requires_grad_(True)
        nx_label = from_numpy(np_label)
        torch_label = torch.from_numpy(np_label).type(torch.int64)
        nx_loss = nn.cross_entropy_loss(nx_input, nx_label)
        torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_input, torch_label)
        nx_loss.backward()
        torch_loss.backward()
        assert torch.allclose(nx_loss.torch(), torch_loss, atol=1e-3, rtol=0)
        assert torch.allclose(nx_input.grad.torch(), torch_input.grad, atol=1e-3, rtol=0)

    def test_single_pass(self):
        # Input data
        np_input = np.random.randn(64, 784).astype(np.float32)
        np_label = np.random.randint(0, 10, (64,), dtype=np.int32)

        # Array implementation
        nx_input = from_numpy(np_input)
        nx_label = from_numpy(np_label)
        nx_model = MnistModel()

        # PyTorch implementation
        torch_model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        # Share weights between Array and PyTorch
        # First layer
        w1: torch.Tensor = nx_model.linear1.weight.torch()
        b1: torch.Tensor = nx_model.linear1.bias.torch()
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        # Second layer
        w2: torch.Tensor = nx_model.linear2.weight.torch()
        b2: torch.Tensor = nx_model.linear2.bias.torch()
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)

        # Forward pass
        torch_input = torch.from_numpy(np_input)
        torch_label = torch.from_numpy(np_label).type(torch.int64)

        nx_logits = nx_model(nx_input)
        torch_logits = torch_model(torch_input)

        # Loss computation
        nx_loss = nn.cross_entropy_loss(nx_logits, nx_label)
        torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_logits, torch_label)

        # Backward pass
        nx_loss.backward()
        torch_loss.backward()

        # Compare results
        assert torch.allclose(nx_logits.torch(), torch_logits, atol=1e-3, rtol=0)
        assert torch.allclose(nx_loss.torch(), torch_loss, atol=1e-3, rtol=0)

        # Compare gradients
        assert torch.allclose(nx_model.linear1.weight.grad.torch(), torch_model[0].weight.grad, atol=1e-3, rtol=0)
        assert torch.allclose(nx_model.linear1.bias.grad.torch(), torch_model[0].bias.grad, atol=1e-3, rtol=0)
        assert torch.allclose(nx_model.linear2.weight.grad.torch(), torch_model[2].weight.grad, atol=1e-3, rtol=0)
        assert torch.allclose(nx_model.linear2.bias.grad.torch(), torch_model[2].bias.grad, atol=1e-3, rtol=0)

    def test_multipass_with_optimizer(self):
        nx_model = MnistModel()
        torch_model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        w1: torch.Tensor = nx_model.linear1.weight.torch()
        b1: torch.Tensor = nx_model.linear1.bias.torch()
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        w2: torch.Tensor = nx_model.linear2.weight.torch()
        b2: torch.Tensor = nx_model.linear2.bias.torch()
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)
        nx_loss_fn = nn.cross_entropy_loss
        torch_loss_fn = torch.nn.CrossEntropyLoss()
        nx_optimizer = optim.GradientDescent(lr=1)
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1, momentum=0, weight_decay=0)
        passes = 3

        for _ in range(passes):
            # Input data
            np_input = np.random.randn(64, 784).astype(np.float32)
            np_label = np.random.randint(0, 10, (64,), dtype=np.int32)

            # Array implementation
            nx_input = from_numpy(np_input)
            nx_label = from_numpy(np_label)
            # PyTorch implementation
            torch_input = torch.from_numpy(np_input)
            torch_label = torch.from_numpy(np_label).type(torch.int64)

            # Loss computation
            nx_logits = nx_model(nx_input)
            nx_loss = nx_loss_fn(nx_logits, nx_label)
            torch_logits = torch_model(torch_input)
            torch_loss: torch.Tensor = torch_loss_fn(torch_logits, torch_label)
            # print(nx_loss.item(), torch_loss.item())
            # print()

            # Backward pass and parameters update
            nx_loss.backward()
            torch_optimizer.zero_grad()
            torch_loss.backward()
            # Compare losses
            assert torch.allclose(nx_loss.torch(), torch_loss, atol=1e-3, rtol=0)

            print(nx_model.linear1.weight.torch())
            print(torch_model[0].weight)
            print(nx_model.linear1.weight.grad.torch())
            print(torch_model[0].weight.grad)

            # Update parameters
            nx_optimizer.update(nx_model.parameters())
            torch_optimizer.step()

            print(nx_model.linear1.weight.torch())
            print(torch_model[0].weight)

            # Compare updated weights and biases
            TestBasicNN.elmwise_assert(nx_model.linear1.weight.grad.torch(), torch_model[0].weight.grad)
            TestBasicNN.elmwise_assert(nx_model.linear1.bias.grad.torch(), torch_model[0].bias.grad)
            TestBasicNN.elmwise_assert(nx_model.linear2.weight.grad.torch(), torch_model[2].weight.grad)
            TestBasicNN.elmwise_assert(nx_model.linear2.bias.grad.torch(), torch_model[2].bias.grad)
            # TestBasicNN.elmwise_assert(nx_model.linear1.weight.torch(), torch_model[0].weight)
            # TestBasicNN.elmwise_assert(nx_model.linear1.bias.torch(), torch_model[0].bias.data)
            # TestBasicNN.elmwise_assert(nx_model.linear2.weight.torch(), torch_model[2].weight.data)
            # TestBasicNN.elmwise_assert(nx_model.linear2.bias.torch(), torch_model[2].bias.data)
