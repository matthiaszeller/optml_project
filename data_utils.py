from pathlib import Path
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(normalize=True, path='data'):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    train = torchvision.datasets.MNIST(str(path), download=True, train=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST(str(path), train=False, transform=torchvision.transforms.ToTensor())

    xtrain = train.data.to(torch.float32)
    xtest = test.data.to(torch.float32)

    if normalize:
        xtrain = xtrain.view(-1, 28 * 28)
        xtest = xtest.view(-1, 28 * 28)
        mu, std = xtrain.mean(0), xtrain.std(0)

        xtrain -= mu
        xtest -= mu
        xtrain[:, std > 0] /= std[std > 0]
        xtest[:, std > 0] /= std[std > 0]

        xtrain = xtrain.view(-1, 28, 28)
        xtest = xtest.view(-1, 28, 28)

    train.data = xtrain
    test.data = xtest

    return train, test


def build_data_loaders(train_dataset, test_dataset, batch_size) -> Tuple[DataLoader, DataLoader]:
    # Add an axis to have one channel
    train_dataset.data = train_dataset.data[:, None, :]
    test_dataset.data = test_dataset.data[:, None, :]

    train_loader = DataLoader(TensorDataset(train_dataset.data, train_dataset.targets), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_dataset.data, test_dataset.targets), batch_size=batch_size)

    return train_loader, test_loader
