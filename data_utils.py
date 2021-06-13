from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd
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
    train_loader = DataLoader(TensorDataset(train_dataset.data[:, None, :], train_dataset.targets), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_dataset.data[:, None, :], test_dataset.targets), batch_size=batch_size)

    return train_loader, test_loader


def get_height_weight_gender(sub_sample=True, add_outlier=False):
    """Load height-weight-gender data and convert it to the metric system."""
    path_dataset = "data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]
        gender = gender[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender

def load_results(fp:str, drop_cst_param: bool = True):
    """Returns a DataFrame from the tuning results and the columns of the DataFrame corresponding to hyperparameters."""

    with open(fp, 'r') as f:
        res = json.load(f)
        
    df = pd.DataFrame(res)
    hyperparams = df.columns.drop(['loss_train', 'metric_train', 'loss_test', 'metric_test'])
    # Average over folds
    df.loss_train = df.loss_train.apply(lambda e: np.array(e).mean(axis=0))
    df.metric_train = df.metric_train.apply(lambda e: np.array(e).mean(axis=0))
    df.loss_test = df.loss_test.apply(np.mean)
    df['metric_test_std'] = df.metric_test.apply(np.std)
    df.metric_test = df.metric_test.apply(np.mean)
    
    if drop_cst_param:
        hyperparams = hyperparams.drop([e for e in hyperparams if df[e].nunique() <= 1])
    
    return df, hyperparams

def get_best_hyperparams(fp:str):
    """Get best set of hyperparameters for any optimizer."""
    if 'nesterov' not in fp:
        df, params = load_results(fp, drop_cst_param=False)
    else:
        df, params = load_results(fp)

    best_res = {'metric_test' : -float('inf')}
    ind = 0
    for idx, row in df.iterrows():
        if (row.metric_test/ row['metric_test_std']) > best_res['metric_test']:
            best_res['metric_test'] = row.metric_test/ row['metric_test_std']
            ind = int(idx)              

    return df.iloc[ind][params].to_dict()


