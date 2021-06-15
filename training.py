import itertools
from time import time
from typing import Iterable, Callable, Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


def accuracy(yhat, y):
    prediction = yhat.argmax(dim=1)
    return (y.eq(prediction)).to(float).mean().item()


def train_epoch(model: Module, dataset: Iterable, optim: Optimizer, loss_fun: Module, metric_fun: Callable, device,
                log_interval: int = 0):
    if metric_fun is None:
        metric_fun = lambda yhat, y: None

    losses = []
    metrics = []
    for batch_id, (x, y) in enumerate(dataset, 1):
        # Move to GPU
        x, y = x.to(device), y.to(device)
        # Reset gradients
        optim.zero_grad()
        # Forward pass
        yhat = model(x)
        # Performance evaluation
        metrics.append(metric_fun(yhat, y))
        # Compute loss
        loss = loss_fun(yhat, y)
        # Backward pass
        loss.backward()
        # Optimization step
        optim.step()

        losses.append(loss.item())
        if log_interval > 0 and batch_id % log_interval == 0:
            print_metric = '' if metric_fun is None else f'\tacc = {metrics[-1]:.4}'
            print(f'batch {batch_id}\tloss = {losses[-1]:.4}{print_metric}')

    return losses, metrics


def training(model: Module, dataset: Iterable, optim: Optimizer, loss_fun: Module, metric_fun: Callable = None,
             epochs: int = 10, device=None, batch_log_interval: int = 100):
    """

    :param model:
    :param dataset:
    :param optim:
    :param loss_fun:
    :param metric_fun:
    :param epochs:
    :param device:
    :param batch_log_interval: -1 means no print per epoch, 0 means print epoch avg only, > 0 is batch interval
    :return:
    """
    if metric_fun is None:
        metric_fun = lambda *args: None
    if device is None:
        device = torch.device('cpu')

    print(f'Launching training on {device}', end=' ', flush=True)
    losses_epoch = []
    metrics_epoch = []
    model.train()
    t = time()
    for epoch in range(epochs):
        losses, metrics = train_epoch(model, dataset, optim, loss_fun, metric_fun, device, batch_log_interval)
        losses_epoch.append(losses)
        metrics_epoch.append(metrics)
        print_metric = '' if metric_fun is None else f'\tavg epoch acc = {np.mean(metrics):.4}'
        if batch_log_interval >= 0:
            print(f'epoch {epoch}\tavg epoch loss = {np.mean(losses):.4}{print_metric}')
        else:
            print('', end='. ', flush=True)

    t = time() - t
    print(f'training took {t:.4} s')

    if metric_fun is None:
        return losses_epoch
    return losses_epoch, metrics_epoch


def training_debug(model: Module, dataset: Iterable, optim: Optimizer, loss_fun: Module,
                   metric_fun: Callable, device, n_steps: int =1):
    losses, metrics = [], []
    for batch, (x, y) in enumerate(dataset, 1):
        print(f'batch number {batch}')
        # Move to GPU
        x, y = x.to(device), y.to(device)
        # Reset gradients
        # print('running optim.zero_grad()')
        optim.zero_grad()
        # Forward pass
        # print('forward pass')
        yhat = model(x)
        # Performance evaluation
        metric = metric_fun(yhat, y)
        print(f'metric = {metric}')
        # Compute loss
        loss = loss_fun(yhat, y)
        print(f'loss = {loss.item()}')
        # Backward pass
        # print('backward pass')
        loss.backward()
        # Optimization step
        print('running optim.step()')
        optim.step()

        metrics.append(metric)
        losses.append(loss.item())
        if batch >= n_steps:
            break

    return losses, metrics


def testing(model: Module, dataset: Iterable, loss_fun: Module, metric_fun: Callable, device):
    losses = []
    metrics = []
    model.eval()
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = loss_fun(yhat, y)
            losses.append(loss.item())
            metrics.append(metric_fun(yhat, y))

    print(f'Avg test loss = {np.mean(losses):.3}\tAvg test acc = {np.mean(metrics):.3}')
    return losses, metrics


def tune_optimizer(model: Module,
                   xtrain: torch.Tensor,
                   ytrain: torch.Tensor,
                   loss_fun: Module,
                   metric_fun: Callable,
                   device,
                   optim_fun: Callable,
                   epochs: int,
                   search_grid: Dict[str, Iterable],
                   train_ratio: float = None,
                   nfolds: int = None,
                   func = training):
    """
    Tune parameters of an optimizer. Evaluate performance either with train/test split, or kfold-CV.

    :param model: a torch module
    :param xtrain: the training data
    :param ytrain: the training targets
    :param loss_fun: loss function
    :param metric_fun: function to assess model performance
    :param device: a torch device to train the model on (either cpu or cuda)
    :param optim_fun: the optimizer class
    :param epochs: number of epochs
    :param search_grid: a dict of parameters to vary
    :param train_ratio: set this if you want train/test splitting instead of KFoldCV
    :param nfolds: number of folds if you want to use KFoldCV instead of train/test split
    :return: list of dictionnaries containing training and test results.
    """
    def split_data(xtrain, ytrain, batch_size, train_ratio, nfolds):
        if train_ratio is not None:
            ids = torch.randperm(xtrain.shape[0])
            split = int(train_ratio * xtrain.shape[0])
            train_loader = DataLoader(TensorDataset(xtrain[ids[:split]], ytrain[ids[:split]]), batch_size=batch_size)
            test_loader = DataLoader(TensorDataset(xtrain[ids[split:]], ytrain[ids[split:]]), batch_size=batch_size)
            yield train_loader, test_loader
        else:
            kf = KFold(n_splits=nfolds, shuffle=True)
            for train_index, test_index in kf.split(xtrain):
                train_loader = DataLoader(TensorDataset(xtrain[train_index], ytrain[train_index]), batch_size=batch_size)
                test_loader = DataLoader(TensorDataset(xtrain[test_index], ytrain[test_index]), batch_size=batch_size)
                yield train_loader, test_loader

    if train_ratio is None and nfolds is None:
        raise ValueError('you must specify either train_ratio or nfolds for performance evaluation')
    elif train_ratio is not None and nfolds is not None:
        raise ValueError('you must specify either train_ratio xor nfolds for performance evaluation')

    # Add channel if it's not there yet
    if xtrain.dim() < 4:
        xtrain = xtrain.clone()[:, None, :]

    results = []
    params, values = list(zip(*search_grid.items()))
    print('Launching hyperparameter tuning:')
    for p, v in zip(params, values):
        print(f'\t{p} = {v}')
    # Copy the state of the un-trained model
    init_state = model.state_dict().copy()
    # Iterate over all parameters via cartesian product using itertools.product
    for p in itertools.product(*values):
        # Initialize tuning parameters
        optim_params = dict(zip(params, p))
        batch_size = optim_params['batch_size']
        print(optim_params)
        del optim_params['batch_size']
        # Training / test loop
        losses_train, metrics_train, losses_test, metrics_test = [], [], [], []
        for train_loader, test_loader in split_data(xtrain, ytrain, batch_size, train_ratio, nfolds):
            # Reset model
            model.load_state_dict(init_state)
            # Instantiate optimizer
            optim = optim_fun(model.parameters(), **optim_params)
            # Train
            losse_train, metric_train = func(model, train_loader, optim, loss_fun,
                                                 metric_fun, epochs, device, batch_log_interval=0)
            # Test
            losse_test, metric_test = testing(model, test_loader, loss_fun, metric_fun, device)
            # Log results of the current fold
            losses_train.append(np.array(losse_train).mean(axis=1).tolist())
            metrics_train.append(np.array(metric_train).mean(axis=1).tolist())
            losses_test.append(np.array(losse_test).mean())
            metrics_test.append(np.array(metric_test).mean())

        # Log results for the current params
        res = {
            'loss_train': losses_train,
            'metric_train': metrics_train,
            'loss_test': losses_test,
            'metric_test': metrics_test
        }
        # Log the optimizer parameters
        optim_params['batch_size'] = batch_size
        res.update(optim_params)
        results.append(res)

    return results
