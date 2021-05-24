import itertools
from time import time
from typing import Iterable, Callable, Dict, List

import numpy as np
import torch
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
    if metric_fun is None:
        metric_fun = lambda *args: None
    if device is None:
        device = torch.device('cpu')

    print(f'Launching training on {device}')
    losses_epoch = []
    metrics_epoch = []
    model.train()
    t = time()
    for epoch in range(epochs):
        losses, metrics = train_epoch(model, dataset, optim, loss_fun, metric_fun, device, batch_log_interval)
        losses_epoch.append(losses)
        metrics_epoch.append(metrics)
        print_metric = '' if metric_fun is None else f'\tavg epoch acc = {np.mean(metrics):.4}'
        print(f'epoch {epoch}\tavg epoch loss = {np.mean(losses):.4}{print_metric}')

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
                   train_ratio: float = 0.8,
                   batch_size: int = 64):
    ids = torch.randperm(xtrain.shape[0])
    split = int(train_ratio * xtrain.shape[0])

    train_loader = DataLoader(TensorDataset(xtrain[ids[:split]], ytrain[ids[:split]]), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(xtrain[ids[split:]], ytrain[ids[split:]]), batch_size=batch_size)

    # Iterate over all parameters
    params, values = list(zip(*search_grid.items()))
    init_state = model.state_dict().copy()
    results = []
    for p in itertools.product(*values):
        optim_params = dict(zip(params, p))
        print(optim_params)
        optim = optim_fun(model.parameters(), **optim_params)
        model.load_state_dict(init_state)
        losses_train, metrics_train = training(model, train_loader, optim, loss_fun,
                                               metric_fun, epochs, device, batch_log_interval=0)
        losses_test, metrics_test = testing(model, test_loader, loss_fun, metric_fun, device)
        res = {
            'loss_train': np.array(losses_train).mean(axis=1).tolist(),
            'metric_train': np.array(metrics_train).mean(axis=1).tolist(),
            'loss_test': np.array(losses_test).mean(),
            'metric_test': np.array(metrics_test).mean()
        }
        res.update(optim_params)
        results.append(res)

    return results
