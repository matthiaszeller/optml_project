

from time import time
from typing import Iterable, Callable

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer


def accuracy(yhat, y):
    prediction = yhat.argmax(dim=1)
    return (y.eq(prediction)).to(float).mean().item()


def train_epoch(model: Module, dataset: Iterable, optim: Optimizer, loss_fun: Module, metric_fun: Callable, device,
                log_interval: int = 0):
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
            print(f'batch {batch_id}\tloss = {losses[-1]:.4}\tacc = {metrics[-1]:.4}')

    return losses, metrics


def training(model: Module, dataset: Iterable, optim: Optimizer, loss_fun: Module, metric_fun: Callable, epochs: int,
             device, batch_log_interval: int = 100):
    print(f'Launching training on {device}')
    losses_epoch = []
    metrics_epoch = []
    model.train()
    t = time()
    for epoch in range(epochs):
        losses, metrics = train_epoch(model, dataset, optim, loss_fun, metric_fun, device, batch_log_interval)
        losses_epoch.append(losses)
        metrics_epoch.append(metrics)
        print(f'epoch {epoch}\tavg epoch loss = {np.mean(losses):.4}\tavg epoch acc = {np.mean(metrics):.4}')

    t = time() - t
    print(f'training took {t:.4} s')

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

    print(f'Avg loss = {np.mean(losses)}\tAvg acc = {np.mean(metrics)}')
    return losses, metrics
