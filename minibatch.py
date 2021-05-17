from typing import Optional, Callable
from typing import Iterable, Callable

import torch
from torch.functional import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from data_utils import get_mnist, build_data_loaders
from net import Net
from training import testing
from training import training, accuracy
import matplotlib.pyplot as plt

import pandas
from ray import tune

# Requires the following to be called before execution:


# use_cuda = True
# device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
# train_dataset, test_dataset = get_mnist(normalize=True)




def minibatch_run(model: Module, train_dataset: Tensor, test_dataset: Tensor,loss_fun: Module, metric_fun: Callable,
             device, bz: int = 16, lr: float = 0.01, epochs: int = 5):
    ''' Runs Minibatch SGD with chosen batch size and learning rate for chosen number of epochs'''    
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=bz)
    minibatch = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses, acc = training(model, train_loader, minibatch, CrossEntropyLoss(), accuracy, epochs=epochs, device=device)
    losses, acc = testing(model, test_loader, CrossEntropyLoss(), accuracy, device=device)
    
    return losses, acc


def minibatch_build(config, device, train_dataset: Tensor = None, test_dataset: Tensor = None):
    '''Used by Tuning Function to Test the Minibatch SGD with different Paramters'''
    net = Net().to(device)    
    
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=config["b_size"])
    minibatch = torch.optim.SGD(net.parameters(), lr=config["lr"])
    
    tuning_iters = 10 
    # Acts like epochs, but ties better into the tune API

    for i in range(tuning_iters):
        training(net, train_loader, minibatch, CrossEntropyLoss(), accuracy, epochs=1, device=device)
        _, acc = testing(net, test_loader, CrossEntropyLoss(), accuracy, device=device)
        tune.report(mean_accuracy=acc)

def minibatch_tune(train_dataset: Tensor, test_dataset: Tensor, space, device):
    '''
    Tunes the Minbatch SGD and returns the best values for learning rate and batch size.
    Pass in datasets to reduce number of downloads
    '''
    analysis = tune.run(
        tune.with_parameters(minibatch_build, device=device, train_dataset=train_dataset, test_dataset=test_dataset), 
        config=space,
        num_samples=10)

    best_config = analysis.get_best_config(metric="mean_accuracy", mode="min")
    print("Best config: ", best_config)

    # Get a dataframe for analyzing trial results.
    df_analysis = analysis.dataframe()
    df_analysis.to_excel("minibatch_tune.xlsx")  
    return df_analysis, best_config