from typing import Optional, Callable
from typing import Iterable, Callable

import torch
from torch.functional import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

import numpy as np

from data_utils import get_mnist, build_data_loaders
from net import Net
from training import testing, training_debug
from training import training, accuracy
import matplotlib.pyplot as plt

from ray import tune

class MiniBatchOptimizer(Optimizer):
    def __init__(self, params, lr : float = 1e-2, decreasing_lr: bool = False) -> None:
        if(lr < 0.0):
            self.lr = 1e-2
        else:
            self.lr = lr
        
        self.iter = 0 # Used to count iterations for decreaing learning rates
        self.dec_lr = decreasing_lr # Used to either have a fixed learning rate or a decreasing one
        self.lr_init = self.lr # Stores initial learning rate

        defaults = dict(lr=self.lr, dec_lr=self.dec_lr, lr_init=self.lr_init)

        super(MiniBatchOptimizer, self).__init__(params, defaults)

    def step(self) -> Optional[float]:
        # Iterate over parameter groups
        for pg in self.param_groups:

            #Get all hyper parameters
            lr = pg['lr']
            lr_init = pg['lr_init']
            iteration_counter = self.iter
            decreasing_lr = pg['dec_lr']
            params = pg['params']

            # Iterate over parameters in the group
            for p in params:
                grad = p.grad.data
                if decreasing_lr: # If we want a decreasing learning rate, divide it by Iterations + 1
                    lr = lr_init / (iteration_counter + 1)
                else:
                    lr = lr_init

                #Update parameters via GD
                p.data = p.data - lr * grad

            self.iter = self.iter + 1 # Update iterator



def minibatch_run(model: Module, train_dataset: Tensor, test_dataset: Tensor,loss_fun: Module,
                    metric_fun: Callable, device, bz: int = 16, lr: float = 0.01, epochs: int = 5, dec_lr : bool = False):
    ''' Runs Minibatch SGD with chosen batch size and learning rate for chosen number of epochs'''    
    print("Run Minibatch SGD with batch size {}, learning rate {:.4f} and decreasing ({}) for {} epochs".format(bz, lr, dec_lr, epochs))
    
    # The batching is done automatically by the data loaders from Pytorch so the Optimiser just adds the step function
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=bz)
    minibatch = MiniBatchOptimizer(model.parameters(), lr=lr, decreasing_lr=dec_lr)
    
    # The training and testing loops are in the training.py file
    losses, acc = training(model, train_loader, minibatch, loss_fun, metric_fun, epochs=epochs, device=device)
    losses, acc = testing(model, test_loader, loss_fun, metric_fun, device=device)
    
    return losses, acc


def minibatch_build(config, model:Module, device, loss_fun: Module, train_dataset: Tensor = None, test_dataset: Tensor = None):
    '''Used by Tuning Function to Test the Minibatch SGD with different Paramters'''  
    # Bayes Search in Ray Tune samples uniformly from real numbers in a range and not just integers
    # Thus, since batch sizes are ints and the decreasing lr parameter is a bool we do some formatting here
    b_sz_int = int(config["b_size"])
    dec_lr_int = round(config["dec_lr"])
    dec_lr_bool = True if dec_lr_int else False

    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=b_sz_int)
    minibatch = MiniBatchOptimizer(model.parameters(), lr=config["lr"], decreasing_lr=dec_lr_bool)
    
    tuning_iters = 5 
    # Acts like epochs, but ties better into the tune API

    for i in range(tuning_iters):
        training(model, train_loader, minibatch, loss_fun, accuracy, epochs=1, device=device)
        _, acc = testing(model, test_loader, loss_fun, accuracy, device=device)
        tune.report(mean_accuracy=acc)

def minibatch_tune(model : Module, train_dataset: Tensor, test_dataset: Tensor, samples, criterion, space, opt, device):
    '''
    Tunes the Minbatch SGD and returns the best values for learning rate and batch size.
    Pass in datasets to reduce number of downloads
    '''
    # Tune works by taking a function, a constrain space, an optimiser and a number of runs 
    # and tries to achieve the highest result possible via Bayes Search
    # Tune with Parameters allows us to pass values through to the tuner so we do not keep rebuilding the model
    analysis = tune.run(
        tune.with_parameters(minibatch_build, model=model, device=device, loss_fun=criterion, train_dataset=train_dataset, test_dataset=test_dataset), 
        config=space,
        search_alg=opt,
        num_samples=samples) # Chnage back to 10

    # Get a dataframe for analyzing trial results.
    df_analysis = analysis.dataframe()
    df_analysis.to_csv("minibatch_tune.csv")  
    
    best_acc = 0.0
    clean = ["[", "]"]

    # This loop is fairly quick and makes us of the already exisiting df for export 
    # to have more control about the sorting of the results
    for index, row in df_analysis.iterrows():    
        base_array = row["mean_accuracy"]
        for char in clean:
            base_array = base_array.replace(char, "")

        array_acc_string = base_array.split(", ")
        array_acc = [float(item) for item in array_acc_string]
        trial_mean = np.mean(np.array(array_acc))
        if trial_mean > best_acc:
            best_acc = trial_mean
            best_performer = row["logdir"]
    
    configs = analysis.get_all_configs()
    best_config = configs[best_performer]
    
    return best_config, best_acc
