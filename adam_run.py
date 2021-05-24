Jupyter Notebook
minibatch.py
Last Monday at 3:49 PM
Python
File
Edit
View
Language
1
from typing import Optional, Callable
2
from typing import Iterable, Callable
3
​
4
import torch
5
from torch.functional import Tensor
6
from torch.nn import CrossEntropyLoss, Module
7
from torch.optim import Optimizer

from adam import AdamOptimizer
8
​
9
from data_utils import get_mnist, build_data_loaders
10
from net import Net
11
from training import testing
12
from training import training, accuracy
13
import matplotlib.pyplot as plt
14
​
15
import pandas
16
from ray import tune
17
​
18
# Requires the following to be called before execution:
19
​
20
​
21
# use_cuda = True
22
# device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
23
# train_dataset, test_dataset = get_mnist(normalize=True)
24
​
25
​
26
​
27
​
28
def adam_run(model: Module, train_dataset: Tensor, test_dataset: Tensor,loss_fun: Module, metric_fun: Callable,
29
             device, bz: int = 16, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0, epsilon : float = 1e-8, epochs: int = 5):
30
    ''' Runs Adam with chosen batch size, learning rate, beta_parameters, weight decay and epsilon for chosen number of epochs'''    
31
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=bz)
32
    adam = AdamOptimizer(model.parameters(), lr=lr, beta1=beta1, beta2=beta2,weight_decay=weight_decay, epsilon=epsilon)
33
    
34
    losses, acc = training(model, train_loader, adam, CrossEntropyLoss(), accuracy, epochs=epochs, device=device)
35
    losses, acc = testing(model, test_loader, CrossEntropyLoss(), accuracy, device=device)
36
    
37
    return losses, acc
38
​
39
​
40
def adam_build(config, device, train_dataset: Tensor = None, test_dataset: Tensor = None):
41
    '''Used by Tuning Function to Test the Minibatch SGD with different Paramters'''
42
    net = Net().to(device)    
43
    
44
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size=config["b_size"])
45
    adam = AdamOptimizer(net.parameters(), lr=config["lr"], beta1=config["beta1"], beta2=config["beta2"], weight_decay=config["weight_decay"], epsilon=config["epsilon"])
46
    
47
    tuning_iters = 10 
48
    # Acts like epochs, but ties better into the tune API
49
​
50
    for i in range(tuning_iters):
51
        training(net, train_loader, adam, CrossEntropyLoss(), accuracy, epochs=100, device=device)
52
        _, acc = testing(net, test_loader, CrossEntropyLoss(), accuracy, device=device)
53
        tune.report(mean_accuracy=acc)
54
​
55
def adam_tune(train_dataset: Tensor, test_dataset: Tensor, space, device):
56
    '''
57
    Tunes the Minbatch SGD and returns the best values for learning rate and batch size.
58
    Pass in datasets to reduce number of downloads
59
    '''
60
    analysis = tune.run(
61
        tune.with_parameters(adam_build, device=device, train_dataset=train_dataset, test_dataset=test_dataset), 
62
        config=space,
63
        num_samples=10)
64
​
65
    best_config = analysis.get_best_config(metric="mean_accuracy", mode="min")
66
    print("Best config: ", best_config)
67
​
68
    # Get a dataframe for analyzing trial results.
69
    df_analysis = analysis.dataframe()
70
    df_analysis.to_excel("adam_tune.xlsx")  
71
    return df_analysis, best_config
