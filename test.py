import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from data_utils import get_mnist, build_data_loaders
from net import Net
from training import testing, training_debug
from training import training, accuracy, tune_optimizer
from optimizer import NesterovOptimizer, AdamOptimizer

use_cuda = True

device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

train_dataset, test_dataset = get_mnist(normalize=True)

net = Net().to(device)

fp = 'adam_tuning.json'
# Want to test those final configs with KFold CV:
# "0":{
#     "lr":0.00008,
#     "beta1":0.9,
#     "beta2":0.999,
#     "weight_decay":0.01,
#     "epsilon":0.00000001,
#     "batch_size":32.0
#   },
# "3": {
#     "lr": 0.0002,
#     "beta1": 0.9,
#     "beta2": 0.999,
#     "weight_decay": 1.0,
#     "epsilon": 0.00000001,
#     "batch_size": 64.0
# },
# "5": {
#     "lr": 0.0002,
#     "beta1": 0.9,
#     "beta2": 0.999,
#     "weight_decay": 0.01,
#     "epsilon": 0.00000001,
#     "batch_size": 128.0
# },
results = tune_optimizer(
    net,
    train_dataset.data,
    train_dataset.targets,
    CrossEntropyLoss(),
    accuracy,
    device,
    AdamOptimizer,
    epochs=10,
    search_grid={
        'lr': [2e-4], #, 0.001, 0.01],
        'batch_size': [128],#, 64, 128],
        'beta1': [0.9],
        'beta2': [0.999],
        'weight_decay': [0.01],#, 1e-1, 1.0],
        'epsilon': [1e-8]
    },
    nfolds=5
)

if Path(fp).exists():
    with open(fp, 'r') as f:
        old_results = json.load(f)

    results = old_results + results

with open(fp, 'w') as f:
    json.dump(results, f, indent=2)

