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

opt = AdamOptimizer(net.parameters(), lr=0.0005)

fp = 'adam_tuning.json'
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
        'lr': [0.0001, 0.001, 0.01],
        'beta1': np.linspace(0.1, 0.9, 3),
        'beta2': np.linspace(0.5, 0.999, 3),
        'weight_decay': np.logspace(-4, 0, 3),
        'epsilon': np.logspace(-10, -8, 3)
    }
)

if Path(fp).exists():
    with open(fp, 'r') as f:
        old_results = json.load(f)

    results = old_results + results

with open(fp, 'w') as f:
    json.dump(results, f, indent=2)

