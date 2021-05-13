from typing import Optional, Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from data_utils import get_mnist, build_data_loaders
from net import Net
from training import testing, training_debug
from training import training, accuracy

use_cuda = True

device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

train_dataset, test_dataset = get_mnist(normalize=True)

# Build data loaders
batch_size = 64
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)

net = Net().to(device)


class NesterovOptimizer(Optimizer):

    def __init__(self, params, lr):
        defaults = dict(lr=lr)

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        # State stores the variables used by optimization algo
        # need to initialize it if empty
        if len(self.state) == 0:
            self.__initialize_state()

        step = self.state['general']['step']
        # Iterate over parameter groups (only 1 group in our case)
        for gp in self.param_groups:
            lr = gp['lr']
            params = gp['params']
            # Iterate over parameters in the group
            for p in params:
                if p.grad is None:
                    continue

                pstate = self.state[p]
                # Implementation of Nesterov algorithm
                pstate['y'] = p.data - lr * p.grad
                pstate['z'] = pstate['z'] - (step + 1) * lr / 2 * p.grad
                p.data = (step + 1) / (step + 3) * pstate['y'] + 2 / (step + 3) * pstate['z']

        # Increment step
        self.state['general']['step'] += 1

    def __initialize_state(self):
        """Initialize state variables:
            - there's a 'general' state in which we store the steps
            - there's a state for each model parameter
        """
        self.state['general']['step'] = 0
        for gp in self.param_groups:
            # Initialize parameter-specific state
            for p in gp['params']:
                # But only for parameters having a gradient
                if p.grad is None:
                    continue

                pstate = self.state[p]
                pstate['z'] = p.data.clone()


opt = NesterovOptimizer(net.parameters(), lr=0.01)
losses, metrics = training_debug(net,
                                 train_loader,
                                 opt,
                                 CrossEntropyLoss(),
                                 accuracy,
                                 device=device,
                                 n_steps=10)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
# losses, metrics = testing(net, test_loader, CrossEntropyLoss(), accuracy, device)
