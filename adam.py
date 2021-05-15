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


class AdamOptimizer(Optimizer):

    def __init__(self, params, lr = 1e-2, beta1=0.9, beta2=0.999, weight_decay = 0.0, epsilon = 5e-9):

        if(lr < 0.0):
            self.lr = 1e-2
            print("Learning rate set to default (1e-2) due to negative learning rate input!")
        else:
            self.lr = lr
        if(beta1 < 0.0 or beta1 >= 1.0):
            self.beta1 = 0.9
            print("Beta1 set to default value (0.9) due to negative beta1 input!")
        else:
            self.beta1 = beta1
        if(beta2 < 0.0 or beta2 >= 1.0):
            self.beta2 = 0.999
            print("Beta2 set to default value (0.999) due to negative beta2 input!")
        else:
            self.beta2 = beta2
        if(weight_decay < 0.0):
            self.weight_decay = 0.0
            print("Weight decay set to default value (0.0) due to negative weight decay input!")
        else:
            self.weight_decay = weight_decay
        if(epsilon < 0):
            self.epsilon = 5e-9
            print("Epsilon set to default value (5e-9) due to negative input!")
        else:
            self.epsilon = epsilon

        defaults = dict(lr = self.lr, beta1 = self.beta1, beta2=self.beta2, weight_decay=self.weight_decay, epsilon=self.epsilon)

        super(AdamOptimizer,self).__init__(params, defaults)

    def __initstate__(self):
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

                if len(self.state[p]) == 0:
                    self.state[p]['moment1'] = torch.zeros(p.size())
                    self.state[p]['moment2'] = torch.zeros(p.size())
                #self.state['z'] = p.data.clone()


    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        # State stores the variables used by optimization algo
        # need to initialize it if empty
        if len(self.state) == 0:
            self.__initialize_state()


        # Iterate over parameter groups (only 1 group in our case)
        step = self.state['general']['step']
        for gp in self.param_groups:
            lr = gp['lr']
            params = gp['params']
            # Iterate over parameters in the group
            for p in params:
                #Only need to continue on parameters with gradient
                if p.grad is None:
                    continue

                # Implementation of Adam algorithm
                p.grad = p.grad.add(p, alpha = weight_decay)

                self.bias1 = 1 - self.beta1 ** step
                self.bias2 = 1 - self.beta2 ** step

                self.min_beta1 = 1 - self.beta1
                self.min_beta2 = 1 - self.beta2


                self.state[p]['moment1'] = self.state[p]['moment1'].mul(self.beta1)
                self.state[p]['moment1'] = self.state[p]['moment1'].add(p.grad, alpha = self.min_beta1)
                self.state[p]['moment1'] = self.state[p]['moment1'].div(self.bias1)

                self.state[p]['moment2'] = self.state[p]['moment2'].mul(self.beta2)
                self.state[p]['moment2'] = self.state[p]['moment2'].addcmul(p.grad, p.grad, value = self.min_beta2)
                self.state[p]['moment2'] = self.state[p]['moment2'].div(self.bias2)

                self.update_val = self.state[p]['moment1'].div(torch.sqrt(self.state[p]['moment2'].add(self.epsilon)))
                p.data = p.data.sub(self.update_val, alpha = self.lr)

        # Increment step
        self.state['general']['step'] += 1



import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
# losses, metrics = testing(net, test_loader, CrossEntropyLoss(), accuracy, device)
