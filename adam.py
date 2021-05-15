from typing import Optional, Callable
import math

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

        for gp in self.param_groups:
            # Initialize parameter-specific state
            for p in gp['params']:
                # But only for parameters having a gradient
                if p.grad is None:
                    continue

                if len(self.state[p]) == 0:
                    self.state[p]['step'] = 0
                    self.state[p]['moment1'] = torch.zeros(p.data.size())
                    self.state[p]['moment2'] = torch.zeros(p.data.size())


    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        # State stores the variables used by optimization algo
        # need to initialize it if empty
        if len(self.state) == 0:
            self.__initstate__()


        # Iterate over parameter groups
        for pg in self.param_groups:

            #Get all hyper parameters
            lr = pg['lr']
            wd = pg['weight_decay']
            epsilon = pg['epsilon']
            params = pg['params']
            beta1 = pg['beta1']
            beta2 = pg['beta2']

            # Iterate over parameters in the group
            for p in params:

                #Get param within state
                state_p = self.state[p]

                #Increment step by 1
                state_p['step'] += 1
                grad = p.grad.data

                #Only need to continue on parameters with gradient
                if grad is None:
                    continue

                #Adam does not support sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("Adam optimizer does not support sparse gradients!")

                #Bias corrections
                bias1 = 1 - (beta1 ** state_p['step'])
                bias2 = 1 - (beta2 ** state_p['step'])

                #m_t
                state_p['moment1'] = state_p['moment1'] * self.beta1 + (1 - self.beta1) * grad
                #v_t
                state_p['moment2'] = state_p['moment2'] * self.beta2 + grad * grad * (1 - self.beta2)

                #Bias corrections are to be multiplied as they are non-tensor singular values
                bias_corr = math.sqrt(bias2) / bias1
                lr *= bias_corr

                #Update parameters according to bias corrections
                p.data = p.data - lr * state_p['moment1'] / (state_p['moment2'].sqrt() + epsilon)
               
                #If weight decay exists, update parameters accordingly
                if wd != 0.0:
                    p.data = p.data - lr * wd * p.data


opt = AdamOptimizer(net.parameters())
losses, metrics = training_debug(net,
                                 train_loader,
                                 opt,
                                 CrossEntropyLoss(),
                                 accuracy,
                                 device=device,
                                 n_steps=100)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
# losses, metrics = testing(net, test_loader, CrossEntropyLoss(), accuracy, device)
