from adversary import attack, protect
from net import Net
import numpy as np
from torch.optim import Optimizer
import torch
from training import training, testing, accuracy
from minibatch import MiniBatchOptimizer
import matplotlib.pyplot as plt
from data_utils import get_mnist, build_data_loaders


train_dataset, test_dataset = get_mnist(normalize=True)
accuracies_lenet= []
examples_lenet = []

epsilons_lenet = np.arange(0, 0.1, 0.05)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
batch_size = 91
learning_rate = 0.01
decreasing_lr = False
use_cuda = False # GPU seems to raise erros on my side
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')


net_naive = Net().to(device)
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)
for x, y in train_loader:
    print(x.shape, y.shape)
    print(train_loader)
    break

mini_opt = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
losses, acc = training(net_naive, train_loader, mini_opt, criterion, accuracy, epochs=epochs, device=device)
losses, acc = testing(net_naive, test_loader, criterion, accuracy, device=device)


for eps in epsilons_lenet:
    acc, ex = attack(net_naive, criterion, test_loader, update_max_norm=eps, device=device)
    accuracies_lenet.append(acc)
    examples_lenet.append(ex)

robust_net = Net().to(device)
protect_epochs = 10
protect_lr = 0.01
protect_bz = 32
protect_dec_lr = False
prot_train_loader, prot_test_loader = build_data_loaders(train_dataset, test_dataset, protect_bz)
mini_opt_proc = MiniBatchOptimizer(robust_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)

robust_net = protect(robust_net, mini_opt_proc, criterion, prot_train_loader, prot_test_loader, device=device, epochs=protect_epochs)
print("Success")

# Attack against Protected Model
accuracies_lenet_robust = []
examples_lenet_robust = []

epsilons_lenet_robust = np.arange(0, 0.5, 0.05)
for eps in epsilons_lenet_robust:
    acc, ex = attack(robust_net, criterion, prot_train_loader, eps, device=device)
    accuracies_lenet_robust.append(acc)
    examples_lenet_robust.append(ex)

# Comparing the models
plt.figure(figsize=(5,5))
plt.plot(epsilons_lenet, accuracies_lenet, "*-", c='blue', label='Convolutional network')
plt.plot(epsilons_lenet_robust, accuracies_lenet_robust, "*-", c='orange', label='Convolutional network (robust)')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))

plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend();