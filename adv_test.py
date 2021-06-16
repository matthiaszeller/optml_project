from os import X_OK
from adversary import *
from net import Net
import numpy as np
from torch.optim import Optimizer
import torch
from training import training, testing, accuracy
from optimizer import MiniBatchOptimizer
import matplotlib.pyplot as plt
from data_utils import get_mnist, build_data_loaders
train_dataset, test_dataset = get_mnist(normalize=True)

epsilons = np.arange(0, 0.5, 0.05)


accuracies_naive_fgsm = []
accuracies_naive_proj = []

criterion = torch.nn.CrossEntropyLoss()

epochs = 10
batch_size = 100
learning_rate = 0.01
decreasing_lr = False
use_cuda = True # GPU seems to raise erros on my side
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)

net_naive = Net().to(device)



mini_opt = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
losses, acc = training(net_naive, train_loader, mini_opt, criterion, accuracy, epochs=epochs, device=device)
losses, acc = testing(net_naive, test_loader, criterion, accuracy, device=device)


# Attack Naive Model
print("********************************")

print("Standard Attack")
for eps in epsilons:
    loss_attack, acc_attack  = attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_naive_fgsm.append(acc_attack)
print("********************************")


# Attack Naive Model
print("********************************")
for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break
delta = torch.zeros_like(X, requires_grad=True)
loss = criterion(net_naive(X + delta), y)
loss.backward()
print(delta.grad.abs().mean().item())

print("Projected Attack")
for eps in epsilons_proj:
    loss_attack, acc_attack  = projected_attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, alpha=1, num_iter=40, device=device)
    accuracies_naive_proj.append(acc_attack)
print("********************************")
for eps in epsilons_proj:
    loss_attack, acc_attack  = projected_attack(net_naive, criterion, accuracy, test_loader_s, epsilon=eps, alpha=0.1, num_iter=40, device=device)


# Robust v1
robust_net = Net().to(device)
protect_epochs = 10
protect_lr = 0.01
protect_bz = 100
protect_dec_lr = False
prot_train_loader = DataLoader(train_dataset, batch_size = protect_bz, shuffle=True)
prot_test_loader = DataLoader(test_dataset, batch_size = protect_bz, shuffle=False)

mini_opt_proc = MiniBatchOptimizer(robust_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)

loss_train, acc_train = protected_training(robust_net, prot_train_loader, mini_opt_proc, criterion, accuracy, epochs=protect_epochs, device=device)
loss_test, acc_test = testing(robust_net, test_loader, criterion, accuracy, device=device)
print("Success")

# Attack against Protected Model
accuracies_robust_fgsm = []
accuracies_robust_proj = []


# Attack Robust Model
print("********************************")
losses, acc = attack(robust_net, criterion, accuracy, prot_test_loader, epsilon=0.1, device=device)
print("Robust Net: ", acc)

for eps in epsilons:
    loss_attack, acc_attack  = attack(robust_net, criterion, accuracy, prot_test_loader, epsilon=eps, device=device)
    accuracies_robust_fgsm.append(acc_attack)

# Attack Robust Model
print("********************************")
losses, acc = projected_attack(robust_net, criterion, accuracy, prot_test_loader, epsilon=2, alpha=1e-1, num_iter=40, device=device)
print("Robust Net: ", acc)

for eps in epsilons_proj:
    loss_attack, acc_attack  = projected_attack(robust_net, criterion, accuracy, prot_test_loader, epsilon=eps, alpha=1e-2, num_iter=40, device=device)
    accuracies_robust_proj.append(acc_attack)




print("********************************")
for entry in  accuracies_naive_fgsm:
    print(entry)
print("********************************")
for entry in  accuracies_naive_proj:
    print(entry)
print("********************************")
for entry in  accuracies_robust_fgsm:
    print(entry)
print("********************************")
for entry in  accuracies_robust_proj:
    print(entry)
    