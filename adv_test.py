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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
train_dataset_s = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
test_dataset_s = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
# train_loader_s = DataLoader(train_dataset_s, batch_size = 100, shuffle=False)
# test_loader_s = DataLoader(test_dataset_s, batch_size = 100, shuffle=False)

train_loader_s = DataLoader(TensorDataset(train_dataset_s.data[:, None, :], train_dataset_s.targets), batch_size=1)
test_loader_s = DataLoader(TensorDataset(test_dataset_s.data[:, None, :], test_dataset_s.targets), batch_size=1)

epsilons_fgsm = np.arange(0, 0.5, 0.05)

epsilons_proj = []

n = 784 # Thus n is the dimension of each image, which we need to scale the L2 norm for Projected Gradient Descent

# From source https://adversarial-ml-tutorial.org/adversarial_examples/ for L2 PGD
scaler = np.sqrt(2*n)/(np.sqrt(np.pi*np.exp(1)))
for eps in epsilons_fgsm:
    new_eps = eps*scaler
    epsilons_proj.append(new_eps)


accuracies_naive_fgsm = []
accuracies_naive_proj = []

criterion = torch.nn.CrossEntropyLoss()

epochs = 10
batch_size = 1
learning_rate = 0.01
decreasing_lr = False
use_cuda = True # GPU seems to raise erros on my side
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')


net_naive = Net().to(device)

train_dataset, test_dataset = get_mnist(normalize=False)
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)

mini_opt = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
losses, acc = training(net_naive, train_loader, mini_opt, criterion, accuracy, epochs=epochs, device=device)
losses, acc = testing(net_naive, test_loader, criterion, accuracy, device=device)


# Attack Naive Model
print("********************************")

print("Standard Attack")
for eps in epsilons_fgsm:
    loss_attack, acc_attack  = attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_naive_fgsm.append(acc_attack)
print("********************************")
for eps in epsilons_fgsm:
    loss_attack, acc_attack  = attack(net_naive, criterion, accuracy, test_loader_s, epsilon=eps, device=device)


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

for eps in epsilons_fgsm:
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
    