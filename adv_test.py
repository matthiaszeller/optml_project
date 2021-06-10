from adversary import *
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
accuracies_lenet_alt= []
accuracies_lenet_proj= []


epsilons_lenet = np.arange(0, 0.5, 0.05)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
batch_size = 32
learning_rate = 0.01
decreasing_lr = False
use_cuda = True # GPU seems to raise erros on my side
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')


net_naive = Net().to(device)
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)


mini_opt = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
losses, acc = training(net_naive, train_loader, mini_opt, criterion, accuracy, epochs=epochs, device=device)
losses, acc = testing(net_naive, test_loader, criterion, accuracy, device=device)


# Attack Naive Model
print("********************************")
print("Standard Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet.append(acc_attack)


# Attack Naive Model
print("********************************")
print("Alternate Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = fgsm_attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet_alt.append(acc_attack)

# Attack Naive Model
print("********************************")
print("Projected Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = projected_attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, alpha=1e-2, num_iter=40, device=device)
    accuracies_lenet_proj.append(acc_attack)

# Robust v1
robust_net = Net().to(device)
protect_epochs = 10
protect_lr = 0.01
protect_bz = 32
protect_dec_lr = False
prot_train_loader, prot_test_loader = build_data_loaders(train_dataset, test_dataset, protect_bz)
mini_opt_proc = MiniBatchOptimizer(robust_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)

loss_train, acc_train = protected_training(robust_net, prot_train_loader, mini_opt_proc, criterion, accuracy, epochs=protect_epochs, device=device)
loss_test, acc_test = testing(robust_net, test_loader, criterion, accuracy, device=device)
print("Success")

# Attack against Protected Model
accuracies_lenet_rb= []
accuracies_lenet_alt_rb= []
accuracies_lenet_proj_rb= []


epsilons_lenet_robust = np.arange(0, 0.5, 0.05)
# Attack Robust Model
print("********************************")
print("Standard Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = attack(robust_net, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet_rb.append(acc_attack)

# Attack Robust Model
print("********************************")
print("Alternate Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = fgsm_attack(robust_net, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet_alt_rb.append(acc_attack)

# Attack Robust Model
print("********************************")
print("Projected Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = projected_attack(robust_net, criterion, accuracy, test_loader, epsilon=eps, alpha=1e-2, num_iter=40, device=device)
    accuracies_lenet_proj_rb.append(acc_attack)


# Robust v2
robustest_net = Net().to(device)
protect_epochs = 10
protect_lr = 0.01
protect_bz = 32
protect_dec_lr = False
prot_train_loader, prot_test_loader = build_data_loaders(train_dataset, test_dataset, protect_bz)
mini_opt_proc = MiniBatchOptimizer(robustest_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)

loss_train, acc_train = protected_training_alt(robustest_net, prot_train_loader, mini_opt_proc, criterion, accuracy, epochs=protect_epochs, device=device)
loss_test, acc_test = testing(robustest_net, test_loader, criterion, accuracy, device=device)
print("Success")

# Attack against Protected Model
accuracies_lenet_rb_2= []
accuracies_lenet_alt_rb_2= []
accuracies_lenet_proj_rb_2= []


epsilons_lenet_robust = np.arange(0, 0.5, 0.05)
# Attack Robust Model
print("********************************")
print("Standard Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = attack(robustest_net, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet_rb_2.append(acc_attack)

# Attack Robust Model
print("********************************")
print("Alternate Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = fgsm_attack(robustest_net, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracies_lenet_alt_rb_2.append(acc_attack)

# Attack Robust Model
print("********************************")
print("Projected Attack")
for eps in epsilons_lenet:
    loss_attack, acc_attack  = projected_attack(robustest_net, criterion, accuracy, test_loader, epsilon=eps, alpha=1e-2, num_iter=40, device=device)
    accuracies_lenet_proj_rb_2.append(acc_attack)

print("Success")
print("********************************")
for entry in  accuracies_lenet:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_alt:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_proj:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_rb:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_alt_rb:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_proj_rb:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_rb_2:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_alt_rb_2:
    print(entry)
print("********************************")
for entry in  accuracies_lenet_proj_rb_2:
    print(entry)
    