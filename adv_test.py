from adversary import attack, protected_training
from net import Net
import numpy as np
from torch.optim import Optimizer
import torch
from training import training, testing, accuracy, tune_optimizer
from optimizer import MiniBatchOptimizer
import matplotlib.pyplot as plt
from data_utils import get_mnist, build_data_loaders
import json
from pathlib import Path
import random
import pandas as pd
from adversary import projected_attack, projected_protect

#Get data + Setup
use_cuda = True # GPU seems to raise erros on my side
device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
train_dataset, test_dataset = get_mnist(normalize=True)

epsilons = np.arange(0, 0.5, 0.05)
criterion = torch.nn.CrossEntropyLoss()
epochs = 10
batch_size = 16

# # Tune Hyperparameters
net_tune = Net().to(device)
mini_opt_tune = MiniBatchOptimizer(net_tune.parameters()) # Just using defaults

dec_lr_set =  [0]*1 + [1]*1
random.shuffle(dec_lr_set)

fp = 'res/minibatch_tuning_round2.json'

results = []

# results = tune_optimizer(
#     net_tune,
#     train_dataset.data,
#     train_dataset.targets,
#     criterion,
#     accuracy,
#     device,
#     MiniBatchOptimizer,
#     epochs=10,
#     search_grid={
#         'lr': np.linspace(0.00001, 0.1, 3),
#         'decreasing_lr': dec_lr_set,
#     }, 
#     batch_size=16
# )

if Path(fp).exists():
    with open(fp, 'r') as f:
        old_results = json.load(f)

    results = old_results + results

with open(fp, 'w') as f:
    json.dump(results, f, indent=2)

# Select Best Hyperparamters
with open(fp, 'r') as f:
        old_results = json.load(f)

# df_analysis = pd.DataFrame(results)
# best_acc = 0.0
# clean = ["[", "]"]
# for index, row in df_analysis.iterrows():    
#         trial_acc = row["metric_test"]
#         if trial_acc > best_acc:
#             best_acc = trial_acc
#             learning_rate = round(row["lr"], 6)
#             decreasing_lr = row["decreasing_lr"]
learning_rate = 0.001
decreasing_lr = False
#print("Best Accuracy was {}% with Learning Rate {} and Decreasing LR: {}".format(100*best_acc, learning_rate, decreasing_lr))

# Train Naive Model
accuracy_naive= []
losses_naive= []
accuracy_naive_proj= []
losses_naive_proj= []
net_naive = Net().to(device)
train_loader, test_loader = build_data_loaders(train_dataset, test_dataset, batch_size)


mini_opt_naive = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
loss_train, acc_train = training(net_naive, train_loader, mini_opt_naive, criterion, accuracy, epochs=epochs, device=device)
loss_test, acc_test = testing(net_naive, test_loader, criterion, accuracy, device=device)

# Attack Naive Model
for eps in epsilons:
    loss_attack, acc_attack  = attack(net_naive, criterion, accuracy, test_loader, epsilon=eps, device=device)
    accuracy_naive.append(acc_attack)
    losses_naive.append(loss_attack)

# for eps in epsilons:
#     loss_attack, acc_attack  = projected_attack(net_naive, criterion, test_loader, epsilon=eps, device=device, lr=learning_rate)
#     accuracy_naive_proj.append(acc_attack)
#     losses_naive_proj.append(loss_attack)

# Train Robust Model
robust_net = Net().to(device)
robust_proj_net = Net().to(device)
protect_epochs = epochs
protect_lr = learning_rate
protect_bz = batch_size
protect_dec_lr = decreasing_lr

prot_train_loader, prot_test_loader = build_data_loaders(train_dataset, test_dataset, protect_bz)

mini_opt_proc = MiniBatchOptimizer(robust_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)
#mini_opt_projected = MiniBatchOptimizer(robust_proj_net.parameters(), lr=protect_lr, decreasing_lr=protect_dec_lr)

print("*********************************")
loss_train, acc_train = protected_training(robust_net, prot_train_loader, mini_opt_proc, criterion, accuracy, epochs=protect_epochs, device=device, epsilon=0.25)
loss_test, acc_test = testing(robust_net, test_loader, criterion, accuracy, device=device)
#robust_proj_net = projected_protect(robust_proj_net, mini_opt_projected, criterion, prot_train_loader, prot_test_loader, device=device, epochs=protect_epochs, lr=protect_lr)
print("Success")

# Attack Robust Model
accuracy_robust = []
losses_robust = []
accuracy_robust_proj = []
losses_robust_proj = []

for eps in epsilons:
    loss_attack, acc_attack = attack(robust_net, criterion, accuracy, prot_test_loader, eps, device=device)
    accuracy_robust.append(acc_attack)
    losses_robust.append(loss_attack)


for eps in epsilons:
    loss_attack, acc_attack  = projected_attack(robust_proj_net, criterion, accuracy, prot_test_loader, epsilon=eps, device=device, lr=learning_rate)
    accuracy_robust_proj.append(acc_attack)
    losses_robust_proj.append(loss_attack)



# Comparing the models
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracy_naive, "*-", c='blue', label='Naive Model')
plt.plot(epsilons, accuracy_robust, "*-", c='orange', label='Robust Model')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 5, step=0.05))

plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend();

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracy_naive_proj, "*-", c='blue', label='Naive Model with Projection')
plt.plot(epsilons, accuracy_robust_proj, "*-", c='orange', label='Robust Model with Projection')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 5, step=0.05))

plt.title("Accuracy vs Epsilon with Projection")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend();
