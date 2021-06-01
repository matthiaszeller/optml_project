import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from net import Net
from training import training, testing, accuracy
from minibatch import MiniBatchOptimizer
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

fp = 'mini_tuning.json'
results = []
criterion = torch.nn.CrossEntropyLoss()

if Path(fp).exists():
    with open(fp, 'r') as f:
        old_results = json.load(f)

    results = old_results + results

df_analysis = pd.DataFrame(results)
best_acc = 0.0
clean = ["[", "]"]
for index, row in df_analysis.iterrows():    
        trial_acc = row["metric_test"]
        if trial_acc > best_acc:
            best_acc = trial_acc
            learning_rate = round(row["lr"], 6)
            decreasing_lr = row["decreasing_lr"]

print("Best Accuracy was {}% with Learning Rate {} and Decreasing LR: {}".format(100*best_acc, learning_rate, decreasing_lr))

# Train Naive Model
accuracy_naive= []
losses_naive= []
net_naive = Net().to(device)

epochs = 10

mini_opt_naive = MiniBatchOptimizer(net_naive.parameters(), lr=learning_rate, decreasing_lr=decreasing_lr)
loss_train, acc_train = training(net_naive, train_loader, mini_opt_naive, criterion, accuracy, epochs=epochs, device=device)
loss_test, acc_test = testing(net_naive, test_loader, criterion, accuracy, device=device)


def fgsm_alt(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break
    
def plot_images(X,y,yp,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()

### Illustrate original predictions
yp = net_naive(X)
plot_images(X, y, yp, 3, 6)