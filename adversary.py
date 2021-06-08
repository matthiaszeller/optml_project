

from time import time
from typing import Iterable

import torch
from torch.functional import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from training import accuracy


## Source: https://adversarial-ml-tutorial.org/adversarial_examples/

def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm,
                               clamp=(0,1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                    .norm(step_norm, dim=-1)\
                    .view(-1, num_channels, 1, 1)

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta
            
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()


def projected_attack(model: Module, loss_fun: Module, test_loader: Iterable, epsilon: float, device, lr: float = 0.01, num_steps: int = 1):
  """
  Attack a trained neural net 
  """
  metrics = []
  losses = []
  # Loop over test set with our dataloaders
  for _, (x, y)in enumerate(test_loader, 1):
    x, y = x.to(device), y.to(device)

    x.requires_grad = True

    # Forward pass
    yhat = model(x)
    loss = loss_fun(yhat, y)

    # Zero all existing gradients
    model.zero_grad()
    loss.backward()

    # Generate an adverserial version of the test data
    step_norm = lr
    eps_norm = epsilon
    x_adverserial = projected_gradient_descent(model, x, y, loss_fun, num_steps, lr, step_norm, epsilon, eps_norm)

    # Re-classify the perturbed batch
    yhat_adv = model(x_adverserial)
    adversarial_accuracy = accuracy(yhat_adv, y)

    metrics.append(adversarial_accuracy)
    losses.append(loss.item())

  # Assuming unchanged Batch
  average_accuracy = sum(metrics) / len(metrics)  

  print("Epsilon: {:.2f}\tTest Accuracy = {:.3f}".format(epsilon, average_accuracy))

  return losses, average_accuracy


def projected_protect(model: Module, optim: Optimizer,loss_fun: Module, train_loader: Iterable, 
            test_loader: Iterable, device, epochs: int = 10, epsilon: float = 0.25, lr: float = 0.01, num_steps: int = 1):
    """
    Protects a model with a chosen optimiser against FGSM.
    """    
    t = time()
    for epoch in range(epochs):
       
        model.train()  # Train an epoch
        for _, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Forward pass for adversarial perturbations
            x.requires_grad = True
            yhat = model(x)

            loss = loss_fun(yhat, y)
            model.zero_grad()
            loss.backward()
            # Attack
            step_norm = lr
            eps_norm = epsilon
            x_adverserial = projected_gradient_descent(model, x, y, loss_fun, num_steps, lr, step_norm, epsilon, eps_norm)
            
            # Evaluate the network (forward pass)
            yhat_adv = model(x_adverserial)
            loss = loss_fun(yhat_adv, y)
            
            # Compute the gradient
            optim.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            # Due to how we specified our Optimizers, this is generic
            optim.step()

        # Test the quality on the test set
        model.eval()
        metrics = []
        for _, (x, y) in enumerate(test_loader, 1):
            x, y = x.to(device), y.to(device)

            # Evaluate the network (forward pass)
            yhat_adv = model(x)
            metrics.append(accuracy(yhat_adv, y))
        
        print("Epoch {:.2f} | Test accuracy: {:.5f}".format(epoch, sum(metrics)/len(metrics)))
    t = time() - t
    print(f'training took {t:.4} s')
    return model

## Taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
# and Lab 10 – Adversarial Robustness(https://colab.research.google.com/drive/1w697nylLw72aFcBEKu7j3yCm6RdpzOi6#scrollTo=eoE7_FDHHkat)


def fgsm(image: Tensor, grad_torch: Tensor, epsilon: float):
    """
    Creates an adverserial version of an inputted image
    """
    # Based on the grad generated by pytorch, get the sign
    grad_sign = grad_torch.sign()
    # Create an adverserial example, by perturbing the image in the direction of the gradient 
    adverserial_image = image + epsilon * grad_sign
    adverserial_image = torch.clamp(adverserial_image, 0, 1)
    # Sent back an adverserial image
    return adverserial_image


def attack(model: Module, loss_fun: Module, test_loader: Iterable, epsilon: float, device):
    """
    Attack a trained neural net
    """
    metrics = []
    losses = []
    # Loop over test set with our dataloaders
    for _, (x, y) in enumerate(test_loader, 1):
        x, y = x.to(device), y.to(device)

        x.requires_grad = True

        # Forward pass
        yhat = model(x)
        loss = loss_fun(yhat, y)

        # Zero all existing gradients
        model.zero_grad()
        loss.backward()

        # Generate an adverserial version of the test data
        x_adverserial = fgsm(x, x.grad, epsilon)

        # Re-classify the perturbed batch
        yhat_adv = model(x_adverserial)
        adversarial_accuracy = accuracy(yhat_adv, y)

        metrics.append(adversarial_accuracy)
        losses.append(loss.item())

    # Assuming unchanged Batch
    average_accuracy = sum(metrics) / len(metrics)

    print("Epsilon: {:.2f}\tTest Accuracy = {:.3f}".format(epsilon, average_accuracy))

    return losses, average_accuracy


def protect(model: Module, optim: Optimizer, loss_fun: Module, train_loader: Iterable,
            test_loader: Iterable, device, epochs: int = 10, epsilon: float = 0.25):
    """
    Protects a model with a chosen optimiser against FGSM.
    """
    t = time()
    for epoch in range(epochs):

        model.train()  # Train an epoch
        for _, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Forward pass for adversarial perturbations
            x.requires_grad = True
            yhat = model(x)

            loss = loss_fun(yhat, y)
            model.zero_grad()
            loss.backward()
            x_adverserial = fgsm(x, x.grad, epsilon)

            # Evaluate the network (forward pass)
            yhat_adv = model(x_adverserial)
            loss = loss_fun(yhat_adv, y)

            # Compute the gradient
            optim.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            # Due to how we specified our Optimizers, this is generic
            optim.step()

        # Test the quality on the test set
        model.eval()
        metrics = []
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader, 1):
                x, y = x.to(device), y.to(device)

                # Evaluate the network (forward pass)
                yhat_adv = model(x)
                metrics.append(accuracy(yhat_adv, y))

        print("Epoch {:.2f} | Test accuracy: {:.5f}".format(epoch, sum(metrics) / len(metrics)))
    t = time() - t
    print(f'training took {t:.4} s')
    return model
