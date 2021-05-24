from torch._C import StringType
import torchvision
import numpy as np

from typing import Optional, Callable
from typing import Iterable, Callable

import torch
from torch.functional import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from data_utils import get_mnist, build_data_loaders
from net import Net
from training import testing
from training import training, accuracy
import matplotlib.pyplot as plt


## Taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
# and Lab 10 – Adversarial Robustness(https://colab.research.google.com/drive/1w697nylLw72aFcBEKu7j3yCm6RdpzOi6#scrollTo=eoE7_FDHHkat)

def fgsm(image: Tensor, data_grad: Tensor, update_max_norm: float):
    """
    Compute the FGSM update on an image (or a batch of images)
    """
    # Based on the 
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + update_max_norm*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



def attack(model: Module, criterion: Module, test_loader: Iterable, update_max_norm: float, device):
  """
  Executes an FGSM attack on a selected model.
  @return (
    accuracy of the model in the perturbed test set,
    adversarial example images: list of 5 samples of a tuple (original prediction - float, prediction - float, example image - torch.tensor)
  )
  """
  accuracy_per_batch = []
  adversarial_examples = []  # a small sample of 5 adversarial images

  # Loop over all examples in test set in batches
  for batch_id, (data, target)in enumerate(test_loader, 1):
    data, target = data.to(device), target.to(device)

    # Indicate that we want PyTorch to compute a gradient with respect to the
    # input batch.
    data.requires_grad = True

    # Forward pass
    output = model(data)
    original_predictions = output.argmax(1) # get the index of the max logit
    original_accuracy = accuracy(output, target)
    loss = criterion(output, target)

    # Zero all existing gradients
    model.zero_grad()
    loss.backward()

    # Perturb the batch with a gradient step (using the `fgsm_update`)
    perturbed_data = fgsm(data, data.grad, update_max_norm)

    # Re-classify the perturbed batch
    output = model(perturbed_data)
    adversarial_predictions = output.argmax(1)
    adversarial_accuracy = accuracy(output, target)

    accuracy_per_batch.append(adversarial_accuracy)

    # Save some adversarial examples for visualization
    if len(adversarial_examples) < 5:
      adv_ex = perturbed_data[0, 0, :, :].detach().cpu().numpy()
      adversarial_examples.append( (original_predictions[0].item(), adversarial_predictions[0].item(), adv_ex) )

  average_accuracy = sum(accuracy_per_batch) / len(accuracy_per_batch)  # assuming all batches are the same size

  print("Epsilon: {:.2f}\tTest Accuracy = {:.3f}".format(update_max_norm, average_accuracy))

  return average_accuracy, adversarial_examples


def protect(model: Module, optim: Optimizer,criterion: Module, train_loader: Iterable, 
            test_loader: Iterable, device, epochs: int = 10):
    """
    Protects a model with a chosen optimiser against FGSM.
    """    
    
    for epoch in range(epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass for adversarial perturbations
            batch_x.requires_grad = True
            output = model(batch_x)
            original_predictions = output.argmax(1) # get the index of the max logit
            original_accuracy = accuracy(output, batch_y)
            loss = criterion(output, batch_y)
            model.zero_grad()
            loss.backward()
            perturbed_data = fgsm(batch_x, batch_x.grad, 0.25)
            
            # Evaluate the network (forward pass)
            prediction = model(perturbed_data)
            loss = criterion(prediction, batch_y)
            
            # Compute the gradient
            optim.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optim.step()

        # Test the quality on the test set
        model.eval()
        accuracies = []
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies.append(accuracy(prediction, batch_y))
        
        print("Epoch {:.2f} | Test accuracy: {:.5f}".format(epoch, sum(accuracies).item()/len(accuracies)))
    
    return model