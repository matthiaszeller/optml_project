# optml_project

## About

Impact of Optimisation Algorithm on Adversarial Training of a Neural Net for Image Recognition.

## Project Struture

### Notebooks

* [`Hyperparameter-tuning.ipynb`](Hyperparameter-tuning.ipynb)
* [`Pipeline.ipynb`](Pipeline.ipynb)

### Codebase

* `net.py`: defines the convolutional model for MNIST dataset
* `optimizer.py`: defines the Adam, Nesterov and Minibatch optimizers
* `training.py`: training and test utility functions as well as tuning function for optimizers
* `adversary.py`
    *   fgsm: Function that generates an adversarial image via Fast Gradient Sign Method
    *   attack: Function that uses fgsm to attack a model
    *   protect: Function trains a model using FGSM Adversarial Training
* `data_utils.py`: utilities to load the data

