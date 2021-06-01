# optml_project

## About

## Project Struture

* net.py
* minibatch.py
    *   class MiniBatchOptimizer: the implemented Minibatch SGD optimiser
* training.py
* adversary.py
    *   fgsm: Function that generates an adversarial image via Fast Gradient Sign Method
    *   attack: Function that uses fgsm to attack a model
    *   protect: Function trains a model using FGSM Adversarial Training
* data_utils.py
