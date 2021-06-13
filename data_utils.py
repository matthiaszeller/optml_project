from pathlib import Path
from typing import Tuple
import json

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(normalize=True, path='data'):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    train = torchvision.datasets.MNIST(str(path), download=True, train=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST(str(path), train=False, transform=torchvision.transforms.ToTensor())

    xtrain = train.data.to(torch.float32)
    xtest = test.data.to(torch.float32)

    if normalize:
        xtrain = xtrain.view(-1, 28 * 28)
        xtest = xtest.view(-1, 28 * 28)
        mu, std = xtrain.mean(0), xtrain.std(0)

        xtrain -= mu
        xtest -= mu
        xtrain[:, std > 0] /= std[std > 0]
        xtest[:, std > 0] /= std[std > 0]

        xtrain = xtrain.view(-1, 28, 28)
        xtest = xtest.view(-1, 28, 28)

    train.data = xtrain
    test.data = xtest

    return train, test


def build_data_loaders(train_dataset, test_dataset, batch_size) -> Tuple[DataLoader, DataLoader]:
    # Add an axis to have one channel
    train_loader = DataLoader(TensorDataset(train_dataset.data[:, None, :], train_dataset.targets), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_dataset.data[:, None, :], test_dataset.targets), batch_size=batch_size)

    return train_loader, test_loader


def get_height_weight_gender(sub_sample=True, add_outlier=False):
    """Load height-weight-gender data and convert it to the metric system."""
    path_dataset = "data/height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]
        gender = gender[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender

def get_best_hyperparams(fp, epsilon = 1e-8):
    """Get best set of hyperparameters for any optimizer."""
    if Path(fp).exists():
        with open(fp, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError("File path does not exist!")

    best_res = {'metric_train': (0.0,0.0), 'metric_test' : 0.0}
    final_res = dict()
    if 'adam' in fp:
        mean = [np.mean(results['metric_train']['0']), np.mean(results['metric_train']['1']), np.mean(results['metric_train']['2'])]
        std = [np.std(np.mean(results['metric_train']['0'],axis=0))+ epsilon, np.std(np.mean(results['metric_train']['1'],axis=0))+ epsilon, np.std(np.mean(results['metric_train']['2'],axis=0))+ epsilon]

        mean_test = [np.std(results['metric_test']['0']), np.std(results['metric_test']['1']), np.std(results['metric_test']['2'])]
        std_test = [np.std(results['metric_test']['0'])+ epsilon, np.std(results['metric_test']['1'])+ epsilon, np.std(results['metric_test']['2'])+ epsilon]
 
        for i in range(1,3):
            if (mean_test[i]/std_test[i]) > best_res['metric_test']:
                best_res['metric_train'] = (mean[i],  std[i])
                best_res['metric_test'] = (mean_test[i]/std_test[i]) 
                for key, val in results.items():
                    if str(i) in val.keys():
                        final_res[key] = val[str(i)]
                    

    else:
        for i in range(len(results)):
            mean_train =  np.mean(results[i]['metric_train'], axis=1)
            mean_train_final = np.mean(mean_train)
            std_train=  np.std(mean_train) + epsilon

            
            mean_test =  np.mean(results[i]['metric_test'])
            std_test =  np.std(mean_test) + epsilon
           
            if (mean_test/std_test) > best_res['metric_test']:

                best_res['metric_train'] = (mean_train_final,std_train)
                best_res['metric_test'] = (mean_test/std_test)
                for key, val in results[i].items():
                    final_res[key] = val
                        
                    

    return final_res



