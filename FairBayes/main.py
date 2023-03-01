import time
import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from models import Classifier
from dataloader import FairnessDataset
from algorithm import FairBayes
from algorithm_multi import FairBayes_multi
from dataloader_multi import FairnessDataset_multi


#####multi-class protected attribute or not##########
multi_class = True


##### Model specifications #####
n_layers = 2 # [positive integers]
n_hidden_units = 32 # [positive integers]


##### Which dataset to test and which fairness notion to consider#####

if multi_class == True:

    dataset_name = 'AdultCensus'

else:
    dataset_name = 'AdultCensus'  # ['AdultCensus',  'COMPAS','Lawschool']
    fairness = 'DemPa' # ['DemPa', 'EqqOp']

##### Other training hyperparameters #####
if dataset_name == 'AdultCensus':
    n_epochs = 200
    lr = 1e-1
    batch_size = 512
    ##### predetermine disparity level #####

    delta_set_dp = np.arange(0, 50, 1) / 200
    delta_set_eo = np.arange(0, 50, 1) / 250

if dataset_name == 'COMPAS':
    n_epochs = 500
    lr = 5e-4
    batch_size = 2048
    ##### predetermine disparity level #####

    delta_set_dp = np.arange(0, 50, 1) / 150
    delta_set_eo = np.arange(0, 50, 1) / 140


if dataset_name == 'Lawschool':
    n_epochs = 200
    lr = 2e-4
    batch_size = 2048
    ##### predetermine disparity level #####

    delta_set_dp = np.arange(0, 50, 1) / 500
    delta_set_eo = np.arange(0, 50, 1) / 340
##### Whether to enable GPU training or not
device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu' )

n_seeds = 20  # Number of random seeds to try




result = pd.DataFrame()
starting_time = time.time()

for seed in range(n_seeds):
    print('Currently working on - seed: {}'.format(seed))
    seed = seed * 5
    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    if multi_class == True:
        dataset = FairnessDataset_multi(dataset=dataset_name, device=device)
        dataset.normalize()
        input_dim = dataset.XZ_train.shape[1]
    else:
        dataset = FairnessDataset(dataset=dataset_name, device=device)
        dataset.normalize()
        input_dim = dataset.XZ_train.shape[1]


    # Create a classifier model
    net = Classifier(n_layers=n_layers, n_inputs=input_dim, n_hidden_units=n_hidden_units)
    net = net.to(device)

    # Set an optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Fair classifier training
    if multi_class==True:
        temp = FairBayes_multi(dataset=dataset,dataset_name=dataset_name,
                                 net=net,
                                 optimizer=optimizer,
                                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)

    else:
        temp = FairBayes(dataset=dataset,dataset_name=dataset_name,
                                 net=net,
                                 optimizer=optimizer,
                                 fairness=fairness,  delta_set=delta_set_eo,
                                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    result = result.append(temp)
    print(result)
print('Average running time: {:.3f}s'.format((time.time() - starting_time) / n_seeds))