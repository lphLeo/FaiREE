import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import CustomDataset
import sys


from utils import threshold_DemPa_multi
from utils import measures_from_Yhat_DemPa_multi


def FairBayes_multi(dataset, dataset_name, net, optimizer, device, n_epochs=200,
              batch_size=2048, seed=0):
    # Retrieve train/test splitted pytorch tensors for index=split
    train_val_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train_val, Y_train_val, Z_train_val, XZ_train_val = train_val_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    # training data size and validation data size

    train_val_size = len(X_train_val)
    Y_train, Y_val = torch.split(Y_train_val, int(train_val_size * 0.8))
    Z_train, Z_val = torch.split(Z_train_val, int(train_val_size * 0.8))
    XZ_train, XZ_val = torch.split(XZ_train_val, int(train_val_size * 0.8))
    Y_val_np = Y_val.detach().cpu().numpy()
    # Retrieve train/test splitted numpy arrays for index=split
    Z_train_np = Z_train.detach().cpu().numpy()
    Z_list = sorted(list(set(Z_train_np)))
    for z in Z_list:
        if (Z_train_np==z).sum()==0:
            print('At least one sensitive group has no data point')
            sys.exit()
    Z_test_np = Z_test.detach().cpu().numpy()
    Y_test_np = Y_test.detach().cpu().numpy()
    Y1_train_np = (Y_train[Z_train == 1]).detach().cpu().numpy()
    Y0_train_np = (Y_train[Z_train == 0]).detach().cpu().numpy()

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

    # An empty dataframe for logging experimental results

    loss_function = nn.BCELoss()
    costs = []
    for epoch in range(n_epochs):
        net.train()
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = net(xz_batch)
            cost = 0

            # prediction loss
            cost = loss_function(Yhat.squeeze(), y_batch)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            costs.append(cost.item())

            # Print the cost per 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch + 1, n_epochs,
                                                                          i + 1, len(data_loader),
                                                                          cost.item()), end='\r')

        ########choose the model with best performance on validation set###########
        with torch.no_grad():

            output_val = net(XZ_val.to(device))
            Yhat_val = (output_val > 0.5).detach().cpu().numpy()
            accuracy = (Yhat_val == Y_val_np).mean()

            if epoch == 0:
                accuracy_max = accuracy
                bestnet_acc_stat_dict = net.state_dict()

            if accuracy > accuracy_max:
                accuracy_max = accuracy
                bestnet_acc_stat_dict = net.state_dict()

        #########Calculate thresholds for fair Bayes-optimal Classifier###########
    net.load_state_dict(bestnet_acc_stat_dict)

    eta_train = net(XZ_train).squeeze().detach().cpu().numpy()

    eta_test = net(XZ_test).squeeze().detach().cpu().numpy()

    t_star = threshold_DemPa_multi(eta_train, Z_train_np)

    df_test = measures_from_Yhat_DemPa_multi(eta_test, Z_test_np,Y_test_np , t_star)
    df_test['seed']=seed
    return df_test