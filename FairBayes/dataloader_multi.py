import os
import copy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tempeh.configurations import datasets
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder, StandardScaler


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(
        device), torch.FloatTensor(XZ).to(device)


def adult(data_root, display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_train_data = pd.read_csv(
        data_root + 'adult.data',
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    raw_test_data = pd.read_csv(
        data_root + 'adult.test',
        skiprows=1,
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    train_data = raw_train_data.drop(["Education"], axis=1)  # redundant with Education-Num
    test_data = raw_test_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    train_data["Target"] = train_data["Target"] == " >50K"
    test_data["Target"] = test_data["Target"] == " >50K."
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                train_data[k] = np.array([rcode[v.strip()] for v in train_data[k]])
                test_data[k] = np.array([rcode[v.strip()] for v in test_data[k]])
            else:
                train_data[k] = train_data[k].cat.codes
                test_data[k] = test_data[k].cat.codes

    return train_data.drop(["Target", "fnlwgt"], axis=1), train_data["Target"].values, test_data.drop(
        ["Target", "fnlwgt"], axis=1), test_data["Target"].values


class CustomDataset():
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessDataset_multi():
    def __init__(self, dataset, device=torch.device('cuda')):
        self.dataset = dataset
        self.device = device
        np.random.seed(12345678)

        if self.dataset == 'AdultCensus':
            self.get_adult_data()

        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
        self.prepare_ndarray()

    def get_adult_data(self):
        X_train, Y_train, X_test, Y_test = adult('./data/adult/')

        self.Z1_train_ = X_train['Sex']
        self.Z1_test_ = X_test['Sex']
        self.Z2_train_ = X_train['Race']==4
        self.Z2_test_ = X_test['Race']==4
        self.Z_train_ = self.Z1_train_ * 2 + self.Z2_train_
        self.Z_test_ = self.Z1_test_ * 2 + self.Z2_test_

        X_train['Race'] = X_train['Race']==4
        X_test['Race'] = X_test['Race']==4

        self.XZ_train_ = X_train
        self.XZ_test_ = X_test
        X_train = X_train.drop(labels=['Sex'], axis=1)
        X_train = X_train.drop(labels=['Race'], axis=1)
        X_test = X_test.drop(labels=['Sex'], axis=1)
        X_test = X_test.drop(labels=['Race'], axis=1)

        self.X_train_ = X_train
        self.X_test_ = X_test
        self.XZ_train_ = pd.get_dummies(self.XZ_train_)
        self.XZ_test_ = pd.get_dummies(self.XZ_test_)
        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(Y_train)
        self.Y_train_ = pd.Series(self.Y_train_, name='>50k')
        self.Y_test_ = le.fit_transform(Y_test)
        self.Y_test_ = pd.Series(self.Y_test_, name='>50k')


    def prepare_ndarray(self):
        self.normalized = False
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = self.XZ_train_.to_numpy(dtype=np.float64)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = self.XZ_test_.to_numpy(dtype=np.float64)
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def normalize(self):
        self.normalized = True
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)

        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_test = scaler_X.transform(self.X_test)
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train), \
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_), \
               (X_test_, Y_test_, Z_test_, XZ_test_)