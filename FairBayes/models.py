import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden_units):
        super(Classifier, self).__init__()
        layers = []
        
        if n_layers == 1: # Logistic Regression
            layers.append(nn.Linear(n_inputs, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(n_inputs, n_hidden_units))
            layers.append(nn.ReLU())
            for i in range(n_layers-2):
                layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden_units,1))
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
                
    def forward(self, x):
        x = self.layers(x)
        return x