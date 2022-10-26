import numpy as np

import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnHeadModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, self.config.hidden_size)),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.head(state)


class ActorModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(config.hidden_size, config.hidden_size)),
            nn.ReLU()
        )

    def forward(self, state):
        return self.model(state)


class CriticModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(config.hidden_size, config.hidden_size)),
            nn.ReLU()
        )

    def forward(self, state):
        return self.model(state)
