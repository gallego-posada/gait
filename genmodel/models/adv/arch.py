import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from pylego import ops


class Discriminator(nn.Module):

    def __init__(self, out_size=1, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        self.net = ops.ResNet(1, [(1, 32, 1), (1, 64, 2), (1, 96, 2)], nonlinearity=self.leaky_relu,
                              negative_slope=negative_slope, enable_gain=False)
        self.fc = nn.Linear(96 * 7 * 7, out_size)

        for module in self.modules():
            if hasattr(module, 'weight'):
                spectral_norm(module, n_power_iterations=1)

    def forward(self, x):
        x = self.leaky_relu(self.net(x).view(-1, 96 * 7 * 7))
        return self.fc(x)


class Generator(nn.Module):

    def __init__(self, z_size):
        super().__init__()
        self.fc = nn.Linear(z_size, 96 * 7 * 7)
        self.norm = nn.BatchNorm2d(96)
        self.net = ops.ResNet(96, [(1, 96, 1), (1, 64, -2), (1, 32, -2), (1, 1, 1)], norm=nn.BatchNorm2d,
                              skip_last_norm=True)

    def generate(self, z):
        z = F.elu(self.norm(self.fc(z).view(-1, 96, 7, 7)))
        return self.net(z)

    def forward(self, x, z):
        return torch.tanh(self.generate(z)), (x * 2.0) - 1.0

    def visualize(self, z):
        return torch.sigmoid(self.generate(z))
