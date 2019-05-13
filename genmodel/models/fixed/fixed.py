import sys

import torch
from torch import nn, optim
from torch.nn import functional as F

from ..basefixed import BaseFixed

sys.path.append('..')
import renyi


gaussian_kernel = lambda x, y: renyi.generic_kernel(x, y, lambda u, v: renyi.rbf_kernel(u, v, sigmas=[2.5]))


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        return torch.sigmoid(self.fc2(t))


class GeneratorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(GeneratorFC, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return torch.sigmoid(out)


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.z_size = z_size
        # self.x_z = Decoder(z_size, h_size, x_size)
        self.x_z = GeneratorFC(z_size, x_size, [h_size])

    def forward(self, z):
        return self.x_z((z * 2.0) - 1.0)


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        model = Fixed(28 * 28, flags.h_size, flags.z_size)
        kwargs['optimizer'] = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.9))
        super().__init__(model, flags, *args, **kwargs)
        uniform = torch.ones(1, flags.batch_size, device=self.device)
        self.uniform = uniform / uniform.sum()

    def loss_function(self, forward_ret, labels=None):
        x_gen = forward_ret
        x = labels.view_as(x_gen)
        return renyi.mink_mixture_divergence(self.uniform, x, self.uniform, x_gen, gaussian_kernel, 2)
