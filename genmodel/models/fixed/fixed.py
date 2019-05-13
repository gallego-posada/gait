import numpy as np
from scipy import special, stats
import torch
from torch import nn
from torch.nn import functional as F

from pylego import ops

from ..basefixed import BaseFixed


class DBlock(nn.Module):
    """ A basic building block for computing parameters of an isotropic distribution."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logvar = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logvar = self.fc_logvar(t)
        return mu, logvar


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = F.elu(self.fc1(input_))
        t = F.elu(self.fc2(t))
        return t


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.encode = Encoder(x_size, h_size, h_size)
        self.z_x = DBlock(h_size, 2 * z_size, z_size)
        self.x_z = Decoder(z_size, h_size, x_size)

    def forward(self, x):
        encoded_x = self.encode(x.view(x.size(0), -1))
        z_mu, z_logvar = self.z_x(encoded_x)
        eps = torch.randn_like(z_mu)
        z = (eps * torch.exp(0.5 * z_logvar)) + z_mu
        x_recon = self.x_z(z)
        return x_recon, z_mu, z_logvar, eps, z


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(Fixed(28 * 28, flags.h_size, flags.z_size), flags, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        x_recon, z_mu, z_logvar, eps, z = forward_ret
        x = labels.view_as(x_recon)

        bce = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
        bce_optimal = F.binary_cross_entropy(x, x, reduction='sum').detach() / x.size(0)
        bce_diff = bce - bce_optimal

        kld = ops.kl_div_gaussian(z_mu, z_logvar).mean()
        return bce_diff + kld
