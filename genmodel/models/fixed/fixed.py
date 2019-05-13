import torch
from torch import nn
from torch.nn import functional as F

from ..basefixed import BaseFixed


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        return torch.sigmoid(self.fc3(t))


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.z_size = z_size
        self.x_z = Decoder(z_size, h_size, x_size)

    def forward(self, z):
        return self.x_z((z * 2.0) - 1.0)


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(Fixed(28 * 28, flags.h_size, flags.z_size), flags, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        x_gen = forward_ret
        x = labels.view_as(x_gen)

        return F.binary_cross_entropy(x_gen, x, reduction='sum') / x.size(0)
