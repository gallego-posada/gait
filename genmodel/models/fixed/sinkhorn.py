import sys

import torch
from torch import nn, optim

from pylego.misc import LinearDecay

from ..basefixed import BaseFixed

sys.path.append('..')
import sinkhorn_pointcloud


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


class Sinkhorn(nn.Module):

    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.z_size = z_size
        self.x_z = Decoder(z_size, h_size, x_size)

    def forward(self, z):
        return self.x_z((z * 2.0) - 1.0)


class SinkhornModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        model = Sinkhorn(28 * 28, flags.h_size, flags.z_size)
        optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate, betas=(flags.beta1, flags.beta2))
        super().__init__(model, flags, optimizer=optimizer, *args, **kwargs)
        uniform = torch.ones(1, flags.batch_size, device=self.device)
        self.uniform = uniform / uniform.sum()

    def loss_function(self, forward_ret, labels=None):
        x_gen = forward_ret
        x = labels.view_as(x_gen)
        D = lambda x, y: sinkhorn_pointcloud.sinkhorn_normalized(x, y, self.flags.sinkhorn_eps, self.uniform,
                                                                 self.uniform, self.flags.batch_size,
                                                                 self.flags.batch_size, self.flags.sinkhorn_iters)
        return D(x, x_gen)
