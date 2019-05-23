import sys

import torch
from torch import nn, optim

from pylego.misc import LinearDecay

from ..basefixed import BaseFixed

sys.path.append('..')
import renyi


def gaussian_kernel(sigma):
    return lambda x, y: renyi.generic_kernel(x, y, lambda u, v: renyi.rbf_kernel(u, v, sigmas=[sigma], log=True))


def poly_kernel(degree):
    return lambda x, y: renyi.generic_kernel(x, y, lambda u, v: renyi.poly_kernel(u, v, degree=degree, log=True))


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size):
        super().__init__()
        self.z_size = z_size
        self.x_z = Decoder(z_size, h_size, x_size)

    def forward(self, z):
        return self.x_z((z * 2.0) - 1.0)


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        model = Fixed(28 * 28, flags.h_size, flags.z_size)
        optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate, betas=(flags.beta1, flags.beta2))
        super().__init__(model, flags, optimizer=optimizer, *args, **kwargs)
        uniform = torch.ones(1, self.flags.batch_size, device=self.device)
        self.uniform = uniform / uniform.sum()
        if self.flags.kernel == 'gaussian':
            self.sigma_decay = LinearDecay(flags.sigma_decay_start, flags.sigma_decay_end, flags.kernel_initial_sigma,
                                           flags.kernel_sigma)
        elif self.flags.kernel == 'poly':
            self.kernel = poly_kernel(self.flags.kernel_degree)

    def loss_function(self, forward_ret, labels=None):
        x_gen = forward_ret
        x = labels.view_as(x_gen)
        if self.flags.kernel == 'gaussian':
            sigma = self.sigma_decay.get_y(self.get_train_steps())
            self.kernel = gaussian_kernel(sigma)
        D = lambda x, y: renyi.renyi_mixture_divergence_stable(self.uniform, x, self.uniform, y, self.kernel,
                                                               self.flags.alpha, use_full=self.flags.use_full,
                                                               use_avg=self.flags.use_avg,
                                                               symmetric=self.flags.symmetric)
        if not self.flags.unbiased:
            return D(x, x_gen)
        else:
            return 2 * D(x, x_gen) - D(x_gen, x_gen)
