import sys

import torch
from torch import nn, optim

from pylego.misc import LinearDecay
from pylego import ops

from ..basefixed import BaseFixed

sys.path.append('..')
import renyi


def gaussian_kernel(sigma):
    return lambda x, y: renyi.generic_kernel(x, y, lambda u, v: renyi.rbf_kernel(u, v, sigmas=[sigma], log=True))


def poly_kernel(degree):
    return lambda x, y: renyi.generic_kernel(x, y, lambda u, v: renyi.poly_kernel(u, v, degree=degree, log=True))


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size, layers=2, ngf=64):
        super().__init__()
        if layers == 1:  # VAE setup
            self.fc = nn.Sequential(
                nn.Linear(z_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, output_size)
            )
        elif layers == 2:  # Sinkhorn setup
            self.fc = nn.Sequential(
                nn.Linear(z_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        elif layers == 3:  # CIFAR10
            self.fc = nn.Sequential(
                ops.View(-1, z_size, 1, 1),
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z_size, ngf * 8, 4, 1, 0),
                nn.BatchNorm2d(ngf * 8),
                nn.LeakyReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1),
                # state size. (nc) x 32 x 32
                ops.View(-1, 3 * 32 * 32)
            )

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size, layers=2):
        super().__init__()
        self.z_size = z_size
        self.x_z = Decoder(z_size, h_size, x_size, layers=layers)

    def forward(self, z):
        return self.x_z((z * 2.0) - 1.0)


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        model = Fixed(28 * 28, flags.h_size, flags.z_size, layers=flags.layers)
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
        D = lambda x, y: renyi.breg_mixture_divergence_stable(self.uniform, x, self.uniform, y, self.kernel,
                                                              symmetric=self.flags.symmetric)
        if not self.flags.unbiased:
            return D(x, x_gen)
        else:
            return 2 * D(x, x_gen) - D(x_gen, x_gen)
