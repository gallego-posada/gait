import contextlib
import functools

import torch
from torch import autograd, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from pylego import ops

from ..baseadv import BaseAdversarial


class Discriminator(nn.Module):

    def __init__(self, out_size=1, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        self.net = ops.ResNet(1, [(1, 32, 1), (1, 64, 2), (1, 96, 2)], nonlinearity=self.leaky_relu,
                              negative_slope=negative_slope, enable_gain=False)
        self.fc = nn.Linear(128 * 7 * 7, out_size)

        for module in self.modules():
            if hasattr(module, 'weight'):
                spectral_norm(module, n_power_iterations=1)

    def forward(self, x):
        x = self.leaky_relu(self.net(x).view(-1, 128 * 7 * 7))
        return self.fc(x)


class Generator(nn.Module):

    def __init__(self, z_size):
        super().__init__()
        self.fc = nn.Linear(z_size, 128 * 7 * 7)
        self.norm = nn.BatchNorm2d(128)
        self.net = ops.ResNet(128, [(1, 96, 1), (1, 64, -2), (1, 32, -2), (1, 1, 1)], norm=nn.BatchNorm2d,
                              skip_last_norm=True)

    def generate(self, z):
        z = F.elu(self.norm(self.fc(z).view(-1, 128, 7, 7)))
        return self.net(z)

    def forward(self, x, z):
        return torch.tanh(self.generate(z)), (x * 2.0) - 1.0

    def visualize(self, z):
        return torch.sigmoid(self.generate(z))


class AdversarialModel(BaseAdversarial):

    def __init__(self, flags, *args, **kwargs):
        generator = Generator(flags.z_size)
        discriminator = Discriminator()
        super().__init__(flags, generator, discriminator, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        x_gen, x_real = forward_ret
        if self.debug:
            debug_context = autograd.detect_anomaly()
        else:
            debug_context = contextlib.nullcontext()
        with debug_context:
            d_p = self.disc(x_real)
            d_q = self.disc(x_gen)

        if self.train_disc():
            if self.flags.gan_loss == 'bce':
                loss = F.binary_cross_entropy_with_logits(d_p, torch.ones_like(d_p)) + \
                    F.binary_cross_entropy_with_logits(d_q, torch.zeros_like(d_q))
            elif self.flags.gan_loss == 'wgan':
                grad_penalty = self.gradient_penalty(x_real, x_gen, context=debug_context)
                loss = -d_p.mean() + d_q.mean() + (10.0 * grad_penalty) + 1e-3 * (d_p ** 2).mean()
            self.d_loss = loss.item()
        else:
            if self.flags.gan_loss == 'bce':
                loss = F.binary_cross_entropy_with_logits(d_p, torch.zeros_like(d_p)) + \
                    F.binary_cross_entropy_with_logits(d_q, torch.ones_like(d_q))
            elif self.flags.gan_loss == 'wgan':
                loss = d_p.mean() - d_q.mean()
            self.g_loss = loss.item()

        return loss, self.g_loss, self.d_loss
