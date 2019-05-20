import contextlib

import torch
from torch import autograd, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from pylego import ops

from ..baseadv import BaseAdversarial


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 4, padding=1)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(64, 128, 3, stride=2)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(128, 256, 3, stride=2)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(256, 1, 2))
        )

    def forward(self, x):
        return self.fc(x).view(x.size(0), -1)


class Generator(nn.Module):

    def __init__(self, z_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_size, 256 * 3 * 3),
            ops.View(-1, 256, 3, 3),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 1, 4, padding=1)
        )

    def forward(self, x, z):
        return torch.tanh(self.fc(z)), (x * 2.0) - 1.0

    def visualize(self, z):
        return torch.sigmoid(self.fc(z))


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
            loss = F.binary_cross_entropy_with_logits(d_p, torch.ones_like(d_p)) + \
                F.binary_cross_entropy_with_logits(d_q, torch.zeros_like(d_q))
            self.d_loss = loss.item()
        else:
            loss = F.binary_cross_entropy_with_logits(d_p, torch.zeros_like(d_p)) + \
                F.binary_cross_entropy_with_logits(d_q, torch.ones_like(d_q))
            self.g_loss = loss.item()

        return loss, self.g_loss, self.d_loss
