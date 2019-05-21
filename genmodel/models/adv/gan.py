import contextlib

import torch
from torch import autograd
from torch.nn import functional as F

from .arch import Discriminator, Generator
from ..baseadv import BaseAdversarial


class AdversarialModel(BaseAdversarial):

    def __init__(self, flags, *args, **kwargs):
        generator = Generator(flags.z_size, flags.h_size)
        discriminator = Discriminator(flags.h_size)
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
