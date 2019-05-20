import collections

import numpy as np
import torch

from pylego import misc

from models.baseadv import BaseAdversarial
from .basemnist import MNISTBaseRunner


class AdversarialRunner(MNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseAdversarial, ['g_loss', 'd_loss'])

    def run_batch(self, batch, train=False):
        z = torch.rand(self.batch_size, self.flags.z_size)
        z, x = self.model.prepare_batch([z, batch[0]])
        loss, g_loss, d_loss = self.model.run_loss([x, z], labels=x)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        assert not np.isnan(g_loss)
        assert not np.isnan(d_loss)
        return collections.OrderedDict([('g_loss', g_loss), ('d_loss', d_loss)])

    def post_epoch_visualize(self, epoch, split):
        print('* Visualizing', split)
        Z = torch.linspace(0.0, 1.0, steps=20)
        Z = torch.cartesian_prod(Z, Z).view(20, 20, 2)
        x_gens = []
        for row in range(20):
            if self.flags.z_size == 2:
                z = Z[row]
            else:
                z = torch.rand(20, self.flags.z_size)
            z = self.model.prepare_batch(z)
            x_gen = self.model.run_batch([z], visualize=True).detach().cpu()
            x_gens.append(x_gen)

        x_full = torch.cat(x_gens, dim=0).numpy()
        if split == 'test':
            fname = self.flags.log_dir + '/test.png'
        else:
            fname = self.flags.log_dir + '/val%03d.png' % epoch
        misc.save_comparison_grid(fname, x_full, border_width=0, retain_sequence=True)
        print('* Visualizations saved to', fname)
