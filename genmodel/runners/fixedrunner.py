import collections

import torch

from pylego import misc

from models.basefixed import BaseFixed
from .basemnist import MNISTBaseRunner


class FixedRunner(MNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseFixed, ['loss'])

    def run_batch(self, batch, train=False):
        z = torch.rand(self.batch_size, self.flags.z_size)
        z, x = self.model.prepare_batch([z, batch[0]])
        loss = self.model.run_loss(z, labels=x)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item())])

    def post_epoch_visualize(self, epoch, split):
        if split == 'train':
            return
        print('* Visualizing', split)
        Z = torch.linspace(0.0, 1.0, steps=20)
        Z = torch.cartesian_prod(Z, Z).view(20, 20, 2)
        x_gens = []
        for row in range(20):
            z = self.model.prepare_batch(Z[row])
            x_gen = self.model.run_batch([z]).view(20, 1, 28, 28).detach().cpu()
            x_gens.append(x_gen)

        x_full = torch.cat(x_gens, dim=0).numpy()
        if split == 'test':
            fname = self.flags.log_dir + '/test.png'
        else:
            fname = self.flags.log_dir + '/val%03d.png' % epoch
        misc.save_comparison_grid(fname, x_full, border_width=0, retain_sequence=True)
        print('* Visualizations saved to', fname)
