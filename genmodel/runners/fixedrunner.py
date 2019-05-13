import collections

from pylego import misc

from models.basefixed import BaseFixed
from .basemnist import MNISTBaseRunner


class FixedRunner(MNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseFixed, ['loss', 'bce_diff', 'kld', 'bce_optimal'])

    def run_batch(self, batch, train=False):
        x = self.model.prepare_batch(batch[0])
        loss, bce_diff, kld, bce_optimal = self.model.run_loss(x, labels=x)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item()),
                                        ('bce_diff', bce_diff.item()),
                                        ('kld', kld.item()),
                                        ('bce_optimal', bce_optimal.item())])

    def post_epoch_visualize(self, epoch, split):
        if split == 'train':
            return
        print('* Visualizing', split)
        batch = next(self.reader.iter_batches(split, 16 * 8, shuffle=True, partial_batching=True, threads=self.threads,
                                              max_batches=1))
        x = self.model.prepare_batch(batch[0])
        x_recon = self.model.run_batch([x])[0].view_as(x).detach().cpu().numpy()
        x = x.cpu().numpy()
        if split == 'test':
            fname = self.flags.log_dir + '/test.png'
        else:
            fname = self.flags.log_dir + '/val%03d.png' % epoch
        misc.save_comparison_grid(fname, x, x_recon, border_shade=1.0)
        print('* Visualizations saved to', fname)
