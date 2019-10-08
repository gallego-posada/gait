import collections
import glob
import datetime

import numpy as np
import torch
from torch.distributions import normal

from pylego import misc

from models.basefixed import BaseFixed
from .basemnist import MNISTBaseRunner


class FixedRunner(MNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseFixed)
        if flags.data == 'cifar10':
            self.img_chan = 3
            self.img_size = 32
        else:
            self.img_chan = 1
            self.img_size = 28

    def run_batch(self, batch, train=False):
        if self.flags.normal_latent:
            z = torch.randn(self.batch_size, self.flags.z_size)
        else:
            z = torch.rand(self.batch_size, self.flags.z_size)
        z, x = self.model.prepare_batch([z, batch[0]])
        if self.flags.double_precision:
            x = x.to(torch.float64)
        loss = self.model.run_loss(z, labels=x)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item())])

    def post_epoch_visualize(self, epoch, split):
        # if self.flags.visualize_only:
        #     self.do_plots()
        # else:
        if True:
            print('* Visualizing', split)
            Z = torch.linspace(0.0 + 1e-3, 1.0 - 1e-3, steps=20)
            Z = torch.cartesian_prod(Z, Z).view(20, 20, 2)
            if self.flags.z_size == 2 and self.flags.normal_latent:
                dist = normal.Normal(0.0, 1.0)
            x_gens = []
            for row in range(20):
                if self.flags.z_size == 2:
                    z = Z[row]
                    if self.flags.normal_latent:
                        z = dist.icdf(z)
                else:
                    if self.flags.normal_latent:
                        z = torch.randn(20, self.flags.z_size)
                    else:
                        z = torch.rand(20, self.flags.z_size)
                z = self.model.prepare_batch(z)
                x_gen = self.model.run_batch([z]).view(20, self.img_chan, self.img_size, self.img_size).detach().cpu()
                x_gens.append(x_gen)

            x_full = torch.cat(x_gens, dim=0).numpy()
            if split == 'test':
                fname = self.flags.log_dir + '/test.png'
            else:
                fname = self.flags.log_dir + '/vis_%03d.png' % self.model.get_train_steps()
            misc.save_comparison_grid(fname, x_full, border_width=0, retain_sequence=True)
            print('* Visualizations saved to', fname)

    # def do_plots(self, N=4, max_step=100000, rows=4):
    #     from matplotlib import rcParams
    #     rcParams['font.family'] = 'serif'
    #     rcParams['font.sans-serif'] = ['Lucida Grande']
    #     import matplotlib.pyplot as plt
    #     from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    #     import tensorflow as tf

    #     np.random.seed(9)
    #     indices = []
    #     for _ in range(rows):
    #         row = []
    #         for __ in range(rows):
    #             row.append((np.random.randint(20), np.random.randint(20)))
    #         indices.append(row)

    #     def getImage(path):
    #         img = plt.imread(path)
    #         data = []
    #         for row in indices:
    #             row_data = []
    #             for idx in row:
    #                 row_data.append(img[self.img_size*idx[0]:self.img_size*(idx[0]+1),
    #                                     self.img_size*idx[1]:self.img_size*(idx[1]+1)])
    #             data.append(np.concatenate(row_data, axis=1))
    #         rowscols = np.concatenate(data, axis=0)
    #         rowscols = np.repeat(rowscols[..., None], 3, axis=2)
    #         return OffsetImage(rowscols)

    #     log_file = sorted(glob.glob(self.flags.log_dir + '/summary/*'))[0]
    #     timestamps = {}
    #     for e in tf.train.summary_iterator(log_file):
    #         if e.step > max_step:
    #             break
    #         timestamps[e.step] = e.wall_time
    #     start_time = datetime.datetime.fromtimestamp(timestamps[1])

    #     existing = glob.glob(self.flags.log_dir + '/vis_*.png')
    #     pairs = [(f.rsplit('_', 1)[-1].split('.', 1)[0], f) for f in existing]
    #     pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()])

    #     x = []
    #     times = []
    #     for step, _ in pairs:
    #         if step in timestamps:
    #             times.append(str(datetime.datetime.fromtimestamp(timestamps[step]) - start_time).rsplit('.', 1)[0])
    #             x.append(step)
    #     paths = [path for _, path in pairs[:len(x)]]

    #     idx = np.round(((np.linspace(0, len(x) - 1, N) / (len(x) - 1)) ** 3.2) * (len(x) - 1)).astype(int)
    #     x = np.array(x)[idx]
    #     y = [0 for i in range(len(x))]
    #     paths = np.array(paths)[idx]
    #     times = np.array(times)[idx]
    #     print(x)
    #     print(times)

    #     fig, ax = plt.subplots(figsize=(5,1.5))
    #     ax.scatter(x, y)

    #     for x0, y0, path, ts in zip(x, y, paths, times):
    #         ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    #         # ax.text(x0, y0 - self.img_size, ts, horizontalalignment='left', verticalalignment='center')
    #         ax.add_artist(ab)

    #     plt.xscale('log')
    #     ax.set_ylim([-14, 30 * len(x)])

    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     # ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.axes.get_yaxis().set_visible(False)
    #     plt.show()
