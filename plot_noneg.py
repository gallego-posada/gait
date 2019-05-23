import collections
import glob
import pickle

import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Lucida Grande']
from matplotlib import pyplot as plt

colors = ['#d11414', '#cebc31', '#4fa51a', '#18a56f', '#18a8af', '#1765b7', '#3421dd', '#8221dd', '#c91cbd']


def plot(fnames, name):
    plots = collections.defaultdict(list)
    for fname in fnames:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        for half_empty, size, alpha, n, local_losses in data:
            plots[(half_empty, size, alpha, n)].append(local_losses)

    x = (np.arange(1, 101) * 100).astype(np.int)

    for k, v in plots.items():
        half_empty, size, alpha, n = k
        if n == 3:
            continue

    fig = plt.figure(figsize=(8, 4.75))
    ax = fig.gca()
    ct = 0
    for k, v in plots.items():
        half_empty, size, alpha, n = k
        if n == 3:
            continue
        v = np.array(v)
        y_min = np.min(v, axis=0)
        y_max = np.max(v, axis=0)

        if half_empty:
            support_str = 'H.S.'
            linestyle = '--'
        else:
            support_str = 'F.S.'
            linestyle = '-'
        plt.plot(x, y_min, color=colors[ct], label=r'$\alpha=%.1f$, $n=%d$, %s' % (alpha, size, support_str),
                 linestyle=linestyle)
        plt.fill_between(x, y_min, y_max, color=colors[ct], alpha=0.08, linewidth=0)
        ct += 1
        if ct >= len(colors):
            ct = 0
    plt.plot(x, np.zeros(x.shape), color='k', alpha=0.75,linewidth=0.5)

    ax.set_xlim([100, 10000])
    ax.set_ylim([-1, 15.0])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.xlabel('Adam iteration')
    plt.ylabel(r'$\mathbb{D}^{\mathbf{K}}_{\alpha}$')
    # plt.show()
    plt.savefig('%s.png' % name, bbox_inches='tight', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    loss_files = glob.glob('losses_*.pk')
    fixed_files = [f for f in loss_files if '_fixed' in f]
    nofixed_files = [f for f in loss_files if not '_fixed' in f]

    plot(nofixed_files, 'nofixed')
    plot(fixed_files, 'fixed')
