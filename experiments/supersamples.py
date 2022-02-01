import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Lucida Grande']
rcParams['font.size'] = 12

import torch
import numpy as np

import sys
sys.path.append(".")

import gait_src as gait
import gait_src.helpers.mog_data as mog_data

# For generating GIF
import glob
from PIL import Image


def generate_mog_data():

    p = [3, 6, 5, 6, 6, 5, 1]
    p = np.array(p) / np.sum(p)

    mus = [[-3.5, -1], [0.5, 0.5], [-2, -2], [2.5, 3],
        [-1, 2], [1, -2], [0, 0]]

    covs = [[[0.2,0.2], [0.2, 1.5]], [[1, 0], [0, 1]],
            [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]],
            [[1, 1.2], [1.2, 3]], [[0.8, 0.5], [0.5, 1]],
            [[3, 0], [0, 3]]]

    x, y = np.mgrid[-5:4:.05, -4:5:.05]
    xy_grid = (x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    rvs = mog_data.create_rvs(p, mus, covs)

    return p, xy_grid, pos, rvs


def plot_supersamples(centroids, p, xy_grid, pos, rvs, show_real_samples=False,
                      save_name="", save_path="results/supersamples/", show_fig=False):

    plt.figure(figsize=(5,5))
    cset = plt.contour(*xy_grid, mog_data.eval_mix_pdf(pos, rvs, p), 50,
                       linewidths=0.7, cmap=cm.ocean_r, antialiased=True)
    plt.grid(False)

    if show_real_samples:
        real_samples = mog_data.sample_mix(rvs, p, 100)
        plt.scatter(real_samples[:, 0], real_samples[:, 1],
                    marker='o', c=(0, 0.9, 0), label='Real Samples',
                    alpha=0.4, zorder=5)

    np_centroids = centroids.data.numpy()
    plt.scatter(np_centroids[:, 0],np_centroids[:, 1],
                s=list(2000*q.data.squeeze().numpy()),
                label='Atomic Approximation', marker='*',
                c='r', alpha=1, zorder=10)

    plt.xlim([-4.5, 4])
    plt.ylim([-4, 4.5])
    plt.axis('off')
    # plt.legend()

    if save_name != "":
        plt.savefig(save_path + save_name, dpi=100)

    if show_fig:
        plt.show()
    else:
        plt.close()


def generate_gif(save_name, save_path, gif_name):
    fp_in =  save_path + save_name + "*.png"
    fp_out = "results/{}.gif".format(gif_name)

    img_paths = glob.glob(fp_in)
    # Remove .png (4 chars) from end of path name
    sorted_paths = sorted(img_paths, key = lambda k: int(k.split('_')[1][:-4]))
    img, *imgs = [Image.open(f) for f in sorted_paths]

    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, loop=0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='GAIT - Supersamples')
    parser.add_argument('--save_gif', action='store_true')
    
    parser.add_argument('--save_path', default="results/supersamples/", type=str)
    parser.add_argument('--save_name', default="supersamples_", type=str)
    parser.add_argument('--gif_name', default="anim_supersamples", type=str)
    
    return parser.parse_args()

if __name__=="__main__":

    args =  parse_arguments()
    
    p, xy_grid, pos, rvs = generate_mog_data()

    # Use a polynomial kernel of degree 1.5
    kernel_fn = lambda _u, _v: gait.kernels.poly_kernel(_u, _v, degree=1.5, c=2.)
    kernel = partial(gait.kernels.generic_kernel, kernel_fn=kernel_fn)

    # n = Number of approximating mixture components
    # m = Size of fresh batch samples
    # d = Dimension of the data
    n, m, d = 200, 200, 2

    # Uniform mixture components used for empirical minibatches below
    tp = torch.Tensor(np.ones((1, m))/m)
    
    # These are the parameters
    # Using a uniform mixture for the approximating distribution
    logit_q = torch.nn.Parameter(0*torch.Tensor(np.random.rand(1, n))).requires_grad_(False)
    q = torch.softmax(logit_q, dim=1)

    # Initialize the centroids randomly
    centroids = torch.nn.Parameter(torch.randn(n, d))
    optim_list = [torch.optim.Adam([centroids], lr=0.01, amsgrad=True)] 

    for iter_num in range(100):

        [o.zero_grad() for o in optim_list]

        # Sample fresh minibatch of real data
        Y = torch.Tensor(mog_data.sample_mix(rvs, p, m))

        loss = gait.breg_mixture_divergence(tp, Y, q, centroids, kernel)

        if iter_num % 10 == 0:
            print("%d - %.4f" % (iter_num, loss.item()))
            
            # Generate plot
            if args.save_gif:
                plot_supersamples(centroids, p, xy_grid, pos, rvs, 
                                show_real_samples=False,
                                save_name=args.save_name + str(iter_num),
                                save_path=args.save_path)

        loss.backward()
        [o.step() for o in optim_list]
        
    if args.save_gif:
        generate_gif(args.save_name, args.save_path, args.gif_name)

