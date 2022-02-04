import os
import sys
sys.path.append(".")
from functools import partial

from plot_style import *

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch

import gait_src as gait


def load_france_data(num_centroids: int = 30):
    # num_centroids: Size of support for approximating measure
    
    df = pd.read_csv("./data/france/pop_fr_geoloc_1975_2010.csv")
    X = np.array([df.long.values, df.lat.values]).T

    # Normalize data coordinates, ensuring values of ~[0, 1]
    X = X / np.array([10., 50.])
                 
    # We normalize the data for a scale around -1 to 1 for better behavior in gradient descent
    # tX = torch.tensor([X[:, 0] / 10,  X[:, 1] / 50]).transpose(0, 1)

    # Proportions according to population
    target_props = (df.pop_2010.values[None, :] / df.pop_2010.values.sum()).squeeze()
    
    # import pdb; pdb.set_trace()
    
    # fix_mu = torch.tensor(df.pop_2010.values[None, :] / df.pop_2010.values.sum())

    # Sample initialization. Will be used for both methods 
    rix = np.random.choice(len(X), num_centroids, p=target_props)
    initX = X[rix, :]

    return X, target_props, initX


def run_kmeans(X, target_props, initX=None, num_centroids=None):
    
    if initX is not None:
        num_centroids = len(initX)
        kmeans = MiniBatchKMeans(n_clusters=num_centroids, 
                                random_state=0,
                                init=initX).fit(X)
    else:
        kmeans = MiniBatchKMeans(n_clusters=num_centroids).fit(X)

    # Make assignments from real points to nearest neighbor in approximation
    assignments = np.zeros((len(X), kmeans.n_clusters))
    assignments[range(len(X)), kmeans.labels_] = 1
    # Compute propotion of population associated with each centroid
    centroid_weights = (assignments * target_props[:, np.newaxis]).sum(0)
    
    return kmeans.cluster_centers_, centroid_weights


def run_gait_approx(X, target_props, initX, batch_size:int=None):
    
    num_centroids = len(initX)
    
    if batch_size is None:
        batch_size = min(100, num_centroids)

    # We use a Monte Carlo estimate of the real distribution by getting a 'batch' of samples
    mu = gait.utils.uniform_distribution(batch_size, use_torch=True)
    
    # Force a uniform distribution for the approximating measure
    nu = gait.utils.uniform_distribution(num_centroids, use_torch=True)
    
    # Location parameters to be trained
    Y = torch.nn.Parameter(torch.tensor(initX))
    optimizer = torch.optim.SGD([Y], lr=5e-2, momentum=0.7)
    
    kernel_fn = lambda _u, _v: gait.kernels.poly_kernel(_u, _v, degree=1.5, c=1.)
    kernel = partial(gait.kernels.generic_kernel, kernel_fn=kernel_fn)

    divergence_fn = lambda x, y: gait.bregman__mixture_divergence(mu, x, nu, y, kernel)
    # divergence_fn = lambda x, y: gait.bregman__mixture_divergence_stable(mu, x, nu, y, kernel)
    
    for iter_num in range(2000):
    
        optimizer.zero_grad()
        
        # Random indices for minibatch of points
        rix = np.random.choice(len(target_props), batch_size, p=target_props)
        # Compare (random estimate of the) target measure and approximating measure
        
        # import pdb; pdb.set_trace()
        loss = divergence_fn(X[rix], Y) 
        
        if iter_num % 250 == 0:
            print("%d - %.4f" % (iter_num, loss.item()))

        loss.backward()
        optimizer.step()


    # Make assignments from real points to nearest centroid in approximation
    dists = gait.utils.batch_pdist(Y, X) 
    assignments = np.zeros((len(X), num_centroids))
    assignments[range(len(X)), torch.argmin(dists, dim=0).data.numpy()] = 1
    # Compute propotion of population associated with each centroid
    centroid_weights = (assignments * target_props.data.numpy()[:, np.newaxis]).sum(0)
    
    return Y, centroid_weights


def generate_plots(kmeans_locs, kmeans_weights, gait_locs, gait_weights, 
                   save_name="", save_path="results/france/", show_fig=False):
    
    
    assert len(kmeans_weights) == len(gait_weights)
    num_centroids = len(kmeans_weights)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), 
                                 gridspec_kw={'height_ratios': [1, 6]})

    ax1.bar(range(1, num_centroids+1), -100*np.sort(-kmeans_weights),
            color=(238/255,100/255,98/255), alpha=0.99, align='center')
    ax1.bar(range(1, num_centroids+1), -100*np.sort(-gait_weights),
            color=(15/255,157/255,88/255), alpha=0.4, align='center')
    ax1.set_ylabel('% pts')
    ax1.set_xticks([1, num_centroids/2, num_centroids])

    ax2.scatter(X[:, 0], X[:, 1], s=list(1000*target_props), label='Target',
                marker='.', facecolors=(66/255, 134/255, 244/255), alpha=0.9)
    ax2.scatter(kmeans_locs[:, 0], kmeans_locs[:, 1], s=2500*kmeans_weights, label='K-Means',
                marker='X', alpha=0.9, edgecolors=(220/255,65/255,40/255),
                facecolors=(238/255,119/255,98/255), linewidths=0.99);
    ax2.scatter(gait_locs[:, 0], gait_locs[:, 1], s=2500*gait_weights,
                label='Ours', marker='o', facecolors=(15/255,157/255,88/255),
                edgecolors="w", linewidths=1.2, alpha=0.7)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.52, 0.02), markerscale=None,
               fancybox=True, shadow=True, ncol=3, fontsize=11)
    ax2.axis('off')

    plt.subplots_adjust(hspace=0.2)

    if save_name != "":
        plt.savefig(save_path + save_name, dpi=800)

    if show_fig:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    
    # Reproduce Fig. 6 in GAIT: A Geometric Approach to Information Theory
    # https://arxiv.org/pdf/1906.08325.pdf
    
    num_centroids = 30
    X, target_props, initX = load_france_data(num_centroids=num_centroids)
    
    kmeans_locs, kmeans_weights = run_kmeans(X, target_props, initX=initX,
                                             num_centroids=num_centroids)
    
    in_tensors = [torch.tensor(_) for _ in [X, target_props, initX]]
    gait_locs, gait_weights = run_gait_approx(*in_tensors)
    
    generate_plots(kmeans_locs, kmeans_weights, 
                   gait_locs.data.numpy(), gait_weights, 
                   save_name="france_data_model", save_path="results/france/")    
