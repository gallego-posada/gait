import os
import sys
sys.path.append(".")
from functools import partial

from plot_style import *

from time import time
import numpy as np
import pandas as pd

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import gait_src as gait


def load_dataset(dataset_name):
    if dataset_name == "mnist":
        dataset_path = '../data/mnist_data'
        dataset_class = torchvision.datasets.MNIST

    dataset =  dataset_class(dataset_path, train=True, download=True,)
    loader = torch.utils.data.DataLoader(dataset)

    return dataset, loader

def show_real_samples(dataset, num_classes=10, img_size=28):
    toshow = []
    for digit in range(0, num_classes):
        ix = dataset.targets == digit
        D = dataset.data[ix].view(-1, img_size, img_size).float()
        toshow.append(D[np.random.choice(D.shape[0])])

    plt.figure(figsize=(num_classes, 2))
    plt.imshow(torch.cat(toshow, dim=1).data.numpy(), cmap='gray_r')
    plt.axis('off')
    plt.show()


def compute_barycenters(dataset_name, dataset, loader, show_partial=False):

    barycenters = []

    for digit in range(0, 10):

        # Get indices for current class
        ix = dataset.targets == digit

        if dataset_name == 'emnist':
            D = dataset.train_data[ix].view(-1, 28, 28).transpose(-1, -2).float()
        else:
            D = dataset.data[ix].view(-1, 28, 28).float()

        # Uniform initialization for weights
        logit_q = torch.nn.Parameter(torch.zeros(1, img_size*img_size))
        log_temp = torch.nn.Parameter(torch.tensor(-1.))

        optimizer = torch.optim.Adam([logit_q, log_temp], lr=0.05)

        start = time()
        
        for i in range(500):
            optimizer.zero_grad()

            # Sample minibatch of 32 images from current class
            p = gait.utils.sample_and_resize(D[0:-1, ...], img_size, 32)

            # Compute distribution corresponding to current approximation of the barycenter
            q = torch.softmax(logit_q / torch.exp(log_temp), dim=1).view(1, img_size, img_size)

            loss = gait.bregman_sim_divergence(img_kernel, p, q).mean()

            if i % 100 == 0:
                print("%d - %d : %.3e" % (digit, i, loss.item()))
                if show_partial:
                    q = torch.softmax(logit_q / torch.exp(log_temp), dim=1).view(img_size, img_size).data
                    plt.figure(figsize=(1, 1))
                    plt.imshow(q, cmap='gray_r');
                    plt.show()

            loss.backward()
            optimizer.step()

        q = torch.softmax(logit_q /  torch.exp(log_temp), dim=1).view(img_size, img_size).data
        barycenters.append(q)

        print('Class time: ', time()-start)

    return barycenters


def show_barycenters(barycenters, save_name="", save_path="results/barycenters/", show_fig=False):

    def normalize_image(img): return (img - img.min()) / (img.max() - img.min())

    plt.figure(figsize=(1.1*len(barycenters), 2.5))
    plt.imshow(torch.cat( [normalize_image(_) for _ in barycenters], dim=1), cmap='gray_r')
    plt.axis('off')

    if save_name != "":
        plt.savefig(save_path + save_name, dpi=600)

    if show_fig:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":

    # Reproduce Fig. 9 in GAIT: A Geometric Approach to Information Theory
    # https://arxiv.org/pdf/1906.08325.pdf

    img_size = 28
    # Precompute kernel over spatial (pixel locations) domain
    [Ygrid, Xgrid] = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    
    
    # Kmat = 1 / (1 + 10. * np.abs(Xgrid - Ygrid))**2.5
    Kmat = np.exp(-np.abs(Xgrid - Ygrid)**2/(0.05**2))
    Kmat = torch.tensor(Kmat)
    img_kernel = lambda x: torch.matmul(torch.matmul(Kmat, x), Kmat)

    dataset_name = "mnist"
    dataset, loader = load_dataset(dataset_name)

    # Uncomment line below to generate plot with real examples from each class
    #show_real_samples(dataset, num_classes=10, img_size=img_size)

    barycenters = compute_barycenters(dataset_name, dataset, loader,
                                      show_partial=False)

    show_barycenters(barycenters,
                     save_name=dataset_name + "_barycenters",
                     save_path="results/barycenters/",
                     show_fig=False)

