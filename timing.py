import time

import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import gait

bs, img_size = 100, 128  # TODO loop with batch sizes and produce a plot
print(f'Considering a batch of {bs} images of size {img_size}x{img_size}.')

[Ygrid, Xgrid] = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
Kmat = np.exp(-np.abs(Xgrid - Ygrid)**2/(0.05**2))
Kmat = torch.tensor(Kmat)

img_kernel = lambda x: torch.matmul(torch.matmul(Kmat, x), Kmat)

p = torch.softmax(torch.rand(bs, img_size * img_size), dim=1).view(bs, img_size, img_size)
q = torch.softmax(torch.rand(bs, img_size * img_size), dim=1).view(bs, img_size, img_size)
before = time.monotonic()
with torch.no_grad():
    gait.breg_sim_divergence(img_kernel, p, q)
after = time.monotonic()
print('Seconds taken for computing GAIT divergence:', after - before)
