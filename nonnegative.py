import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils


class Model(nn.Module):

    def __init__(self, size=4, alpha=1, n=100, d=2, sigma=1, half_empty=True):
        super().__init__()
        self.alpha = alpha
        self.d = d
        self.sigma = sigma
        self.size = size
        self.half_empty = half_empty

        self.locs = nn.Parameter(torch.randn(size, n))
        if half_empty:
            self.p = nn.Parameter(torch.randn(1, size // 2) * 5.0)
        else:
            self.p = nn.Parameter(torch.randn(1, size) * 5.0)
        self.q = nn.Parameter(torch.randn(1, size) * 5.0)

    def forward(self):
        log_K = -(utils.batch_pdist(self.locs, self.locs) / self.sigma) ** self.d
        if self.half_empty:
            p = torch.cat([F.softmax(self.p, dim=1), torch.zeros_like(self.p)], dim=1)
        else:
            p = F.softmax(self.p, dim=1)
        q = F.softmax(self.q, dim=1)

        log_pK = torch.logsumexp(log_K[None, ...] + torch.log(p[:, None, :]), dim=2)
        log_qK = torch.logsumexp(log_K[None, ...] + torch.log(q[:, None, :]), dim=2)

        rat1 = (log_pK, log_qK)
        rat2 = (log_qK, log_pK)

        if np.abs(self.alpha - 1.0) < 1e-8:
            dp1 = (p * (rat1[0] - rat1[1])).sum(-1)
            dp2 = (q * (rat2[0] - rat2[1])).sum(-1)
            loss = 0.5 * (dp1 + dp2)
        else:
            power_pq = torch.log(p) + (self.alpha - 1) * (rat1[0] - rat1[1])
            power_qp = torch.log(q) + (self.alpha - 1) * (rat2[0] - rat2[1])
            loss = 0.5 * (1 / (self.alpha - 1)) * (torch.logsumexp(power_pq, -1) + torch.logsumexp(power_qp, -1))

        return loss  #, p, q, torch.exp(log_K)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    losses = []
    for n in [100, 5]:
        for half_empty in [False, True]:
            for size in [1000, 10]:
                for alpha in [0.5, 1, 2]:
                    local_losses = []
                    print('n', n, 'alpha', alpha, 'size', size, 'half_empty', half_empty)
                    model = Model(alpha=alpha, size=size, half_empty=half_empty, n=n).cuda()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

                    for itr in range(10000):
                        optimizer.zero_grad()
                        loss = model()
                        if itr % 100 == 0:
                            print(loss)
                            local_losses.append(loss.item())
                        loss.backward()
                        optimizer.step()

                    losses.append((half_empty, size, alpha, n, local_losses))
                    print()

    with open('losses_%d.pk' % np.random.randint(1000), 'wb') as f:
        pickle.dump(losses, f)
