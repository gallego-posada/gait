import numpy as np
from scipy import special, stats
import torch
from torch import nn
from torch.nn import functional as F

from pylego import ops

from ..basefixed import BaseFixed


def di_log_prob(x):
    n = x.size(1)
    eta = 3.25 * n
    gamma_1_eta = special.gamma(1.0 / eta)
    beta = np.sqrt((n * gamma_1_eta) / special.gamma(3.0 / eta))
    r = torch.norm(x, p=2, dim=1, keepdim=True)
    log_pr = -((r / beta) ** eta) + np.log(eta / (beta + gamma_1_eta))
    # FIXME 1e-45 being added to the log term:
    log_px = log_pr - (n - 1) * torch.log(r + 1e-45) - np.log((2.0 * (np.pi ** (0.5 * n))) / special.gamma(0.5 * n))
    return log_px


def di_sample_like(z):
    n = z.size(1)
    eta = 3.25 * n
    beta = np.sqrt((n * special.gamma(1.0 / eta)) / special.gamma(3.0 / eta))
    r = np.abs(stats.gennorm.rvs(eta, size=(z.size(0), 1)) * beta)
    pre_v = torch.randn_like(z)
    v = pre_v / torch.norm(pre_v, p=2, dim=1, keepdim=True)
    return v * torch.as_tensor(r, dtype=v.dtype, device=v.device)


class DBlock(nn.Module):
    """ A basic building block for computing parameters of an isotropic distribution."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logvar = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logvar = self.fc_logvar(t)
        return mu, logvar


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = F.elu(self.fc1(input_))
        t = F.elu(self.fc2(t))
        return t


class Decoder(nn.Module):

    def __init__(self, z_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class Fixed(nn.Module):

    def __init__(self, x_size, h_size, z_size, q_dist):
        super().__init__()
        self.q_dist = q_dist

        self.encode = Encoder(x_size, h_size, h_size)
        self.z_x = DBlock(h_size, 2 * z_size, z_size)
        self.x_z = Decoder(z_size, h_size, x_size)

    def forward(self, x):
        encoded_x = self.encode(x.view(x.size(0), -1))
        z_mu, z_logvar = self.z_x(encoded_x)
        if self.q_dist == 'gaussian':
            eps = torch.randn_like(z_mu)
        elif self.q_dist == 'di':
            eps = di_sample_like(z_mu)
        z = (eps * torch.exp(0.5 * z_logvar)) + z_mu
        x_recon = self.x_z(z)

        return x_recon, z_mu, z_logvar, eps, z


class FixedModel(BaseFixed):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(Fixed(28 * 28, flags.h_size, flags.z_size, flags.q_dist), flags, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        x_recon, z_mu, z_logvar, eps, z = forward_ret
        x = labels.view_as(x_recon)

        bce = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
        bce_optimal = F.binary_cross_entropy(x, x, reduction='sum').detach() / x.size(0)
        bce_diff = bce - bce_optimal

        if self.flags.q_dist == 'gaussian' and self.flags.p_dist == 'gaussian' and not self.flags.sampled_kl:
            kld = ops.kl_div_gaussian(z_mu, z_logvar).mean()
        else:
            zero = z_mu.detach() * 0.0
            if self.flags.q_dist == 'gaussian':
                log_qz = ops.gaussian_log_prob(zero, zero, eps)
            elif self.flags.q_dist == 'di':
                log_qz = di_log_prob(eps)
            log_qz = log_qz - 0.5 * z_logvar.sum(dim=-1)  # adjust from g(y) = f(x) |det(dx/dy)|
            if self.flags.p_dist == 'gaussian':
                log_pz = ops.gaussian_log_prob(zero, zero, z)
            elif self.flags.p_dist == 'di':
                log_pz = di_log_prob(z)
            kld = (log_qz - log_pz).mean()

        loss = bce_diff + self.flags.kl_weight * kld

        return loss, bce_diff, kld, bce_optimal
