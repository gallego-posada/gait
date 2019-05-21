import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):

    def __init__(self, hidden_size, out_size=1, negative_slope=0.01, attn=False):
        super().__init__()
        self.attn = attn
        self.out_size = out_size
        if attn:
            self.weights = nn.Parameter(torch.zeros(1, 28 * 28))
        if out_size > 0:
            if hidden_size <= 0:
                self.fc = nn.Linear(28 * 28, out_size)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(28 * 28, hidden_size),
                    nn.LeakyReLU(negative_slope),
                    nn.Linear(hidden_size, out_size)
                )
            for module in self.modules():
                if hasattr(module, 'weight'):
                    spectral_norm(module, n_power_iterations=2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.attn:
            x = x * torch.sqrt(28 * 28 * F.softmax(self.weights, dim=1))
        if self.out_size > 0:
            x = self.fc(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 28 * 28)
        )

    def generate(self, z):
        return self.fc((z * 2.0) - 1.0).view(-1, 1, 28, 28)

    def forward(self, x, z):
        return torch.tanh(self.generate(z)), (x * 2.0) - 1.0

    def visualize(self, z):
        return torch.sigmoid(self.generate(z))
