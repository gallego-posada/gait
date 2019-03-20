import numpy as np
import torch
EPS = 1e-10

def min_clamp(x, c=EPS):
	return torch.clamp_min(x, c)

def from_numpy(x, requires_grad=False):
	x = torch.from_numpy(x)
	x.requires_grad_(requires_grad)
	return x