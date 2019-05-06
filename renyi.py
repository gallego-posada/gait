import numpy as np
import torch
import utils


def sim_entropy(K, p, alpha=1, normalize=False):
	"""
	Compute similarity sensitive Renyi entropy of a (batch of) distribution(s) p
	over an alphabet of n elements.
	
	Inputs:
		K [n x n tensor] : Positive semi-definite similarity matrix
		p [batch_size x n tensor] : Probability distributions over n elements
		alpha [float] : Divergence order
	
	Output:
		[batch_size x 1 tensor] of entropy for each distribution
    """
	p = p.transpose(0, 1)
	Kp = K @ p
	
	if normalize:
		Kp = Kp / torch.norm(Kp, p=1, dim=0, keepdim=True)
	
	if alpha == 1:
		res = -(p * torch.log(utils.min_clamp(Kp))).sum(dim=0, keepdim=True)
	else:
		Kpa = utils.min_clamp(Kp) ** (alpha - 1)
		v = (p * Kpa).sum(dim=0, keepdim=True) 
		v = torch.log(utils.min_clamp(v)) / (1 - alpha)
		res = v
	
	return res.transpose(0, 1)


def sim_cross_entropy(K, p, q, alpha=1, normalize=False):
	"""
	Compute similarity sensitive Renyi cross entropy of a (batch of) distribution(s) q
	with respect to a (batch of) distribution(s) pover an alphabet of n elements.
	
	Inputs:
		K [n x n tensor] : Positive semi-definite similarity matrix
		p [batch_size x n tensor] : Probability distributions over n elements
		q [batch_size x n tensor] : Probability distributions over n elements
		alpha [float] : Divergence order
	
	Output:
		[batch_size x 1 tensor] i-th entry is cross entropy of i-th row of q w.r.t i-th row of p 
    """
	p = p.transpose(0, 1)
	q = q.transpose(0, 1)
	Kq = K @ q
	
	#print(Kq)
	
	if normalize:
		Kq = Kq / torch.norm(Kq, p=1, dim=0, keepdim=True)
	#print(Kq)
	
	if alpha == 1:
		res = -(p * torch.log(utils.min_clamp(Kq))).sum(dim=0, keepdim=True)
	else:
		Kqa = utils.min_clamp(Kq) ** (alpha - 1)
		v = (p * Kqa).sum(dim=0, keepdim=True) 
		v = torch.log(utils.min_clamp(v)) / (1 - alpha)
		res = v
	
	return res.transpose(0, 1)

def sim_divergence(K, p, q, alpha=1):    
	"""
	Compute similarity sensitive Jensen-Renyi divergence of between a pair of (batches of)
	distribution(s) p and q over an alphabet of n elements.
	
	Inputs:
		K [n x n tensor] : Positive semi-definite similarity matrix
		p [batch_size x n tensor] : Probability distributions over n elements
		q [batch_size x n tensor] : Probability distributions over n elements
		alpha [float] : Divergence order
	
	Output:
		[batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
    """
	p = p.transpose(0, 1)
	q = q.transpose(0, 1)
	
	r = (p + q)/ 2
	
	Kp = K @ p
	Kq = K @ q
	Kr = K @ r
	
	p_by_r = utils.min_clamp(Kp / Kr)
	q_by_r = utils.min_clamp(Kq / Kr)
	
	if alpha == 1:
        
		p_by_r = torch.log(p_by_r)
		q_by_r = torch.log(q_by_r)
		res = -(p * p_by_r).sum(dim=0, keepdim=True) - (q * q_by_r).sum(dim=0, keepdim=True) 
        
	else:
        
		p_by_r = p_by_r ** (alpha - 1)
		q_by_r = q_by_r ** (alpha - 1)

		res = (p * p_by_r).sum(dim=0, keepdim=True) * (q * q_by_r).sum(dim=0, keepdim=True) 
		res = torch.log(utils.min_clamp(res))/(alpha - 1)
     
	return res.transpose(0, 1)


def cosine_similarity(W):
	return torch.cosine_similarity(W[..., None, :], W[..., None, :, :], dim=-1)
    
def poly_kernel(W, degree=2, p=2):
	pdist = torch.norm(W[..., None, :] - W[..., None, :, :], p=p, dim=-1)
	return 1 / (1 + pdist)**degree

def rbf_kernel(W, sigmas=[1.], p=2, d=2):
	pdist = torch.norm(W[..., None, :] - W[..., None, :, :], p=p, dim=-1)
	res = torch.zeros_like(pdist)
	for sigma in sigmas:
		res += torch.exp(- pdist**d / (sigma**2))
	return res / len(sigmas)

def generic_kernel(X, Y, kernel_fn):
	W = torch.cat((X, Y))
	K = kernel_fn(W)
	return K

def mixture_divergence(mu, X, nu, Y, alpha, kernel):
	n, m = mu.shape[1], nu.shape[1]

	muT = mu.transpose(0, 1)
	nuT = nu.transpose(0, 1)
	
	# Compute similarity matrix
	K = kernel(X, Y)
	
	#v = (mu * (utils.min_clamp(v) **(alpha-1))).sum(dim=0, keepdim=True)
	u = (2 * K[:n, :n] @ muT) / utils.min_clamp(K[:n, :n] @ muT + K[:n, -m:] @ nuT)
	v = (2 * K[-m:, -m:] @ nuT) / utils.min_clamp(K[-m:, -m:] @ nuT + K[-m:, :n] @ muT)
		
	if alpha == 1:
		
		u = -mu @ torch.log(utils.min_clamp(u))
		v = -nu @ torch.log(utils.min_clamp(v))
		
		return (u + v) , K
	
	else:
        
		u = mu @ (utils.min_clamp(u) **(alpha-1))
		u = torch.log(utils.min_clamp(u))

		v = nu @ (utils.min_clamp(v) **(alpha-1)) 
		v = torch.log(utils.min_clamp(v))
    
		return (u + v) / (alpha - 1), K
	

def conv_divergence(mu, nu, alpha, kernel, Knu=None):

	Kmu = kernel(mu)

	if Knu is None:
		Knu = kernel(nu)

	Kr = (Kmu + Knu) / 2
	u = Kmu / Kr
	v = Knu / Kr

	if alpha == 1:

		u = torch.log(utils.min_clamp(u))
		u = -(mu * u).sum(-1).sum(-1, keepdim=True)

		v = torch.log(utils.min_clamp(v))
		v = -(nu * v).sum(-1).sum(-1, keepdim=True)

		return (u + v)

	else:

		u = (mu * u**(alpha-1)).sum(-1).sum(-1, keepdim=True)
		u = torch.log(u)

		v = (nu * v**(alpha-1)).sum(-1).sum(-1, keepdim=True)
		v = torch.log(v)

		return (u + v) / (alpha - 1)
