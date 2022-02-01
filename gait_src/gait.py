import numpy as np
import torch
from torch.nn import functional as F
from . import utils


def gait_sim_entropy(K, p, alpha=1):
    """
    Compute similarity sensitive GAIT entropy of a (batch of) distribution(s) p
    over an alphabet of n elements.
    
    Inputs:
        K [n x n tensor] : Positive semi-definite similarity matrix
        p [batch_size x n tensor] : Probability distributions over n elements
        alpha [float] : Divergence order
    
    Output:
        [batch_size x 1 tensor] of entropy for each distribution
    """
    pK = p @ K

    if np.allclose(alpha, 1.0):
        ent = -(p * torch.log(pK)).sum(dim=-1)
    else:
        Kpa = pK ** (alpha - 1)
        v = (p * Kpa).sum(dim=-1, keepdim=True)
        ent = torch.log(v) / (1 - alpha)

    return ent


def gait_sim_entropy_stable(log_K, p, alpha=1):
    """
    Compute similarity sensitive GAIT entropy of a (batch of) distribution(s) p
    over an alphabet of n elements.
    
    Inputs:
        log_K [n x n tensor] : Log of positive semi-definite similarity matrix
        p [batch_size x n tensor] : Probability distributions over n elements
        alpha [float] : Divergence order
    
    Output:
        [batch_size x 1 tensor] of entropy for each distribution
    """
    log_pK = torch.logsumexp(log_K[None, ...] + torch.log(p[:, None, :]), dim=2)

    if np.allclose(alpha, 1.0):
        ent = -(p * log_pK).sum(dim=-1)
    else:
        log_Kpa = log_pK * (alpha - 1)
        log_v = torch.logsumexp(torch.log(p) + log_Kpa, dim=-1, keepdim=True)
        ent = log_v / (1 - alpha)

    return ent


def sim_cross_entropy(K, p, q, alpha=1, normalize=False):
    """
    TODO: this is not mathematically correct!!
    
    Compute similarity sensitive gait cross entropy of a (batch of) distribution(s) q
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
    
    
    if normalize:
        Kq = Kq / torch.norm(Kq, p=1, dim=0, keepdim=True)
    
    if alpha == 1:
        res = -(p * torch.log(utils.min_clamp(Kq))).sum(dim=0, keepdim=True)
    else:
        Kqa = utils.min_clamp(Kq) ** (alpha - 1)
        v = (p * Kqa).sum(dim=0, keepdim=True) 
        v = torch.log(utils.min_clamp(v)) / (1 - alpha)
        res = v
    
    return res.transpose(0, 1)


def bregman_sim_divergence(K, p, q, symmetric=False):
    # NOTE: if you make changes in this function, do them in *_stable function under this as well.
    """
    Compute similarity sensitive Bregman divergence of between a pair of (batches of)
    distribution(s) p and q over an alphabet of n elements.    Inputs:
       p [batch_size x n tensor] : Probability distributions over n elements
       q [batch_size x n tensor] : Probability distributions over n elements
       K [n x n tensor or callable] : Positive semi-definite similarity matrix or function
       symmetric [boolean]: Use symmetrized Bregman divergence.
    Output:
       div [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q
    """
    if symmetric:
        r = (p + q) / 2.
    if callable(K):
        pK = K(p)
        qK = K(q)
        if symmetric:
            rK = K(r)
    else:
        pK = p @ K
        qK = q @ K
        if symmetric:
            rK = r @ K
    if symmetric:
        rat1 = (pK, rK)
        rat2 = (qK, rK)
    else:
        rat1 = (pK, qK)

    if callable(K):  # we're dealing with an image
        sum_dims = (-2, -1)
    else:
        sum_dims = -1

    if symmetric:
        t1 = (p * (torch.log(rat1[0]) - torch.log(rat1[1]))).sum(sum_dims)
        t2 = (r * (rat1[0] / rat1[1])).sum(sum_dims)
        t3 = (q * (torch.log(rat2[0]) - torch.log(rat2[1]))).sum(sum_dims)
        t4 = (r * (rat2[0] / rat2[1])).sum(sum_dims)
        return (2 + t1 - t2 + t3 - t4) / 2.
    else:
        t1 = (p * (torch.log(rat1[0]) - torch.log(rat1[1]))).sum(sum_dims)
        t2 = (q * (rat1[0] / rat1[1])).sum(sum_dims)
        return 1 + t1 - t2


def bregman_sim_divergence_stable(log_K, p, q, symmetric=False):
    """
    Compute similarity sensitive Bregman divergence of between a pair of (batches of)
    distribution(s) p and q over an alphabet of n elements.    Inputs:
       p [batch_size x n tensor] : Probability distributions over n elements
       q [batch_size x n tensor] : Probability distributions over n elements
       log_K [n x n tensor or callable] : Log of positive semi-definite similarity matrix or function
       symmetric [boolean]: Use symmetrized Bregman divergence.
    Output:
       div [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q
    """
    if symmetric:
        r = (p + q) / 2.
    if callable(log_K):
        log_pK = log_K(p)
        log_qK = log_K(q)
        if symmetric:
            log_rK = log_K(r)
    else:
        log_pK = torch.logsumexp(log_K[None, ...] + torch.log(p[:, None, :]), dim=2)
        log_qK = torch.logsumexp(log_K[None, ...] + torch.log(q[:, None, :]), dim=2)
        if symmetric:
            log_rK = torch.logsumexp(log_K[None, ...] + torch.log(r[:, None, :]), dim=2)
    if symmetric:
        rat1 = (log_pK, log_rK)
        rat2 = (log_qK, log_rK)
    else:
        rat1 = (log_pK, log_qK)

    if callable(log_K):  # we're dealing with an image
        sum_dims = (-2, -1)
    else:
        sum_dims = -1

    if symmetric:
        t1 = (p * (rat1[0] - rat1[1])).sum(sum_dims)
        t2 = (r * torch.exp(rat1[0] - rat1[1])).sum(sum_dims)
        t3 = (q * (rat2[0] - rat2[1])).sum(sum_dims)
        t4 = (r * torch.exp(rat2[0] - rat2[1])).sum(sum_dims)
        return (2 + t1 - t2 + t3 - t4) / 2.
    else:
        t1 = (p * (rat1[0] - rat1[1])).sum(sum_dims)
        t2 = (q * torch.exp(rat1[0] - rat1[1])).sum(sum_dims)
        return 1 + t1 - t2


def breg_mixture_divergence(p, Y, q, X, kernel, symmetric=False):
    # NOTE: if you make changes in this function, do them in *_stable function under this as well.
    """
    Compute similarity sensitive GAIT divergence of between a pair of empirical distributions
    p and q with supports Y and X, respectively
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        kernel [callable] : Function to compute the kernel matrix
        symmetric [boolean] : Use the symmetric version of the divergence
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """

    Kyy = kernel(Y, Y)
    Kyx = kernel(Y, X)
    Kxx = kernel(X, X)

    f_p = torch.cat([p, torch.zeros_like(q)], -1)
    f_q = torch.cat([torch.zeros_like(p), q], -1)
    f_K = torch.cat([torch.cat([Kyy, Kyx], 1), torch.cat([Kyx.t(), Kxx], 1)], 0)

    return bregman_sim_divergence(f_K, f_p, f_q, symmetric=symmetric)


def breg_mixture_divergence_stable(p, Y, q, X, log_kernel, symmetric=False):
    """
    Compute similarity sensitive GAIT divergence of between a pair of empirical distributions
    p and q with supports Y and X, respectively
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        log_kernel [callable] : Function to compute the log kernel matrix
        symmetric [boolean] : Use the symmetric version of the divergence
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """

    log_Kyy = log_kernel(Y, Y)
    log_Kyx = log_kernel(Y, X)
    log_Kxx = log_kernel(X, X)

    f_p = torch.cat([p, torch.zeros_like(q)], -1)
    f_q = torch.cat([torch.zeros_like(p), q], -1)
    f_log_K = torch.cat([torch.cat([log_Kyy, log_Kyx], 1), torch.cat([log_Kyx.t(), log_Kxx], 1)], 0)

    return bregman_sim_divergence_stable(f_log_K, f_p, f_q, symmetric=symmetric)


def test_mixture_divergence(p, Y, q, X, log_kernel, symmetric=False, use_avg=False):
    """
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        log_kernel [callable] : Function to compute the log kernel matrix
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """
    log_Kyy = log_kernel(Y, Y)
    log_Kyx = log_kernel(Y, X)
    log_Kxx = log_kernel(X, X)

    log_Kyy_p = torch.logsumexp(log_Kyy + torch.log(p), dim=1, keepdim=True).transpose(0, 1)
    log_Kxy_p = torch.logsumexp(log_Kyx.transpose(0, 1) + torch.log(p), dim=1, keepdim=True).transpose(0, 1)
    log_Kyx_q = torch.logsumexp(log_Kyx + torch.log(q), dim=1, keepdim=True).transpose(0, 1)
    log_Kxx_q = torch.logsumexp(log_Kxx + torch.log(q), dim=1, keepdim=True).transpose(0, 1)

    log_K = torch.cat([torch.cat([log_Kyy, log_Kyx], dim=1), torch.cat([log_Kyx.transpose(0, 1), log_Kxx], dim=1)],
                      dim=0)
    T = F.softmax(log_K, dim=1)

    log_Kp = torch.cat([log_Kyy_p, log_Kxy_p], dim=1)
    log_Kq = torch.cat([log_Kyx_q, log_Kxx_q], dim=1)

    if use_avg:
        log_Kp_Kq = torch.logsumexp(torch.stack([log_Kp, log_Kq]), 0)
        rat1 = (np.log(2) + log_Kp, log_Kp_Kq)
        rat2 = (np.log(2) + log_Kq, log_Kp_Kq)
    else:
        rat1 = (log_Kp, log_Kq)
        rat2 = (log_Kq, log_Kp)

    div = p @ (T * (rat1[0] - rat1[1])).sum(dim=1, keepdim=True)[:p.size(1)]
    if symmetric:
        div = div + q @ (T * (rat2[0] - rat2[1])).sum(dim=1, keepdim=True)[p.size(1):]

    return div
