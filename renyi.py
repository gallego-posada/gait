import numpy as np
import torch
from torch.nn import functional as F

import utils


def renyi_sim_entropy(K, p, alpha=1):
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
    pK = p @ K

    if np.allclose(alpha, 1.0):
        ent = -(p * torch.log(pK)).sum(dim=-1)
    else:
        Kpa = pK ** (alpha - 1)
        v = (p * Kpa).sum(dim=-1, keepdim=True)
        ent = torch.log(v) / (1 - alpha)

    return ent


def renyi_sim_entropy_stable(log_K, p, alpha=1):
    """
    Compute similarity sensitive Renyi entropy of a (batch of) distribution(s) p
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

    
    
def mink_sim_divergence(K, p, q, alpha=2, use_inv=False, use_avg=False):
    """
    Compute similarity sensitive Minkowski divergence of between a pair of (batches of)
    distribution(s) p and q over an alphabet of n elements.

    Inputs:
        p [batch_size x n tensor] : Probability distributions over n elements
        q [batch_size x n tensor] : Probability distributions over n elements
        K [n x n tensor or callable] : Positive semi-definite similarity matrix or function
        alpha [float] : Divergence order
        use_inv [boolean] : Determines whether to compare similarity or dissimilarity profiles
    Output:
        div [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
    """
    
    if callable(K):
        pK = K(p)
        qK = K(q)
    else:
        pK = p @ K
        qK = q @ K
    
    if use_inv:
        pK = 1 / utils.min_clamp(pK)
        qK = 1 / utils.min_clamp(qK)
        
    diff = torch.abs(pK - qK) ** alpha
    
    if use_avg:
        exp_measure = (p+q)/2
    else:
        exp_measure = p
        
    div = (exp_measure * diff).sum(dim=-1) 
    if callable(K):
        div =  div.sum(dim=-1)
   
    return div  ** (1 / alpha)   
    

def mink_mixture_divergence(p, Y, q, X, kernel, alpha, use_inv=False, use_avg=False):
    """
    Compute similarity sensitive Minkowski divergence of between a pair of empirical distributions
    p and q with supports Y and X, respectively
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        alpha [float] : Divergence order
        kernel [callable] : Function to compute the kernel matrix
        use_avg [boolean] : Determines whether to use Jensen-Shannon-like divergence wrt midpoint
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """
    
    Kyy = kernel(Y, Y)
    Kyx = kernel(Y, X)
    
    pK = p @ Kyy.transpose(0, 1)
    qK = q @ Kyx.transpose(0, 1)
    
    if use_inv:
        pK = 1 / utils.min_clamp(pK)
        qK = 1 / utils.min_clamp(qK)
        
    diff = torch.abs(pK - qK) ** alpha
    
    if use_avg:
        exp_measure = (p+q)/2
    else:
        exp_measure = p
        
    div =  (exp_measure * diff).sum(dim=-1) ** (1 / alpha)
    
    return div   


    
def renyi_sim_divergence(K, p, q, alpha=2, use_avg=False):
    """
    Compute similarity sensitive Renyi divergence of between a pair of (batches of)
    distribution(s) p and q over an alphabet of n elements.

    Inputs:
        p [batch_size x n tensor] : Probability distributions over n elements
        q [batch_size x n tensor] : Probability distributions over n elements
        K [n x n tensor or callable] : Positive semi-definite similarity matrix or function
        alpha [float] : Divergence order
        use_inv [boolean] : Determines whether to compare similarity or dissimilarity profiles
    Output:
        div [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
    """

    if callable(K):
        pK = K(p)
        qK = K(q)
    else:
        pK = p @ K
        qK = q @ K

    if use_avg:
        r = (p + q) / 2
        if callable(K):
            rK = K(r)
        else:
            rK = r @ K
        rat1 = (pK, rK)
        rat2 = (qK, rK)
    else:
        rat1 = (pK, qK)
        rat2 = (qK, pK)

    if callable(K):  # we're dealing with an image
        sum_dims = (-2, -1)
    else:
        sum_dims = -1

    if np.allclose(alpha, 1.0):            
        dp1 = (p * (torch.log(rat1[0]) - torch.log(rat1[1]))).sum(sum_dims)
        dp2 = (q * (torch.log(rat2[0]) - torch.log(rat2[1]))).sum(sum_dims)
        return 0.5 * (dp1 + dp2)
    else:
        power_pq = torch.log(p) + (alpha - 1) * (torch.log(rat1[0]) - torch.log(rat1[1]))
        power_qp = torch.log(q) + (alpha - 1) * (torch.log(rat2[0]) - torch.log(rat2[1]))
        return 0.5 * (1 / (alpha - 1)) * (torch.logsumexp(power_pq, sum_dims) + torch.logsumexp(power_qp, sum_dims))


def renyi_sim_divergence_stable(log_K, p, q, alpha=2, use_avg=False):
    """
    Compute similarity sensitive Renyi divergence of between a pair of (batches of)
    distribution(s) p and q over an alphabet of n elements.

    Inputs:
        p [batch_size x n tensor] : Probability distributions over n elements
        q [batch_size x n tensor] : Probability distributions over n elements
        log_K [n x n tensor or callable] : Log of positive semi-definite similarity matrix or function
        alpha [float] : Divergence order
        use_inv [boolean] : Determines whether to compare similarity or dissimilarity profiles
    Output:
        div [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
    """
    
    if callable(log_K):
        log_pK = log_K(p)
        log_qK = log_K(q)
    else:
        log_pK = torch.logsumexp(log_K[None, ...] + torch.log(p[:, None, :]), dim=2)
        log_qK = torch.logsumexp(log_K[None, ...] + torch.log(q[:, None, :]), dim=2)

    if use_avg:
        r = (p + q) / 2
        if callable(log_K):
            log_rK = log_K(r)
        else:
            log_rK = torch.logsumexp(log_K[None, ...] + torch.log(r[:, None, :]), dim=2)
        rat1 = (log_pK, log_rK)
        rat2 = (log_qK, log_rK)
    else:
        rat1 = (log_pK, log_qK)
        rat2 = (log_qK, log_pK)

    if callable(log_K):  # we're dealing with an image
        sum_dims = (-2, -1)
    else:
        sum_dims = -1

    if np.allclose(alpha, 1.0):            
        dp1 = (p * (rat1[0] - rat1[1])).sum(sum_dims)
        dp2 = (q * (rat2[0] - rat2[1])).sum(sum_dims)
        return 0.5 * (dp1 + dp2)
    else:
        power_pq = torch.log(p) + (alpha - 1) * (rat1[0] - rat1[1])
        power_qp = torch.log(q) + (alpha - 1) * (rat2[0] - rat2[1])
        return 0.5 * (1 / (alpha - 1)) * (torch.logsumexp(power_pq, sum_dims) + torch.logsumexp(power_qp, sum_dims))


def renyi_mixture_divergence(p, Y, q, X, kernel, alpha, use_avg=False, use_full=False, symmetric=True):
    """
    Compute similarity sensitive Renyi divergence of between a pair of empirical distributions
    p and q with supports Y and X, respectively
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        kernel [callable] : Function to compute the kernel matrix
        alpha [float] : Divergence order
        use_avg [boolean] : Determines whether to use Jensen-Shannon-like divergence wrt midpoint
        use_full [boolean] : Use the kernel even for the outer probability term
        symmatric [boolean] : Use the symmetric version of the divergence
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """
    
    Kyy = kernel(Y, Y)
    Kyx = kernel(Y, X)
    Kxx = kernel(X, X)
    
    Kyy_p = p @ Kyy.transpose(0, 1)
    Kxy_p = p @ Kyx
    
    Kyx_q = q @ Kyx.transpose(0, 1)
    Kxx_q = q @ Kxx.transpose(0, 1)

    if use_full:
        Kp = torch.cat([Kyy_p, Kxy_p], dim=1)
        Kq = torch.cat([Kyx_q, Kxx_q], dim=1)
        if use_avg:
            rat1 = (2 * Kp, Kp + Kq)
            rat2 = (2 * Kq, Kq + Kp)
        else:
            rat1 = (Kp, Kq)
            rat2 = (Kq, Kp)
        P = Kp
        Q = Kq
    else:
        if use_avg:
            rat1 = (2 * Kyy_p, Kyy_p + Kyx_q)
            rat2 = (2 * Kxx_q, Kxx_q + Kxy_p)
        else:
            rat1 = (Kyy_p, Kyx_q)
            rat2 = (Kxx_q, Kxy_p)
        P = p
        Q = q

    if np.allclose(alpha, 1.0):
        div = (P * (torch.log(rat1[0]) - torch.log(rat1[1]))).sum(dim=-1)
        if symmetric:
            div = 0.5 * (div + (Q * (torch.log(rat2[0]) - torch.log(rat2[1]))).sum(dim=-1))
    else:
        power_pq = torch.log(P) + (alpha - 1) * (torch.log(rat1[0]) - torch.log(rat1[1]))
        div = (1 / (alpha - 1)) * torch.logsumexp(power_pq, 1)
        if symmetric:
            power_qp = torch.log(Q) + (alpha - 1) * (torch.log(rat2[0]) - torch.log(rat2[1]))
            div = 0.5 * (div + (1 / (alpha - 1)) * torch.logsumexp(power_qp, 1))

    return div


def renyi_mixture_divergence_stable(p, Y, q, X, log_kernel, alpha, use_avg=False, use_full=False, symmetric=True):
    """
    Compute similarity sensitive Renyi divergence of between a pair of empirical uniform distributions p and q
    with non-overlapping supports Y and X respectively
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        log_kernel [callable] : Function to compute the log kernel matrix
        alpha [float] : Divergence order
        use_avg [boolean] : Determines whether to use Jensen-Shannon-like divergence wrt midpoint
        use_full [boolean] : Use the kernel even for the outer probability term
        symmatric [boolean] : Use the symmetric version of the divergence
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

    if use_full:
        log_Kp = torch.cat([log_Kyy_p, log_Kxy_p], dim=1)
        log_Kq = torch.cat([log_Kyx_q, log_Kxx_q], dim=1)
        if use_avg:
            log_Kp_Kq = torch.logsumexp(torch.stack([log_Kp, log_Kq]), 0)
            rat1 = (np.log(2) + log_Kp, log_Kp_Kq)
            rat2 = (np.log(2) + log_Kq, log_Kp_Kq)
        else:
            rat1 = (log_Kp, log_Kq)
            rat2 = (log_Kq, log_Kp)
        log_P = log_Kp
        log_Q = log_Kq
        P = torch.exp(log_P)
        Q = torch.exp(log_Q)
    else:
        if use_avg:
            rat1 = (np.log(2) + log_Kyy_p, torch.logsumexp(torch.stack([log_Kyy_p, log_Kyx_q]), 0))
            rat2 = (np.log(2) + log_Kxx_q, torch.logsumexp(torch.stack([log_Kxx_q, log_Kxy_p]), 0))
        else:
            rat1 = (log_Kyy_p, log_Kyx_q)
            rat2 = (log_Kxx_q, log_Kxy_p)
        P = p
        Q = q
        log_P = torch.log(P)
        log_Q = torch.log(Q)

    if np.allclose(alpha, 1.0):
        div = (P * (rat1[0] - rat1[1])).sum(dim=-1)
        if symmetric:
            div = 0.5 * (div + (Q * (rat2[0] - rat2[1])).sum(dim=-1))
    else:
        power_pq = log_P + (alpha - 1) * (rat1[0] - rat1[1])
        div = (1 / (alpha - 1)) * torch.logsumexp(power_pq, 1)
        if symmetric:
            power_qp = log_Q + (alpha - 1) * (rat2[0] - rat2[1])
            div = 0.5 * (div + (1 / (alpha - 1)) * torch.logsumexp(power_qp, 1))

    return div


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


def cosine_similarity(X, Y, log=False):
    ret = utils.batch_cosine_similarity(X, Y)
    if log:
        return torch.log(ret)
    else:
        return ret

def poly_kernel(X, Y, degree=2, p=2, log=False):
    pdist = utils.batch_pdist(X, Y, p)
    ret = utils.min_clamp_prob(1 / (1 + pdist)**degree)
    if log:
        return torch.log(ret)
    else:
        return ret

def rbf_kernel(X, Y, sigmas=[1.], p=2, degree=2, log=False):
    pdist = utils.batch_pdist(X, Y, p)
    res = torch.zeros_like(pdist)
    if log:
        log_res = torch.zeros_like(pdist)
    for sigma in sigmas:
        logits = - (pdist/sigma)**degree
        res += torch.exp(logits)
        if log:
            log_res += logits
    ret = res / len(sigmas)
    if log:
        return log_res / len(sigmas)  # incorrect for log if len(sigmas) > 1
    else:
        return utils.min_clamp_prob(ret)

def generic_kernel(X, Y, kernel_fn, full=False, log=False):
    if full:
        W = torch.cat((X, Y))
        return kernel_fn(W, W)
    else:
        return kernel_fn(X, Y)
