import numpy as np
import torch
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
    
    
    if alpha == 1:
        ent = -(p * torch.log(pK)).sum(dim=-1)
    else:
        Kpa = pK ** (alpha - 1)
        v = (p * Kpa).sum(dim=-1, keepdim=True) 
        ent = torch.log(v) / (1 - alpha)
    
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
        return dp1 + dp2
    else:
        power_pq = torch.log(p) + (alpha - 1) * (torch.log(rat1[0]) - torch.log(rat1[1]))
        power_qp = torch.log(q) + (alpha - 1) * (torch.log(rat2[0]) - torch.log(rat2[1]))
        return (1 / (alpha - 1)) * (torch.logsumexp(power_pq, sum_dims) + torch.logsumexp(power_qp, sum_dims))


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
            div = div + (Q * (torch.log(rat2[0]) - torch.log(rat2[1]))).sum(dim=-1)
    else:
        power_pq = torch.log(P) + (alpha - 1) * (torch.log(rat1[0]) - torch.log(rat1[1]))
        div = (1 / (alpha - 1)) * torch.logsumexp(power_pq, 1)
        if symmetric:
            power_qp = torch.log(Q) + (alpha - 1) * (torch.log(rat2[0]) - torch.log(rat2[1]))
            div = div + (1 / (alpha - 1)) * torch.logsumexp(power_qp, 1)

    return div


def test_mixture_divergence(p, Y, q, X, kernel, use_avg=False, symmetric=True):
    """
    Inputs:
        p [1 x n tensor] : Probability distribution over n elements
        Y [n x d tensor] : Locations of the atoms of the measure p
        q [1 x m tensor] : Probability distribution over m elements
        X [n x d tensor] : Locations of the atoms of the measure q
        kernel [callable] : Function to compute the kernel matrix
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
    """

    Kyy = kernel(Y, Y)
    Kyx = kernel(Y, X)
    Kxx = kernel(X, X)

    one = torch.ones_like(p)

    Kyy_p = p @ Kyy.transpose(0, 1)
    Kxy_p = p @ Kyx
    Kyx_q = q @ Kyx.transpose(0, 1)
    Kxx_q = q @ Kxx.transpose(0, 1)

    Kyy_1 = one @ Kyy.transpose(0, 1)
    Kxy_1 = one @ Kyx
    Kyx_1 = one @ Kyx.transpose(0, 1)
    Kxx_1 = one @ Kxx.transpose(0, 1)

    Kp = torch.cat([Kyy_p, Kxy_p], dim=1)
    Kq = torch.cat([Kyx_q, Kxx_q], dim=1)
    K1 = torch.cat([Kyy_1 + Kyx_1, Kxy_1 + Kxx_1], dim=1)

    if use_avg:
        Km = (Kp + Kq) / 2
        div = ((Kp/K1) * (torch.log(Kp) - torch.log(Km))).sum(dim=1)
        if symmetric:
            div = div + ((Kq/K1) * (torch.log(Kq) - torch.log(Km))).sum(dim=1)
    else:
        div = ((Kp/K1) * (torch.log(Kp) - torch.log(Kq))).sum(dim=1)
        if symmetric:
            div = div + ((Kq/K1) * (torch.log(Kq) - torch.log(Kp))).sum(dim=1)

    return div


def cosine_similarity(X, Y):
    return utils.batch_cosine_similarity(X, Y)

def poly_kernel(X, Y, degree=2, p=2):
    pdist = utils.batch_pdist(X, Y, p)
    return utils.min_clamp_prob(1 / (1 + pdist)**degree)

def rbf_kernel(X, Y, sigmas=[1.], p=2, degree=2):
    pdist = utils.batch_pdist(X, Y, p)
    res = torch.zeros_like(pdist)
    for sigma in sigmas:
        res += torch.exp(- (pdist/sigma)**degree)
    return utils.min_clamp_prob(res / len(sigmas))

def multiquad_kernel(X, Y, sigma=1., p=2, degree=2):
    pdist = utils.batch_pdist(X, Y, p)
    C = 2 * X.size(-1) * (sigma ** degree)
    return utils.min_clamp_prob(C / (C + (pdist ** degree)))

def generic_kernel(X, Y, kernel_fn, full=False):
    if full:
        W = torch.cat((X, Y))
        return kernel_fn(W, W)
    else:
        return kernel_fn(X, Y)
