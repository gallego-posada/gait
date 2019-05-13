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

    
# def sim_divergence(K, p, q, alpha=2, use_avg=False):
#     """
#     Compute similarity sensitive Jensen-Renyi divergence of between a pair of (batches of)
#     distribution(s) p and q over an alphabet of n elements.

#     Inputs:
#         K [n x n tensor] : Positive semi-definite similarity matrix
#         p [batch_size x n tensor] : Probability distributions over n elements
#         q [batch_size x n tensor] : Probability distributions over n elements
#         alpha [float] : Divergence order
    
#     Output:
#         [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
#     """

#     tK = K.transpose(0, 1)
#     pK = p @ tK
#     qK = q @ tK

#     if use_avg:
#         r = (p + q) / 2
#         rK = r @ tK
#         rat1 = pK / utils.min_clamp(rK)
#         rat2 = qK / utils.min_clamp(rK)
#     else:   
#         rat1 = pK / utils.min_clamp(qK)
#         rat2 = qK / utils.min_clamp(pK)
    
#     if np.allclose(alpha, 1.0):            
#         dp1 = (p * torch.log(utils.min_clamp(rat1))).sum(dim=1)
#         dp2 = 0+(q * torch.log(utils.min_clamp(rat2))).sum(dim=1)
        
#         return dp1 + dp2
#     else:    
#         dp1 = (p * rat1**(alpha-1)).sum(dim=1)
#         dp2 = 1*(q * rat2**(alpha-1)).sum(dim=1)

#         return 1/(alpha - 1) * (torch.log(dp1) + torch.log(dp2))

    
def mink_sim_divergence(K, p, q, alpha=2, use_inv=False):
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
    
    div =  (p * diff).sum(dim=-1) 
    if callable(K):
        div =  div.sum(dim=-1)
   
    return div  ** (1 / alpha)   
    

def mink_mixture_divergence(p, Y, q, X, kernel, alpha, use_inv=False):
    """
    Compute similarity sensitive divergence of between a pair of empirical distributions
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
    div =  (p * diff).sum(dim=-1) ** (1 / alpha)
    
    return div   

    
def mixture_divergence(mu, X, nu, Y, alpha, kernel, use_avg=False):
    """
    Compute similarity sensitive divergence of between a pair of empirical distributions
    mu and nu with supports X and Y, respectively
    Inputs:
        mu [1 x n tensor] : Probability distribution over n elements
        X [n x d tensor] : Locations of the atoms of the measure mu
        nu [1 x m tensor] : Probability distribution over m elements
        Y [n x d tensor] : Locations of the atoms of the measure nu
        alpha [float] : Divergence order
        kernel [callable] : Function to compute the kernel matrix between locations X and Y
        use_avg [boolean] : Determines whether to use Jensen-Shannon-like divergence wrt midpoint
    
    Output:
        div [1 x 1 tensor] similarity sensitive divergence of between mu and nu
        K [(n+m) x (n+m) tensor] : Positive semi-definite similarity matrix
        
    """

    n, m = mu.shape[1], nu.shape[1]
   
    K = kernel(X, Y)
    
    Kmu = K[:, :n] @ mu.transpose(0, 1)
    Knu = K[:, -m:] @ nu.transpose(0, 1)
    
    Kxx_mu, Kyx_mu = Kmu[:n], Kmu[-m:]
    Kxy_nu, Kyy_nu = Knu[:n], Knu[-m:]

    if use_avg:
        rat1 = 2 * Kxx_mu / utils.min_clamp(Kxx_mu + Kxy_nu)
        rat2 = 2 * Kyy_nu / utils.min_clamp(Kyy_nu + Kyx_mu)
    else:
        rat1 = Kxx_mu / utils.min_clamp(Kxy_nu)
        rat2 = Kyy_nu / utils.min_clamp(Kyx_mu)
    
    if alpha == 1:
        div = (mu @ utils.clamp_log(rat1)) + (nu @ utils.clamp_log(rat2))
    else:
        div = (1/(alpha - 1)) * ( utils.clamp_log(mu @ rat1**(alpha-1)) + utils.clamp_log(nu @ rat2**(alpha-1)))
        
    return div, K
    
    
def cosine_similarity(X, Y):
    return utils.batch_cosine_similarity(X, Y)

def poly_kernel(X, Y, degree=2, p=2):
    pdist = utils.batch_pdist(X, Y, p)
    return 1 / (1 + pdist)**degree

def rbf_kernel(X, Y, sigmas=[1.], p=2, degree=2):
    pdist = utils.batch_pdist(X, Y, p)
    res = torch.zeros_like(pdist)
    for sigma in sigmas:
        res += torch.exp(- (pdist/sigma)**degree)
    return res / len(sigmas)

def generic_kernel(X, Y, kernel_fn, full=False):
    if full:
        W = torch.cat((X, Y))
        return kernel_fn(W, W)
    else:
        return kernel_fn(X, Y)

# def mink_divergence(p, q, K, alpha=1 , use_inv=False):   
    
#     if callable(K):
#         pK = K(p)
#         qK = K(q)
#     else:
#         pK = p @ K
#         qK = q @ K
    
#     if use_inv:
#         pK = 1 / utils.min_clamp(pK)
#         qK = 1 / utils.min_clamp(qK)
        
#     diff = torch.abs(pK - qK) ** alpha
#     div =  (p * diff).sum(dim=1).sum(dim=1) ** (1 / alpha)
#     return div

# def sim_divergence(K, p, q, alpha=1):    
#     """
#     Compute similarity sensitive Jensen-Renyi divergence of between a pair of (batches of)
#     distribution(s) p and q over an alphabet of n elements.
    
#     Inputs:
#         K [n x n tensor] : Positive semi-definite similarity matrix
#         p [batch_size x n tensor] : Probability distributions over n elements
#         q [batch_size x n tensor] : Probability distributions over n elements
#         alpha [float] : Divergence order
    
#     Output:
#         [batch_size x 1 tensor] i-th entry is divergence between i-th row of p and i-th row of q 
#     """
#     p = p.transpose(0, 1)
#     q = q.transpose(0, 1)
    
#     r = (p + q)/ 2
    
#     Kp = K @ p
#     Kq = K @ q
#     Kr = K @ r
    
#     p_by_r = utils.min_clamp(Kp / Kr)
#     q_by_r = utils.min_clamp(Kq / Kr)
    
#     if alpha == 1:
        
#         p_by_r = torch.log(p_by_r)
#         q_by_r = torch.log(q_by_r)
#         res = -(p * p_by_r).sum(dim=0, keepdim=True) - (q * q_by_r).sum(dim=0, keepdim=True) 
        
#     else:
        
#         p_by_r = p_by_r ** (alpha - 1)
#         q_by_r = q_by_r ** (alpha - 1)

#         res = (p * p_by_r).sum(dim=0, keepdim=True) * (q * q_by_r).sum(dim=0, keepdim=True) 
#         res = torch.log(utils.min_clamp(res))/(alpha - 1)
     
#     return res.transpose(0, 1)

    
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
