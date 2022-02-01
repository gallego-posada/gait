import torch
from . import utils

def cosine_similarity(X, Y, log=False):
    ret = utils.batch_cosine_similarity(X, Y)
    if log:
        return torch.log(ret)
    else:
        return ret

def poly_kernel(X, Y, degree=2, c=1, p=2, log=False):
    pdist = utils.batch_pdist(X, Y, p)
    if log:
        return -torch.log(1 + c * pdist) * degree
    else:
        return utils.min_clamp_prob(1 / (1 + c * pdist)**degree)
    
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
