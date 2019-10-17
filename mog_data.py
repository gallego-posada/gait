from scipy.stats import multivariate_normal
import numpy as np

def sample_mix(rvs, p, n_samples):
    out = []
    for i in range(n_samples):
        j = np.random.choice(len(p), p=p)
        out += [np.random.multivariate_normal(rvs[j].mean, rvs[j].cov)]
    return np.stack(out)

def eval_mix_pdf(x, rvs,  p):
    cont = np.zeros_like(x[:, :, 0])
    for _ in range(len(rvs)):
        cont += p[_] * rvs[_].pdf(x)
    return cont

def create_rvs(p, mus, covs):
    rvs = []
    for _ in range(len(mus)):
        rvs.append(multivariate_normal(np.array(mus[_]), 0.5*np.array(covs[_])))
    return rvs

