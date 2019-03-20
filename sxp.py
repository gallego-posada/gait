import torch
import utils

def batch_simplex_project(y):
	"""
	Computes the projection onto the n-simplex of a (batch of) n-dimensional vectors
	using the projsplx algorithm by Yunmei Chen and Xiaojing Ye (arXiv:1101.6081v2)
	
	See simplex_project() for details.
	"""
    
	batch_size, n = y.shape
	s = torch.sort(y, descending=True, dim=1)[0]
    
	bget_vec = torch.zeros(batch_size, dtype=torch.uint8)
	tmpsum_vec = torch.zeros(batch_size)
	tmax_vec = torch.zeros(batch_size)
    
	for ii in range(1, n):
		tmpsum_vec[bget_vec ^ 1] += s[:, ii-1][bget_vec ^ 1]
		tmax_vec[bget_vec ^ 1] = (tmpsum_vec[bget_vec ^ 1] - 1) / ii
		bget_vec = torch.max(bget_vec, tmax_vec >= s[:, ii])

	# Byte ^ 1 = not(Byte)
	tmax_vec[bget_vec ^ 1] =  (tmpsum_vec[bget_vec ^ 1] + s[:, -1][bget_vec ^ 1] - 1)/n    
    
	return utils.min_clamp(y - tmax_vec.unsqueeze(-1), 0)

def one_simplex_project(y):
	"""
	Computes the projection onto the n-simplex of *one* n-dimensional vector using the
	projsplx algorithm by Yunmei Chen and Xiaojing Ye (arXiv:1101.6081v2) 
	Note: this is a python translation of their MATLAB code.
	
	Inputs:
		y [1 x n tensor] : n-dimensional vector
	
	Output:
		[1 x n tensor] projection of y onto n-simplex 
	"""

	n = y.shape[-1]
	s = torch.sort(y, descending=True, dim=1)[0]
    
	bget = False
	tmpsum = 0
	for ii in range(1, n):
		tmpsum = tmpsum + s[0, ii-1]
		tmax = (tmpsum - 1)/ii
		if tmax >= s[0, ii]:
			bget = True
			break
			
	if not bget:
		tmax = (tmpsum + s[0, -1] - 1)/n
		
	return utils.min_clamp(y - tmax, 0)

def simplex_project(y):
	"""
	Computes the projection onto the n-simplex of a (batch of) n-dimensional vectors
	using the projsplx algorithm by Yunmei Chen and Xiaojing Ye (arXiv:1101.6081v2)
	
	Inputs:
		y [batch_size x n tensor] : n-dimensional vectors
	
	Output:
		[batch_size x n tensor] i-th row is the projection of i-th row of y onto n-implex 
    """
		
	b, n = y.shape
	
	if b<100 or (b < 0.05 * n and n < 500):
		# If the problem if small enough, do iterative version
		# to avoid overhead
		res = torch.zeros_like(y)
		for _ in range(y.shape[0]):
			res[_, :] = one_simplex_project(y[_:_+1, :])
		return res
	else:
		# Run vectorized version for large problems
		
		# Performance tests on Lenovo Processor Intel(R) 
		# Core(TM) i5-8350U CPU @ 1.70GHz, 1896 Mhz, 
		# 4 Core, 8 Logical Processors

		# b=100.000, n=100 takes 1s
		# --------------------------
		# b=10.000, n=1000 takes 0.2s
		# b=1000, n=10.000 takes 2s
		# b=1000, n=100.000 takes 12s
		return batch_simplex_project(y)
		
#----------------------------------------------------------------#
# Peformance comparison against solving QP formulation with CVXOPT
#----------------------------------------------------------------#

# from cvxopt import matrix, solvers
# solvers.options['show_progress'] = False

# n = 5000
# y = torch.rand(1, n)
# npy = y.squeeze().numpy().astype(np.double)

# p = -matrix(npy)
# Q = matrix(np.eye(n))
# G = -Q
# h = matrix(0.0, (n,1))
# A = matrix(1.0, (1,n))
# b = matrix(1.0)

# %time sol = solvers.qp(Q, p, G, h, A, b)
# sol = np.array(sol['x'].T)
# %time my_sol = sxp.simplex_project(y).numpy()

# print(sol.sum(), my_sol.sum())
# print('Error: ', np.linalg.norm(sol - my_sol))

## Wall time: 22.7 s
## Wall time: 6.03 ms
## 0.9999999999999996 0.99997294
## Error:  9.258782399124384e-06
	
#----------------------------------------------------------------#