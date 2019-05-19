import numpy as np
import torch
from torch.nn import functional as F
from skimage.transform import resize

EPS = {}
for dtype in [torch.float16, torch.float32, torch.float64]:
    EPS[dtype] = torch.finfo(dtype).eps * 2


def batch_pdist(X, Y, p=2):
    return torch.norm(X[..., None, :] - Y[..., None, :, :], p=p, dim=-1)

def batch_cosine_similarity(X, Y, p=2):
    return torch.cosine_similarity(X[..., None, :], Y[..., None, :, :], dim=-1)

def min_clamp_prob(x, c=None):
    if c is None:
        c = EPS[x.dtype]
    return c + (x * (1 - c))

def min_clamp(x, c=None):
    if c is None:
        c = EPS[x.dtype]
    return c + x

def clamp_log_prob(x, c=None):
    return torch.log(min_clamp_prob(x, c))

def clamp_log(x, c=None):
    return torch.log(min_clamp(x, c))

def from_numpy(x, requires_grad=False):
	x = torch.from_numpy(x)
	x.requires_grad_(requires_grad)
	return x

def draw_circle(img_size):
	xrng = np.linspace(-1, 1, img_size)
	[xx, yy] = np.meshgrid(xrng, xrng)
	A = 1.0 * (xx**2 + yy**2 <= 0.7)
	return torch.tensor(A)

def radians(angle):
	return angle * np.pi / 180.


def sample_and_resize(D, img_size, num_samples=1):
    res = []
    for _ in range(num_samples):
        ix = int(np.random.choice(len(D), 1))
        sample_img = D[ix, ...].data.numpy()
        #sample_img = tform(sample_img)[0, ...].data.numpy()
        if img_size != 28:
            sample_img = resize(sample_img, (img_size, img_size), mode='constant')
        sample_img = np.abs(sample_img) / np.abs(sample_img).sum()
        res.append(sample_img)
    return torch.tensor(res).double()

def convolve(img, filt):
    #kwidth = 27
    #g = torch.exp(- torch.abs(torch.tensor(np.linspace(-kwidth/(2*img_size),kwidth/(2*img_size), kwidth)))**1 / (0.2**2))
    #img_kernel = lambda x: convolve(x, g)

#     c1 = img_kernel(nu)
#     c2 = convolve(nu, g)

#     plt.imshow(c1[3].data.numpy())
#     plt.colorbar()
#     plt.show()

#     plt.imshow(c2[3].data.numpy())
#     plt.colorbar()


    # MUCH slower than matmul
    if len(img.shape) == 3:
        img = img[:, None, :, :]
    elif len(img.shape) == 2:
        img = img[None, None, :, :]
    
    # img shape (batch, channels, size, size)
    # filter shape (size)
    
    filters = filt[None, None, None, :].type_as(img)
    outconv = F.conv2d(img, filters, padding=[0, len(filt)//2])
    outconv = F.conv2d(outconv, filters.transpose(-1, -2), padding=[len(filt)//2, 0])
    return outconv[:, 0, :, :]
