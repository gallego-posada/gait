import numpy as np
import torch
EPS = 1e-10

def min_clamp(x, c=EPS):
	return torch.clamp_min(x, c)

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