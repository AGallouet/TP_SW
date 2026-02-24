import torch
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt



# torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("CUDA Enabled")
else:
	device = torch.device("cpu")
	print("CUDA Disabled")


def generate_random_directions(d, n, device):
	"""
	Generate n random vectors on the d-dimensional hypersphere
	The result is of size (d, n) so that projecting a measure
	u of size (m, d) is just a matrix product u times result
	"""
	thetas = torch.randn(d, n, device = device)
	thetas = torch.nn.functional.normalize(thetas, dim = 0)
	return thetas




def sw(u, v, nb_slices, batch_size = 64):
	"""Approximates SW_2(u,v) using nb_slices projections
	Computations are divided in batches to avoid memory overflow when measures are big."""
	m = u.size(0) # Number of diracs in measure u
	d = u.size(1)
	device = u.device
	niter = nb_slices // batch_size
	rest = nb_slices % batch_size
	result = 0
	for _ in range(niter):
		thetas = generate_random_directions(d, batch_size, device=device)
		uproj = torch.matmul(u,thetas)
		vproj = torch.matmul(v,thetas)
		uproj = uproj.sort(dim = 0)[0]
		vproj = vproj.sort(dim = 0)[0]
		result +=  torch.pow(uproj - vproj,2).sum() / m
	if rest != 0:
		thetas = generate_random_directions(d, rest, device=device)
		uproj = torch.matmul(u,thetas)
		vproj = torch.matmul(v,thetas)
		uproj = uproj.sort(dim = 0)[0]
		vproj = vproj.sort(dim = 0)[0]
		result +=  torch.pow(uproj - vproj,2).sum() / m
	
	return torch.sqrt(result / nb_slices)


def sw_grad(u, v, n, thetas = None):
	""" u, v should be m * d, and thetas should be d * n"""
	m = u.size(0)
	d = u.size(1)

	device = u.device
	if thetas == None:
		thetas = generate_random_directions(d, n, device)
	uproj = torch.matmul(u,thetas)
	vproj = torch.matmul(v,thetas)
	uprojsort, usortidx = uproj.sort(dim = 0)
	usortidxrev = usortidx.argsort(dim = 0)
	vprojsort = vproj.sort(dim = 0)[0]
	dist = (torch.pow(uprojsort - vprojsort,2).sum(dim = 0) / m).sum() / n
	# Grad should be multiplied by (2/m) to be exact
	grad = (1/n) * torch.matmul((uproj - vprojsort.take_along_dim(usortidxrev, dim = 0)), thetas.T)
	# should return a m * d tensor
	return dist, grad



im_size = "medium" # (small, medium of big)
image = Image.open("images/source_" + im_size + ".jpg").convert('RGB')
model = Image.open("images/colors_" + im_size + ".jpg").convert('RGB')


u = torch.tensor(np.asarray(image), device = device, dtype = torch.float32) / 255.
v = torch.tensor(np.asarray(model), device = device, dtype = torch.float32) / 255.

#Images should be the same sizes
assert u.shape == v.shape
assert u.shape[2] == 3
height = u.shape[0]
width = u.shape[1]
d = 3
m = height * width
print("measure size = ", m)

# To get image back use u.reshape([height, width, 3])
u = u.reshape([m, d]) 
v = v.reshape([m, d]) 
initial = u

nb_step = 101


nb_slices = 20
tau = 3



initial_distance = sw(u,v,100)
print("initial dist = ", initial_distance)

t0 = time.time()
mu = u.clone().detach()


for i in range(nb_step):
    torch.cuda.synchronize()
    t = time.time()
    dist, grad = sw_grad(mu, v, nb_slices)
    if i % (nb_step // 10) == 0:
        print("\n Run  ", i)
        print("Dist = ", dist.item())
    mu -= tau * grad
    torch.cuda.synchronize()

print("Computing time = ", time.time() - t0 )

print("min = ", mu.min(), "max = ", mu.max())
mu = torch.minimum(mu, torch.ones(mu.shape, device = device))
mu = torch.maximum(mu, torch.zeros(mu.shape, device = device))

# Color transfered image is saved with this name
outfilename = "images/image_transfer_" + im_size

mu *= 255.

mu = mu.reshape([height, width, 3]).to(dtype = torch.uint8)
output = mu.cpu().numpy()
Image.fromarray(output, mode = 'RGB').save(outfilename + ".jpg")


