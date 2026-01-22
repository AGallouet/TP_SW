import torch
from PIL import Image
import numpy as np

# For installation
# pip install numpy
# pip install torch
# pip install pillow

# Cuda is the GPU device on pytorch,
# Vectors and matrix computations runs 
# significantly faster on GPU.
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("CUDA Enabled")
else:
	device = torch.device("cpu")
	print("CUDA Disabled")
	


# -------------------------------
# For pytorch tensor manipulation
# -------------------------------



# Create a tensor
x = torch.tensor([2.2, 1.1, 3.3, 0, 5.5], device = device) # Declaring the device is good practice to avoid errors between CPU and cuda (GPU)
x_sort, ind_sort = torch.sort(x) #torch.sort return sorted tensor and their position in the initial one
x_sort2 = x[ind_sort] # another way to obtain sorted tensor


print("x = ", x)
print("sorted vector is : \n  ", x_sort, "\n =", x_sort2)
print("Permutation to sort values :", ind_sort)
# torch.argsort only returns the indices to sort the array (2 return value of torch.sort)
# To obtain the inverse permutation of the one that sorts tensor, one can use torch.argsort on the permutation itself
ind_inverse = torch.argsort(ind_sort)
print("Inverse permutation = ", ind_inverse)
# Another way of saying this is that torch.argsort(torch.argsort(x)) gives the rank 
# of each elements of x in the ordering.

# A way to retrieve the original tensor is thus
print("original tensor :", x_sort[ind_inverse])

# For a random tensor with normal law:
y = torch.randn(10, device = device)

# -----------------------------------------------------------
# Now about how to treat RGB image with PIL.Image and pytorch
# -----------------------------------------------------------



# This line loads the image source_small into an Image object
image = Image.open("images/source_small.jpg").convert('RGB')

# Transform the image into a torch.tensor (which is a vector in pytorch)
# It is supported on the chosen above device (cpu or gpu) and 
# its elements will be single precision float
x = torch.tensor(np.asarray(image), device = device, dtype = torch.float32)

# RGB images are between 0 and 255, it is often simpler to normalize between 0 and 1.
x = x / 255. # divided by float to get float tensor instead of integers

# The image is of size height * width, and in dimension dim = 3 (for RGB)
height, width, dim = x.shape

# It is often simpler to work on n * d tensors instead, we can reshape our image with
m = height * width
x = x.reshape([m, dim])

# We are going to apply a permutation to the colors of the image
# First we need a permutation matrix in 3 dimension
permutation_matrix = torch.tensor([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]], device = device) # Has to be on the same device


# Matrix multiplication 
x_perm = torch.matmul(x, permutation_matrix)
# Equivelent to 
x_perm = x @ permutation_matrix

# Rescale to [0, 255]
x_perm = x_perm * 255.

# Give back the image shape
x_perm = x_perm.reshape([height, width, 3]).to(dtype = torch.uint8)

# Cast on numpy array and from GPU to CPU if necessary to be able to save the output
output = x_perm.cpu().numpy()

# Saving my output in RGB style with name "transformed.jpg"
Image.fromarray(output, mode = 'RGB').save("images/transformed.jpg")

