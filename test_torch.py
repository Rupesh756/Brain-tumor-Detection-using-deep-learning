import numpy as np
import torch

print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.tensor([1, 2, 3])
print("Tensor test:", x)

print("Both libraries are working!")