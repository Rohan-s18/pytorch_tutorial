"""
author: rohan singh
simple introduction to pytroch tensor attributes
"""

# imports and set up
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


# creating the tensor
x = torch.randn(size=(100,200), device=device) 


# printing the attributes of the tensor
print(x.dtype)
print(x.device)
print(x.shape)
print(x.ndim)
print(x.requires_grad)
print(x.grad) if x.requires_grad else print("no gradients")
print(x.layout)
