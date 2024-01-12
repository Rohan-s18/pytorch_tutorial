"""
author: rohan singh
simple introduction to pytroch tensor intialization
"""

# imports and set up
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"



# creating sample tensors
x = torch.tensor([[1,2,3],[4,5,6]], device=device)
y = torch.tensor([[7,8,9],[10,11,12]], device=device) 

# arithemetic tensor creation
z = x + y

print(z)
print(z.size())
print(z.device)


# creating particular tensors
a = torch.empty(size=(100,200))
b= torch.zeros(size=(100,200))
c = torch.ones(size=(100,200))


# creating random tensors (this is based on size)
d = torch.rand(size=(100,200))             # uniform distribution
e = torch.randn(size=(100,200))            # normalized distribution
f = torch.randint(5,10,size=(100,200))     # integer distribution


# other ways to intialize tensors
g = torch.linspace(start=0, end=10, steps=100)           # creates linearly spaced points between 0 and 10
h = torch.logspace(start=0, end=10, steps=100)           # creates logarithmic spaced points between 0 and 10
i = torch.eye(n=10)                                      # creates an identity matrix of size n x n
j = torch.full(size=(10,10), fill_value=-3)              # creates a tensor of value "-3"
k = torch. normal(mean=80, std=10, size=(1,100))         # creates a normalized distribution

print(k)
