"""
author: rohan singh
python script for a simple MNIST torch models
"""



# imports
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L




# models
encoder = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64,3))
decoder = nn.Sequential(nn.Linear(3,64), nn.ReLU(), nn.Linear(64, 28*28))
