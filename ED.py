import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# Defining Autoencoder model
class Autoencoder(torch.nn.Module):
   def __init__(self, input_size, encoding_dim):
       super(Autoencoder, self).__init__()
       self.encoder = torch.nn.Sequential(
           torch.nn.Linear(input_size, 16),
           torch.nn.ReLU(),
           torch.nn.Linear(16, encoding_dim),
           torch.nn.ReLU()
       )
       self.decoder = torch.nn.Sequential(
           torch.nn.Linear(encoding_dim, 16),
           torch.nn.ReLU(),
           torch.nn.Linear(16, input_size),
           torch.nn.Sigmoid()
       )

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x