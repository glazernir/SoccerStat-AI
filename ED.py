import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# Defining the Autoencoder model
class Autoencoder(torch.nn.Module):
    def __init__(self, input_size, encoding_dim):
        """
        Initialize the Autoencoder model.

        Parameters:
        - input_size: The number of input features (e.g., flattened image pixels).
        - encoding_dim: The size of the encoded (compressed) representation.
        """
        super(Autoencoder, self).__init__()

        # Encoder: Compresses input from input_size to encoding_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 16),  # First layer reduces to 16 neurons
            torch.nn.ReLU(),  # Activation function introduces non-linearity
            torch.nn.Linear(16, encoding_dim),  # Second layer reduces to encoding_dim
            torch.nn.ReLU(),  # Activation function introduces non-linearity
            torch.nn.Sigmoid()  # Sigmoid ensures values remain between 0 and 1
        )

        # Decoder: Reconstructs data from encoding_dim back to input_size
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 16),  # First layer expands to 16 neurons
            torch.nn.ReLU(),  # Activation function introduces non-linearity
            torch.nn.Linear(16, input_size),  # Final layer reconstructs the original input size
            torch.nn.Sigmoid()  # Sigmoid ensures reconstructed values remain between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters:
        - x: The input data.

        Returns:
        - The reconstructed output after encoding and decoding.
        """
        x = self.encoder(x)  # Pass input through the encoder
        x = self.decoder(x)  # Pass encoded representation through the decoder
        return x
