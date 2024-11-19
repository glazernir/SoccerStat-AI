import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from ED import Autoencoder

def trainDataset(dataset):

    # for training
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=32,
                                         shuffle=True)

    # Converting to PyTorch tensor
    dataset_tensor = torch.FloatTensor(dataset.to_numpy())
    # Setting random seed for reproducibility
    torch.manual_seed(42)

    input_size = dataset.shape[1]  # Number of input features
    encoding_dim = 100  # Desired number of output dimensions
    model = Autoencoder(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Training the autoencoder
    num_epochs = 20
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(dataset_tensor)
        loss = criterion(outputs, dataset_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Encoding the data using the trained autoencoder
    encoded_data = model.encoder(dataset_tensor).detach().numpy()



