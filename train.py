import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from ED import Autoencoder
import pandas as pd

def trainDataset(dataset):

    player_numbers = dataset.iloc[:, 0]
    data_to_encode = dataset.iloc[:, 1:]
    dataset_normalized = (data_to_encode - data_to_encode.min()) / (data_to_encode.max() - data_to_encode.min())

    # Converting to PyTorch tensor
    dataset_tensor = torch.FloatTensor(dataset_normalized.to_numpy())

    #create a TensorDataset
    dataset_obj = TensorDataset(dataset_tensor,dataset_tensor)

    loader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

    # Setting random seed for reproducibility
    torch.manual_seed(42)

    input_size = data_to_encode.shape[1]  # Number of input features
    encoding_dim = 100  # Desired number of output dimensions
    model = Autoencoder(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Training the autoencoder
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in loader:
            inputs, _ = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        # Loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Encoding the data using the trained autoencoder
    dataset_tensor = torch.FloatTensor(dataset_normalized.to_numpy())
    # encoded_data = model.encoder(dataset_tensor).detach().numpy()
    # encoded_data = model.decoder(encoded_data).detach().numpy()
    expandData = model.encoder(dataset_tensor)
    expandData.to_csv('expanded_data.csv',index=False)
    trained_data = model.forward(dataset_tensor)

    trained_data_df = pd.DataFrame(trained_data.detach().numpy())
    trained_data_df['Player id'] = player_numbers.values
    trained_data_df.to_csv('encoded_data.csv',index=False)

    return trained_data

