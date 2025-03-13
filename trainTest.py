import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ED import Autoencoder
import pandas as pd


# Train an autoencoder on the training dataset and evaluate it on the test dataset.
def train_and_test_autoencoder(train_data, test_data, batch_size=32, encoding_dim=32, num_epochs=20,
                               learning_rate=0.003):
    # Function to normalize the dataset and convert it into tensors for model training
    def normalize_and_prepare(dataset):
        # Remove unnecessary columns (e.g., unnamed columns from CSV files)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        # Extract player IDs (to be used later for saving results)
        player_numbers = dataset["player_id"]

        # Drop player_id column to only keep features for encoding
        data_to_encode = dataset.drop(columns=["player_id"])

        # Apply min-max normalization (scaled between 0 and 1) to prevent numerical instability
        epsilon = 1e-8  # Small constant to avoid division by zero
        normalized_data = (data_to_encode - data_to_encode.min()) / (
                    data_to_encode.max() - data_to_encode.min() + epsilon)

        # Convert DataFrame to PyTorch tensor
        tensor_data = torch.FloatTensor(normalized_data.to_numpy())

        return player_numbers, tensor_data

    # Normalize and prepare train and test datasets
    train_players, train_tensor = normalize_and_prepare(train_data)
    test_players, test_tensor = normalize_and_prepare(test_data)

    # Create PyTorch TensorDataset objects for training and testing
    train_dataset = TensorDataset(train_tensor, train_tensor)  # Autoencoder learns to reconstruct input
    test_dataset = TensorDataset(test_tensor, test_tensor)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize the autoencoder model
    input_size = train_tensor.shape[1]  # Number of input features
    model = Autoencoder(input_size, encoding_dim)

    # Define loss function (Mean Squared Error) and optimizer (Adam)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the autoencoder
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track loss per epoch

        for batch in train_loader:
            inputs, _ = batch  # Extract input features
            outputs = model(inputs)  # Forward pass (encode & decode)
            loss = criterion(outputs, inputs)  # Compute reconstruction loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss for the epoch

        # Compute average loss per epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

    # Evaluate the autoencoder on the test dataset
    model.eval()  # Set model to evaluation mode
    test_loss = 0

    with torch.no_grad():  # No gradient computation for inference
        for batch in test_loader:
            inputs, _ = batch  # Extract input features
            outputs = model(inputs)  # Forward pass (encode & decode)
            loss = criterion(outputs, inputs)  # Compute reconstruction loss
            test_loss += loss.item()  # Accumulate loss

    # Compute average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    # Encode the train and test datasets using the encoder part of the autoencoder
    expandedData_train = model.encoder(train_tensor)  # Compressed representation of train data
    expandedData_test = model.encoder(test_tensor)  # Compressed representation of test data

    # Convert encoded data to DataFrame
    expandedData_train_df = pd.DataFrame(expandedData_train.detach().numpy())
    expandedData_test_df = pd.DataFrame(expandedData_test.detach().numpy())

    # Add player IDs for reference and save as CSV
    expandedData_train_df['player_id'] = train_players.values
    expandedData_train_df.to_csv('expanded_data_train.csv', index=False)

    expandedData_test_df['player_id'] = test_players.values
    expandedData_test_df.to_csv('expanded_data_test.csv', index=False)

    # Generate fully reconstructed train and test datasets
    encoded_train = model.forward(train_tensor).detach().numpy()  # Forward pass on train set
    encoded_test = model.forward(test_tensor).detach().numpy()  # Forward pass on test set

    # Convert reconstructed data to DataFrame with feature column names
    train_encoded_df = pd.DataFrame(encoded_train, columns=[f'Feature_{i + 1}' for i in range(input_size)])
    test_encoded_df = pd.DataFrame(encoded_test, columns=[f'Feature_{i + 1}' for i in range(input_size)])

    # Add player IDs for reference and save as CSV
    train_encoded_df['player_id'] = train_players.values
    train_encoded_df.to_csv('encoded_train_data.csv', index=False)

    test_encoded_df['player_id'] = test_players.values
    test_encoded_df.to_csv('encoded_test_data.csv', index=False)

    # Return trained model and average test loss
    return model, avg_test_loss
