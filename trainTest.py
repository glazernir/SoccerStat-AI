import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from ED import Autoencoder
import pandas as pd


#Train an autoencoder on the train dataset and evaluate it on the test dataset.
def train_and_test_autoencoder(train_data, test_data, batch_size=32, encoding_dim=100, num_epochs=20, learning_rate=0.003):

    def normalize_and_prepare(dataset):
        player_numbers = dataset.iloc[:, 0]
        data_to_encode = dataset.iloc[:, 1:]
        normalized_data = (data_to_encode - data_to_encode.min()) / (data_to_encode.max() - data_to_encode.min())
        tensor_data = torch.FloatTensor(normalized_data.to_numpy())
        return player_numbers, tensor_data

    train_players, train_tensor = normalize_and_prepare(train_data)
    test_players, test_tensor = normalize_and_prepare(test_data)


    train_dataset = TensorDataset(train_tensor, train_tensor)
    test_dataset = TensorDataset(test_tensor, test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    torch.manual_seed(42)

    # Initialize the autoencoder
    input_size = train_tensor.shape[1]  # Number of input features
    model = Autoencoder(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training the autoencoder
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0
        for batch in train_loader:
            inputs, _ = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

    # Evaluate the model on the test dataset
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    expandedData_train = model.encoder(train_tensor)
    expandedData_test = model.encoder(test_tensor)

    expandedData_train_df = pd.DataFrame(expandedData_train.detach().numpy())
    expandedData_test_df = pd.DataFrame(expandedData_test.detach().numpy())

    expandedData_train_df['player_id'] = train_players.values
    expandedData_train_df.to_csv('expanded_data_train.csv', index=False)

    expandedData_test_df['player_id'] = test_players.values
    expandedData_test_df.to_csv('expanded_data_test.csv', index=False)

    encoded_train = model.forward(train_tensor).detach().numpy()
    encoded_test = model.forward(test_tensor).detach().numpy()

    train_encoded_df = pd.DataFrame(encoded_train, columns=[f'Feature_{i+1}' for i in range(input_size)])
    train_encoded_df['player_id'] = train_players.values
    train_encoded_df.to_csv('encoded_train_data.csv', index=False)

    test_encoded_df = pd.DataFrame(encoded_test, columns=[f'Feature_{i+1}' for i in range(input_size)])
    test_encoded_df['player_id'] = test_players.values
    test_encoded_df.to_csv('encoded_test_data.csv', index=False)

    return model, avg_test_loss
