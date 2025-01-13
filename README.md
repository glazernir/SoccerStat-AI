This project processes soccer player data to train an autoencoder model for feature extraction. It includes data cleaning, calculating player stats, training the model, and saving the results for analysis.
The analysis involves evaluating the performance of the trained autoencoder by calculating the reconstruction loss on the test dataset, extracting and saving encoded features, and generating expanded data for further analysis of player performance trends over time.

ED.py - defines an autoencoder model in PyTorch with an encoder to compress data and a decoder to reconstruct it. It uses linear layers with activation functions like ReLU and Sigmoid. 
coloring.py - processes soccer player data, visualizes it using PCA and scatter plots, and supports data preparation and player comparisons.
performance_vectors.py - processes soccer player data, computes features, splits it into train and test sets, and trains an autoencoder to evaluate performance
trainTest.py - trains an autoencoder on normalized soccer player data, evaluates it on a test set, and saves both encoded and expanded representations of the data to CSV files
