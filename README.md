**Overview**

This project focuses on processing and analyzing soccer player performance data using an autoencoder model to transform and visualize player statistics. The primary goal is to understand player performance differences by reducing and analyzing complex data representations.


The workflow consists of several key stages:
Data Preprocessing: Constructing a structured dataset with 16 performance-related features for each player, giving greater emphasis to recent matches.
**1. Autoencoder Training:** Expanding the 16-dimensional dataset into a 100-dimensional representation using an autoencoder model.

**E2. valuation:** Assessing the model's performance on a test dataset by analyzing reconstruction loss.

**3. Dimensionality Reduction:** Applying Principal Component Analysis (PCA) to reduce the 100-dimensional embeddings into a 2-dimensional space.

**4. Data Visualization:** Using color-coded PCA graphs based on different features to reveal patterns and insights about player performance.

This project enables a more interpretable representation of player data, helping to uncover trends, similarities, and differences between players based on their performance metrics.

ED.py - defines an autoencoder model in PyTorch with an encoder to compress data and a decoder to reconstruct it. It uses linear layers with activation functions like ReLU and Sigmoid.       

coloring.py - processes soccer player data, visualizes it using PCA and scatter plots, and supports data preparation and player comparisons.

performance_vectors.py - processes soccer player data, computes features, splits it into train and test sets, and trains an autoencoder to evaluate performance.

trainTest.py - trains an autoencoder on normalized soccer player data, evaluates it on a test set, and saves both encoded and expanded representations of the data to CSV files.
