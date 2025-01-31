## Installation
The project is based on Python 3.10 and PyTorch 2.3.1 All the necessary packages are in requirements.txt. We recommend creating a virtual environment using Anaconda as follows:

  1. Download and install Anaconda Python from here: https://www.anaconda.com/products/individual/

  2.  Download and install PyTorch from here: https://pytorch.org/
      
  3. Enter the following commands to create a virtual environment:
     <pre> conda create -n my_env python=3.10 anaconda
      activate my_env
      pip install -r requirements.txt </pre>


## Overview

This project focuses on processing and analyzing soccer player performance data using an autoencoder model to transform and visualize player statistics. The primary goal is to understand player performance differences by reducing and analyzing complex data representations.


The workflow consists of several key stages:
  **1. Data Preprocessing:** Constructing a structured dataset with 16 performance-related features for each player, giving greater emphasis to recent matches.

  **2. Autoencoder Training:** Expanding the 16-dimensional dataset into a 100-dimensional representation using an autoencoder model.

  **3. Evaluation:** Assessing the model's performance on a test dataset by analyzing reconstruction loss.

  **4. Dimensionality Reduction:** Applying Principal Component Analysis (PCA) to reduce the 100-dimensional embeddings into a 2-dimensional space.

  **5. Data Visualization:** Using coloring on PCA graphs based on different features to reveal patterns and insights about player performance.

This project enables a more interpretable representation of player data, helping to uncover trends, similarities, and differences between players based on their performance metrics.


## Dataset

The dataset originates from Football Data from Transfermarkt [https://www.kaggle.com/datasets/davidcariboo/player-scores/data?select=appearances.csv]. It contains detailed player statistics for multiple seasons, including attributes such as goals, assists, minutes played, passing accuracy, defensive actions, and more. Before training the model, the raw data undergoes cleaning, normalization, and transformation to ensure consistency and accuracy.

## I think should be removed
ED.py - defines an autoencoder model in PyTorch with an encoder to compress data and a decoder to reconstruct it. It uses linear layers with activation functions like ReLU and Sigmoid.       

coloring.py - processes soccer player data, visualizes it using PCA and scatter plots, and supports data preparation and player comparisons.

performance_vectors.py - processes soccer player data, computes features, splits it into train and test sets, and trains an autoencoder to evaluate performance.

trainTest.py - trains an autoencoder on normalized soccer player data, evaluates it on a test set, and saves both encoded and expanded representations of the data to CSV files.
