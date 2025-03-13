# 🚀 SoccerStat-AI🚀
![image](https://github.com/user-attachments/assets/6754b550-7269-43fc-ac4c-9a9290f29c49)

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

## Steps to Run the Project:
use runProject.py file. 
1. ## Load Original Players Stats:
The project starts by loading player statistics from "datasets/appearances.csv" - the path to original data.

2. ## Prepare Data & Calculate Performance Vectors
The run_performance_vectors function processes and normalize the data. You can shorten the process by selecting only specific years for the process - check "calculate_features" function in run_performance_vectors.py

3. ## Create Train-Test Sets
Generate train and test datasets for PCA analysis:

- 'C' for splitting by competition.
- 'R' for random split.
  (If you choose split by competition, you need to specify What competitions will be included in the test set - for example, we chose only FR1 competition for     
   test. read chooseTestSet.py for a smart choice). 

4. ## Model Training
Train the autoencoder model and display the test loss.

5. ## Run PCA & Display Coloring
The run_coloring function visualizes PCA results and the coloring according to every parameter. 

6. ## Head-to-Head Player Comparison
Compare two specific players using their Player_id (Example: 89200, 40680).
Each coloring of the PCA results by each parameter will pop up in order at the end of the run.
To compare two players' stats based on the color of their dots/their location on the graph, you can click on the dot and the player ID of the dot will be printed as output. You can then pass the desired player IDs for comparision as parameters to the function head_to_head_comparison. 












