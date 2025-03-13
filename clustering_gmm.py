import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Load dataset from CSV file
    df = pd.read_csv("expanded_data_train.csv")

    # Apply Gaussian Mixture Model (GMM) clustering with 3 components (clusters)
    gmm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    df['cluster'] = gmm.fit_predict(df)  # Assign cluster labels based on GMM predictions

    # Perform PCA to reduce data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.iloc[:, :-1])  # Exclude cluster column for PCA

    # Plot the clustering results
    plt.figure(figsize=(10, 6))
    for cluster_id in range(gmm.n_components):
        cluster_points = reduced_data[df['cluster'] == cluster_id]  # Select points in the current cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

    # Add plot title and labels
    plt.title("GMM Clustering Results")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()

    # Display the plot
    plt.show()
