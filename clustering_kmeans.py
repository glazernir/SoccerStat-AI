import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Load dataset from CSV file
    df = pd.read_csv(r'expanded_data_train.csv')

    # Apply KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(df)  # Assign cluster labels to data points

    # Perform PCA to reduce data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.iloc[:, :-1])  # Exclude cluster column for PCA

    # Plot the clustering results
    plt.figure(figsize=(10, 6))
    for cluster_id in range(kmeans.n_clusters):
        cluster_points = reduced_data[df['cluster'] == cluster_id]  # Select points in the current cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

    # Add plot title and labels
    plt.title("KMeans Clustering Results")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()

    # Display the plot
    plt.show()
