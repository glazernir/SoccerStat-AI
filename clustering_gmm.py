import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

if __name__ == "__main__":
    df = pd.read_csv("expanded_data.csv")

    gmm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    df['cluster'] = gmm.fit_predict(df)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.iloc[:, :-1])

    plt.figure(figsize=(10, 6))
    for cluster_id in range(gmm.n_components):
        cluster_points = reduced_data[df['cluster'] == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:,1], label=f"Cluster {cluster_id}")

    plt.title("GMM Clustering Results")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.show()
