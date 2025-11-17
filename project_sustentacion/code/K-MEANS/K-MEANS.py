from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Generación de datos sintéticos realistas ---
X, y_true = make_blobs(
    n_samples=500,
    centers=[[-5, 0], [0, 5], [5, -2]],
    cluster_std=[1.2, 1.0, 1.5],
    random_state=42
)

# --- Modelo K-Means ---
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# --- Visualización ---
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='black', s=120, marker='X')
plt.title("K-Means Clustering - Clusters Esféricos (Sintético)")
plt.show()
