from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# --- Datos realistas ---
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

# --- Clustering jerárquico ---
hier = AgglomerativeClustering(n_clusters=4)
labels = hier.fit_predict(X)

# --- Visualización ---
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
plt.title("Clustering Jerárquico - Agrupación por similitud")
plt.show()
