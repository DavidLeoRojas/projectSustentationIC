# clustering_algorithms.py - realistic synthetic clustering datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score

OUTDIR = Path(__file__).resolve().parents[1] / "results"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Generate realistic datasets
X1, y1 = make_blobs(
    n_samples=500,
    centers=[[-5, 0], [0, 5], [5, -2]],
    cluster_std=[1.2, 1.0, 1.5],
    random_state=42
)

X2, y2 = make_moons(
    n_samples=300,
    noise=0.07,
    random_state=42
)

X3, y3 = make_circles(
    n_samples=300,
    noise=0.05,
    factor=0.3,
    random_state=42
)

# Combine into one big dataset
X = np.vstack([X1, X2, X3])
y = np.concatenate([
    y1,
    y2 + 3,
    y3 + 5
])

algorithms = {
    "kmeans": KMeans(n_clusters=6, random_state=42),
    "dbscan": DBSCAN(eps=0.35, min_samples=6),
    "hierarchical": AgglomerativeClustering(n_clusters=6)
}

results = {}

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)

    # Metrics
    ari = adjusted_rand_score(y, labels) if len(np.unique(labels)) > 1 else None
    try:
        valid = labels >= 0
        dbi = davies_bouldin_score(X[valid], labels[valid]) if np.unique(labels).size > 1 else None
    except:
        dbi = None

    results[name] = {"ARI": ari, "DBI": dbi}

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=10)
    plt.title(f"{name.upper()} -Dataset")
    plt.savefig(OUTDIR / f"cluster_{name}.png", dpi=150)
    plt.close()

with open(OUTDIR / "clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Realistic clustering completed.")
