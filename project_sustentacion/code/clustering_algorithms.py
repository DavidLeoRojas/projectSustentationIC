# clustering_algorithms.py
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
import numpy as np
import json

outdir = Path(__file__).resolve().parents[1] / 'results'
outdir.mkdir(parents=True, exist_ok=True)

X1, y1 = make_moons(n_samples=500, noise=0.08, random_state=42)
X2, y2 = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)
X3, y3 = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=42)
X3 += np.array([4.5, -1.5])

X = np.vstack([X1, X2, X3])
y = np.concatenate([y1, y2 + 2, y3 + 4])

algorithms = {
    "kmeans": KMeans(n_clusters=6),
    "dbscan": DBSCAN(eps=0.2, min_samples=5),
    "hierarchical": AgglomerativeClustering(n_clusters=6)
}

results = {}

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)

    try: ari = adjusted_rand_score(y, labels)
    except: ari = None

    try:
        if len(np.unique(labels)) > 1:
            dbi = davies_bouldin_score(X[labels>=0], labels[labels>=0])
        else:
            dbi = None
    except:
        dbi = None

    results[name] = {"ari": ari, "dbi": dbi}

    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=10)
    plt.title(f"{name} clustering")
    plt.savefig(outdir / f"cluster_{name}.png", dpi=150)
    plt.close()

with open(outdir / "clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Clustering completado.")
