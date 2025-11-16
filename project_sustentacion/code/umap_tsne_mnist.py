# umap_tsne_mnist.py - Synthetic realistic dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except:
    HAS_UMAP = False

# Paths
OUTDIR = Path(__file__).resolve().parents[1] / "results"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Synthetic realistic dataset (4 clases bien definidas)
X, y = make_classification(
    n_samples=1200,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# Standardize
Xs = StandardScaler().fit_transform(X)

# UMAP or PCA fallback
if HAS_UMAP:
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=15)
    plt.title("UMAP - Realistic Synthetic Dataset")
    plt.savefig(OUTDIR / "umap_synthetic.png", dpi=150)
    plt.close()
else:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=15)
    plt.title("PCA Fallback - Synthetic Dataset")
    plt.savefig(OUTDIR / "pca_synthetic.png", dpi=150)
    plt.close()

# t-SNE (scikit-learn 1.7 compatible)
tsne = TSNE(
    n_components=2,
    init="pca",
    perplexity=40,
    max_iter=700,
    random_state=42
)

X_tsne = tsne.fit_transform(Xs)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=15)
plt.title("t-SNE - Synthetic Dataset")
plt.savefig(OUTDIR / "tsne_synthetic.png", dpi=150)
plt.close()

print("Realistic synthetic UMAP/PCA/t-SNE completed.")
