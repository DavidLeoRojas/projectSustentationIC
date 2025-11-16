# umap_tsne_mnist.py (fast version)
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    import umap as umap_lib
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

outdir = Path(__file__).resolve().parents[1] / 'results'
outdir.mkdir(parents=True, exist_ok=True)

# Use sklearn digits as proxy for MNIST (offline-friendly)
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Subsample for speed
max_samples = 1000
if X.shape[0] > max_samples:
    rng = np.random.RandomState(42)
    idx = rng.choice(X.shape[0], max_samples, replace=False)
    X = X[idx]
    y = y[idx]

# Standardize
Xs = StandardScaler().fit_transform(X)

# UMAP or PCA
if HAS_UMAP:
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=y, s=10, cmap='tab10')
    plt.title('UMAP projection (digits proxy)')
    plt.colorbar(scatter)
    plt.savefig(outdir / 'umap_digits.png', dpi=150)
    plt.close()
else:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=10, cmap='tab10')
    plt.title('PCA projection (fallback for UMAP)')
    plt.colorbar(scatter)
    plt.savefig(outdir / 'umap_fallback_pca_digits.png', dpi=150)
    plt.close()

# Fast t-SNE
tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30, max_iter=500)
X_tsne = tsne.fit_transform(Xs)
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=10, cmap='tab10')
plt.title('t-SNE projection (digits proxy)')
plt.colorbar(scatter)
plt.savefig(outdir / 'tsne_digits.png', dpi=150)
plt.close()

print("Procesamiento UMAP/PCA y t-SNE completado.")
