from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# --- Generación de datos con formas NO esféricas ---
X1, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
X2, _ = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)

# Mezclamos ambos para tener formas complejas
X = np.vstack([X1, X2])

# --- Modelo DBSCAN ---
db = DBSCAN(eps=0.25, min_samples=5)
labels = db.fit_predict(X)

# --- Visualización ---
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
plt.title("DBSCAN - Formas arbitrarias y detección de ruido")
plt.show()
