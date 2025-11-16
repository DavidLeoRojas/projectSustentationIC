# lda_wine.py
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

outdir = Path(__file__).resolve().parents[1] / 'results'
outdir.mkdir(parents=True, exist_ok=True)

data = load_wine()
X = data.data
y = data.target

Xs = StandardScaler().fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(Xs, y)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='tab10', s=40)
plt.title('LDA projection (Wine dataset)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar(scatter)
plt.savefig(outdir / 'lda_wine.png', dpi=150)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with open(outdir / 'lda_wine_report.txt', 'w') as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(classification_report(y_test, y_pred))

print("LDA completado. Accuracy:", acc)
