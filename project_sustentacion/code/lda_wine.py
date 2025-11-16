# lda_wine.py using WineQT.csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "datasets" / "WineQT.csv"
OUTDIR = BASE / "results"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Load dataset
df = pd.read_csv(DATASET)

# TODO: Replace this when you tell me the real label column name
label_col = "quality"   # ← AJUSTARÁN CUANDO ME DIGAS

y = df[label_col]
X = df.drop(columns=[label_col])

# Standardize
Xs = StandardScaler().fit_transform(X)

# LDA projection
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(Xs, y)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='viridis', s=40)
plt.title("LDA - WineQT Dataset")
plt.colorbar(scatter)
plt.savefig(OUTDIR / "lda_wineQT.png", dpi=150)
plt.close()

# Evaluate classification
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with open(OUTDIR / "lda_wineQT_report.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(classification_report(y_test, y_pred))

print("WineQT LDA completed.")
