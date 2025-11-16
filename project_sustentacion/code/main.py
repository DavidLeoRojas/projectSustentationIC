# main.py
import subprocess, sys, os

scripts = [
    "umap_tsne_mnist.py",
    "lda_wine.py",
    "clustering_algorithms.py"
]

base = os.path.dirname(__file__)

for script in scripts:
    print(f"Ejecutando {script}...")
    subprocess.run([sys.executable, os.path.join(base, script)], check=True)

print("Todos los algoritmos fueron ejecutados. Revisa la carpeta /results.")
