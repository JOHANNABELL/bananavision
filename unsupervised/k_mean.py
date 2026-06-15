import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def k_mean():
    df = pd.read_csv("../../DL/embeddings_train.csv")
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    K_range = range(2, 11)
    inertias = []
    silhouettes = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Nombre de clusters k')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouettes, 'ro-')
    plt.xlabel('Nombre de clusters k')
    plt.ylabel('Score de silhouette')
    plt.title('Score de silhouette')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("optimal_k_coude_silhouette.png", dpi=150)
    plt.show()

    best_k = K_range[np.argmax(silhouettes)]
    print(f"Meilleur k selon le score de silhouette : {best_k}")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_best = km_best.fit_predict(X_scaled)

    sil_vals = silhouette_samples(X_scaled, labels_best)

    df['silhouette_score'] = sil_vals

    df.to_csv("embeddings_with_silhouette.csv", index=False)
    print(f"CSV sauvegardé : embeddings_with_silhouette.csv (k={best_k})")

if __name__ == '__main__':
    k_mean()