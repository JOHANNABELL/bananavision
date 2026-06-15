import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_clustering(csv_path="../../DL/embeddings_train.csv", k_fixed=None):
    """
    Entraîne le clustering K-Means sur les embeddings.
    Si k_fixed est donné, l'utilise. Sinon, détermine le meilleur k par silhouette.
    Sauvegarde le scaler et le modèle.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k_fixed is None:
        # Trouver le meilleur k entre 2 et 10
        best_k = 2
        best_score = -1
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        print(f"Meilleur k (silhouette) : {best_k} (score={best_score:.3f})")
    else:
        best_k = k_fixed
        print(f"Utilisation de k = {best_k} (fixé par l'utilisateur)")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Sauvegarde
    joblib.dump(scaler, "scaler_clustering.joblib")
    joblib.dump(kmeans, "kmeans_model.joblib")
    print("Modèles sauvegardés : scaler_clustering.joblib, kmeans_model.joblib")

    # Optionnel : statistiques
    df['cluster'] = clusters
    print("\nComposition des clusters :")
    for c in range(best_k):
        print(f"\nCluster {c} :")
        print(df[df['cluster'] == c]['label'].value_counts(normalize=True))

    # Matrice de confusion
    if 'label' in df.columns:
        cm = pd.crosstab(df['label'], df['cluster'])
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Correspondance classes réelles / clusters")
        plt.savefig("matrice_clusters.png")
        plt.close()

    return scaler, kmeans

def predict_cluster(embedding_vector):
    """Prédit le cluster pour un nouvel embedding"""
    scaler = joblib.load("scaler_clustering.joblib")
    kmeans = joblib.load("kmeans_model.joblib")
    scaled = scaler.transform([embedding_vector])
    return kmeans.predict(scaled)[0]

if __name__ == "__main__":
    train_clustering()