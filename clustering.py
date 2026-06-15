import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import joblib

def clustering():
    df = pd.read_csv("../../DL/embeddings_train.csv")
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].values
    y_true = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_optimal = 6
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df['cluster_id'] = clusters

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title(f'Clusters K-Means (k={k_optimal}) – PCA')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig("clusters_pca.png", dpi=150)
    plt.show()

    print(f"\nComposition des {k_optimal} clusters (pourcentages) :")
    for c in range(k_optimal):
        print(f"\n--- Cluster {c} ---")
        compo = df[df['cluster_id'] == c]['label'].value_counts(normalize=True)
        print(compo.to_string())

    cm = pd.crosstab(df['label'], df['cluster_id'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in cm.columns],
                yticklabels=cm.index)
    plt.xlabel('Cluster prédit')
    plt.ylabel('Classe réelle')
    plt.title('Matrice de correspondance classes réelles vs clusters')
    plt.tight_layout()
    plt.savefig("matrice_clusters_vs_reels.png", dpi=150)
    plt.show()

    ari = adjusted_rand_score(y_true, clusters)
    print(f"\nIndice de Rand ajusté (ARI) : {ari:.4f}")
    print("(1 = clustering parfait, 0 = aléatoire)")

    df.to_csv("embeddings_with_clusters.csv", index=False)
    print("Données enrichies sauvegardées dans 'embeddings_with_clusters.csv'")

    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(kmeans, "kmeans_model.joblib")
    print("Modèle de clustering et scaler sauvegardés (scaler.joblib, kmeans_model.joblib)")

    return scaler, kmeans


def predict_cluster(embedding_vector):
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans_model.joblib")

    scaled = scaler.transform([embedding_vector])
    cluster_id = kmeans.predict(scaled)[0]
    return cluster_id


if __name__ == '__main__':
    clustering()