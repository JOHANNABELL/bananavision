import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, weights_path, num_classes, device):
        super().__init__()
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def extract_embeddings(dataset_dir, weights_path, batch_size, device):
    dataset = datasets.ImageFolder(dataset_dir, transform=get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = EfficientNetFeatureExtractor(weights_path, len(dataset.classes), device).to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_embeddings = model(images).cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels.numpy().tolist())

    return np.vstack(embeddings), np.array(labels), dataset.classes


def load_embeddings_csv(csv_path, split="train"):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV embeddings vide : {csv_path}")

        feature_cols = [
            name
            for name in reader.fieldnames
            if name.startswith("feat_") and name[5:].isdigit()
        ]
        feature_cols.sort(key=lambda name: int(name[5:]))
        required = {"split", "label", "label_id"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Colonnes manquantes dans {csv_path}: {sorted(missing)}")
        if not feature_cols:
            raise ValueError(f"Aucune colonne feat_* trouvée dans {csv_path}")

        embeddings = []
        labels = []
        label_names_by_id = {}
        for row in reader:
            if split and row["split"] != split:
                continue
            label_id = int(row["label_id"])
            label_names_by_id[label_id] = row["label"]
            labels.append(label_id)
            embeddings.append([float(row[col]) for col in feature_cols])

    if not embeddings:
        split_msg = f" pour split={split}" if split else ""
        raise ValueError(f"Aucun embedding trouvé dans {csv_path}{split_msg}")

    class_names = [
        label_names_by_id[idx]
        for idx in sorted(label_names_by_id)
    ]
    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels), class_names


def build_cluster_metadata(cluster_ids, labels, class_names):
    votes = defaultdict(list)
    for cluster_id, label_id in zip(cluster_ids, labels):
        votes[int(cluster_id)].append(int(label_id))

    cluster_to_label = {}
    cluster_stats = {}
    for cluster_id, label_ids in votes.items():
        counts = Counter(label_ids)
        dominant_label_id, dominant_count = counts.most_common(1)[0]
        dominant_label = class_names[dominant_label_id]
        cluster_to_label[str(cluster_id)] = dominant_label
        cluster_stats[str(cluster_id)] = {
            "dominant_label": dominant_label,
            "purity": dominant_count / len(label_ids),
            "count": len(label_ids),
            "counts": {class_names[label_id]: count for label_id, count in counts.items()},
        }
    return cluster_to_label, cluster_stats


def parse_args():
    parser = argparse.ArgumentParser(description="BananaVision K-Means refiner trainer")
    parser.add_argument("--data_dir", default="dataset/multiclass/train")
    parser.add_argument("--weights", default="outputs/module_a/best_efficientnet_b0_multiclass.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--scaler_out", default="scaler.joblib")
    parser.add_argument("--kmeans_out", default="kmeans_model.joblib")
    parser.add_argument("--metadata_out", default="cluster_info.json")
    parser.add_argument("--embeddings_csv", default="embeddings/embeddings_multiclass.csv")
    parser.add_argument(
        "--embedding_split",
        default="train",
        help="Split du CSV embeddings utilisé pour entraîner K-Means. Utiliser vide pour tous.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    embeddings_csv = Path(args.embeddings_csv)
    embedding_split = args.embedding_split or None
    if embeddings_csv.exists():
        embeddings, labels, class_names = load_embeddings_csv(embeddings_csv, embedding_split)
        source = f"csv:{embeddings_csv}"
        print(f"[Embeddings] Chargés depuis {embeddings_csv} split={embedding_split or 'all'}")
    else:
        embeddings, labels, class_names = extract_embeddings(
            args.data_dir,
            args.weights,
            args.batch_size,
            device,
        )
        source = "recomputed"
        print("[Embeddings] CSV absent, recalcul depuis les images.")
    print(f"[Embeddings] {embeddings.shape[0]} images x {embeddings.shape[1]} features")
    print(f"[Classes] {class_names}")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings_scaled)

    cluster_to_label, cluster_stats = build_cluster_metadata(cluster_ids, labels, class_names)
    silhouette = silhouette_score(embeddings_scaled, cluster_ids)
    ari = adjusted_rand_score(labels, cluster_ids)

    joblib.dump(scaler, args.scaler_out)
    joblib.dump(kmeans, args.kmeans_out)
    metadata = {
        "source": "train_kmeans_refiner.py",
        "embedding_source": source,
        "embedding_split": embedding_split,
        "weights": args.weights,
        "data_dir": args.data_dir,
        "class_names": class_names,
        "cluster_to_label": cluster_to_label,
        "cluster_stats": cluster_stats,
        "silhouette": silhouette,
        "adjusted_rand_index": ari,
    }
    Path(args.metadata_out).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Saved] {args.scaler_out}")
    print(f"[Saved] {args.kmeans_out}")
    print(f"[Saved] {args.metadata_out}")
    print(f"[Metrics] silhouette={silhouette:.4f} | ARI={ari:.4f}")
    print("[Cluster mapping]")
    for cluster_id, label in sorted(cluster_to_label.items(), key=lambda item: int(item[0])):
        purity = cluster_stats[cluster_id]["purity"]
        count = cluster_stats[cluster_id]["count"]
        print(f"  cluster {cluster_id} -> {label} purity={purity:.1%} n={count}")


if __name__ == "__main__":
    main()
