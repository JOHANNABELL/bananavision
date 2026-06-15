import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode="train"):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_imagefolder_splits(data_dir, task, batch_size, num_workers):
    data_dir = Path(data_dir)
    val_name = "val" if task == "binary" else "valid"
    split_dirs = {
        "train": data_dir / "train",
        "val": data_dir / val_name,
        "test": data_dir / "test",
    }
    for split, path in split_dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"Dossier {split} introuvable : {path}")

    train_set = datasets.ImageFolder(split_dirs["train"], transform=get_transforms("train"))
    val_set = datasets.ImageFolder(split_dirs["val"], transform=get_transforms("val"))
    test_set = datasets.ImageFolder(split_dirs["test"], transform=get_transforms("test"))

    class_names = train_set.classes
    print(f"[Data] Classes : {class_names}")
    print(f"[Data] Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, class_names, train_set


def build_head(in_features, num_classes, task):
    output_dim = 1 if task == "binary" else num_classes
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, output_dim),
    )


def unfreeze_last_n_layers(feature_extractor, n):
    for layer in list(feature_extractor.children())[-n:]:
        for param in layer.parameters():
            param.requires_grad = True


def build_model(architecture, num_classes, task, freeze_backbone=True):
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            unfreeze_last_n_layers(model.features, n=3)
        model.classifier = build_head(in_features, num_classes, task)
        return model

    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            unfreeze_last_n_layers(model.features, n=5)
        model.classifier = build_head(in_features, num_classes, task)
        return model

    raise ValueError(f"Architecture inconnue : {architecture}")


def get_optimizer(model, phase):
    if phase == 1:
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=1e-4,
        )

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )


def compute_binary_pos_weight(train_set, device):
    targets = np.array(train_set.targets)
    class_zero = np.sum(targets == 0)
    class_one = np.sum(targets == 1)
    if class_one == 0:
        raise ValueError("Impossible de calculer pos_weight : aucune image de classe 1.")
    pos_weight = class_zero / class_one
    print(
        "[Binary] ImageFolder: 0=bananes_export, 1=bananes_rejetees. "
        f"pos_weight={pos_weight:.4f}"
    )
    return torch.tensor([pos_weight], dtype=torch.float32, device=device)


def make_criterion(task, train_set, device):
    if task == "binary":
        return nn.BCEWithLogitsLoss(pos_weight=compute_binary_pos_weight(train_set, device))
    return nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer, criterion, device, task):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if task == "binary":
            logits = outputs.squeeze(1)
            loss = criterion(logits, labels.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if task == "binary":
                logits = outputs.squeeze(1)
                loss = criterion(logits, labels.float())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
            else:
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": float(np.mean(all_preds == all_labels)),
        "labels": all_labels,
        "preds": all_preds,
        "probs": np.array(all_probs),
    }


def train_full(model, loaders, criterion, args, device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"best_{args.arch}_{args.task}.pth"
    phase1_path = output_dir / f"best_{args.arch}_{args.task}_phase1.pth"

    best_val_loss = float("inf")
    patience_counter = 0

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    print(f"\n[Phase 1] Feature extraction - {args.epochs_phase1} epochs")
    optimizer = get_optimizer(model, phase=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    for epoch in range(1, args.epochs_phase1 + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, criterion, device, args.task)
        val_metrics = evaluate(model, loaders["val"], criterion, device, args.task)
        scheduler.step(val_metrics["loss"])
        print(
            f"Epoch {epoch:02d}/{args.epochs_phase1} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), phase1_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Early stopping] Phase 1 epoch {epoch}")
                break

    model.load_state_dict(torch.load(phase1_path, map_location=device))
    if args.arch == "efficientnet_b0":
        unfreeze_last_n_layers(model.features, n=3)
    else:
        unfreeze_last_n_layers(model.features, n=5)

    print(f"\n[Phase 2] Fine-tuning - {args.epochs_phase2} epochs")
    optimizer = get_optimizer(model, phase=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    patience_counter = 0
    for epoch in range(1, args.epochs_phase2 + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, criterion, device, args.task)
        val_metrics = evaluate(model, loaders["val"], criterion, device, args.task)
        scheduler.step(val_metrics["loss"])
        print(
            f"Epoch FT {epoch:02d}/{args.epochs_phase2} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Early stopping] Phase 2 epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[Model] Meilleur modèle chargé : {best_path}")
    return model, best_path


def save_reports(metrics, class_names, args):
    output_dir = Path(args.output_dir)
    print("\n[Classification report]")
    print(classification_report(metrics["labels"], metrics["preds"], target_names=class_names))

    cm = confusion_matrix(metrics["labels"], metrics["preds"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Matrice de confusion")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=".1%"
    )
    axes[1].set_title("Matrice normalisée")
    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{args.arch}_{args.task}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] {cm_path}")

    roc_auc = None
    if args.task == "binary":
        fpr, tpr, _ = roc_curve(metrics["labels"], metrics["probs"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC binaire BananaVision")
        plt.legend(loc="lower right")
        roc_path = output_dir / f"roc_curve_{args.arch}.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figure] {roc_path}")

    summary = {
        "task": args.task,
        "architecture": args.arch,
        "class_names": class_names,
        "test_accuracy": metrics["accuracy"],
        "roc_auc": roc_auc,
        "binary_semantics": {
            "sigmoid_output": "P(bananes_rejetees) when task=binary",
            "export_probability": "1 - sigmoid_output",
        },
    }
    summary_path = output_dir / f"summary_{args.arch}_{args.task}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Summary] {summary_path}")


def compute_inference_time(model, device, runs=100):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    mean_ms = float(np.mean(timings))
    print(f"[Inference] {mean_ms:.2f} ms/image")
    return mean_ms


class BackboneEmbeddingExtractor(nn.Module):
    def __init__(self, model, architecture):
        super().__init__()
        self.features = model.features
        if architecture == "efficientnet_b0":
            self.pool = model.avgpool
        elif architecture == "mobilenet_v2":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Architecture inconnue pour embeddings : {architecture}")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def get_split_dirs(data_dir, task):
    data_dir = Path(data_dir)
    val_name = "val" if task == "binary" else "valid"
    return {
        "train": data_dir / "train",
        "val": data_dir / val_name,
        "test": data_dir / "test",
    }


def export_embeddings(
    model,
    architecture,
    data_dir,
    task,
    class_names,
    embeddings_dir,
    batch_size,
    num_workers,
    device,
    max_images=None,
):
    split_dirs = get_split_dirs(data_dir, task)
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"embeddings_{task}"
    csv_path = embeddings_dir / f"{output_stem}.csv"
    npy_path = embeddings_dir / f"{output_stem}.npy"

    extractor = BackboneEmbeddingExtractor(model, architecture).to(device)
    extractor.eval()

    all_embeddings = []
    metadata_rows = []
    feature_dim = None

    print(f"\n[Embeddings] Extraction officielle Module A2 -> {embeddings_dir}")
    with torch.no_grad():
        for split_name, split_path in split_dirs.items():
            if not split_path.exists():
                raise FileNotFoundError(f"Dossier {split_name} introuvable : {split_path}")

            dataset = datasets.ImageFolder(split_path, transform=get_transforms("val"))
            if dataset.classes != class_names:
                raise ValueError(
                    f"Classes incohérentes pour {split_name}: {dataset.classes} != {class_names}"
                )

            indices = list(range(len(dataset)))
            if max_images is not None:
                indices = indices[: max(0, max_images)]

            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            selected_samples = [dataset.samples[i] for i in indices]
            cursor = 0

            for images, labels in loader:
                images = images.to(device)
                batch_embeddings = extractor(images).cpu().numpy().astype(np.float32)
                if feature_dim is None:
                    feature_dim = batch_embeddings.shape[1]
                all_embeddings.append(batch_embeddings)

                batch_samples = selected_samples[cursor : cursor + len(labels)]
                cursor += len(labels)
                for (image_path, _), label_id in zip(batch_samples, labels.numpy().tolist()):
                    metadata_rows.append(
                        [
                            split_name,
                            str(Path(image_path)),
                            class_names[int(label_id)],
                            int(label_id),
                        ]
                    )

            print(f"[Embeddings] {split_name}: {len(indices)} images")

    if not all_embeddings:
        raise ValueError("Aucun embedding généré : dataset vide ou --max_images=0.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(npy_path, embeddings)

    header = ["split", "path", "label", "label_id"] + [
        f"feat_{idx}" for idx in range(embeddings.shape[1])
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row, feature_vector in zip(metadata_rows, embeddings):
            writer.writerow(row + feature_vector.tolist())

    print(f"[Embeddings] Dimension={feature_dim}")
    print(f"[Embeddings] CSV : {csv_path}")
    print(f"[Embeddings] NPY : {npy_path}")
    return csv_path, npy_path


def parse_args():
    parser = argparse.ArgumentParser(description="BananaVision Module A2 CNN")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--arch", choices=["efficientnet_b0", "mobilenet_v2"], default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs_phase1", type=int, default=10)
    parser.add_argument("--epochs_phase2", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/module_a")
    parser.add_argument("--export_embeddings", action="store_true")
    parser.add_argument("--embeddings_dir", default="embeddings")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limite par split pour tester rapidement l'extraction d'embeddings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    loaders, class_names, train_set = load_imagefolder_splits(
        args.data_dir, args.task, args.batch_size, args.num_workers
    )
    if args.task == "binary" and class_names != ["bananes_export", "bananes_rejetees"]:
        raise ValueError(
            "Pour la tâche binaire, l'ordre attendu est "
            "['bananes_export', 'bananes_rejetees']."
        )

    num_classes = 1 if args.task == "binary" else len(class_names)
    model = build_model(args.arch, num_classes, args.task, freeze_backbone=True).to(device)
    criterion = make_criterion(args.task, train_set, device)
    model, _ = train_full(model, loaders, criterion, args, device)

    test_metrics = evaluate(model, loaders["test"], criterion, device, args.task)
    save_reports(test_metrics, class_names, args)
    compute_inference_time(model, device)
    if args.export_embeddings:
        export_embeddings(
            model=model,
            architecture=args.arch,
            data_dir=args.data_dir,
            task=args.task,
            class_names=class_names,
            embeddings_dir=args.embeddings_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()
