import csv
from datetime import datetime
from pathlib import Path

def log_result(result, csv_path="banana_vision_log.csv"):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "image", "classe_cnn", "confiance_cnn",
                "cluster_id", "filiere", "state_id", "action", "action_libelle",
                "v_star", "temps_ms"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            result["image_path"],
            result["classe_cnn"],
            f"{result['confiance_cnn']:.4f}",
            result["cluster_id"],
            result["filiere"],
            result["state_id"],
            result["action"],
            result["action_libelle"],
            f"{result['v_star']:.2f}",
            f"{result['temps_ms']:.2f}"
        ])