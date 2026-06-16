import io
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

from dataset.data_processing import get_eval_transform
from log_results import log_result
from moduleC.mdp_engine import MDPEngine
from moduleC.state_builder import build_state

modele_cnn = "DL/best_model.pth"
scaler_cluster = "ML/unsupervised/scaler.joblib"
k_mean = "ML/unsupervised/kmeans_model.joblib"
cluster_mapping = "ML/unsupervised/cluster_to_classe.joblib"
lien_politique_mdp = "moduleC/module_c/politique_optimale.json"

class_names = [
    "mure_malade",
    "mure_sain",
    "tropmure_malade",
    "tropmure_sain",
    "vert_malade",
    "vert_sain"
]

export_class_name = "vert_sain"

ml_resources = {}


def load_resources(device):
    num_classes = len(class_names)
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(modele_cnn, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_cluster)
    kmeans = joblib.load(k_mean)
    cluster_to_filiere = joblib.load(cluster_mapping)

    engine = MDPEngine(lien_politique_mdp)

    return model, scaler, kmeans, cluster_to_filiere, class_names, engine


def predict_full_pipeline(image_bytes, image_filename, model, scaler, kmeans, class_names_list, engine, device, cluster_to_filiere):
    transform = get_eval_transform()

    img = Image.open(image_bytes).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        embedding_model = nn.Sequential(*list(model.children())[:-1])
        emb = embedding_model(img_tensor)
        emb_flat = torch.flatten(emb, start_dim=1)
    end = time.perf_counter()
    inf_time_ms = (end - start) * 1000.0

    pred_class = class_names_list[pred_idx.item()]
    confidence = conf.item()
    all_probs = probs.cpu().numpy().flatten()
    embedding = emb_flat.cpu().numpy().flatten().astype(np.float32)

    scaled_emb = scaler.transform(embedding.reshape(1, -1)).astype(np.float32)
    cluster_id = kmeans.predict(scaled_emb)[0]

    filiere = cluster_to_filiere.get(cluster_id, "inconnu")

    proba_export = all_probs[class_names_list.index(export_class_name)]
    proba_rejet = 1.0 - proba_export
    binary_conf = max(proba_export, proba_rejet)

    distances = kmeans.transform(scaled_emb)[0]
    d_min = distances[cluster_id]
    d_second = min(distances[i] for i in range(len(distances)) if i != cluster_id)
    cluster_conf = 1.0 - d_min / (d_min + d_second)

    state_input = {
        "binary_conf": float(binary_conf),
        "cluster_id": int(cluster_id),
        "cluster_confiance": float(cluster_conf),
    }
    state_info = build_state(state_input)
    state_id = state_info["state_id"]

    decision = engine.get_optimal_action(state_id)

    result = {
        "image_path": image_filename,
        "classe_cnn": pred_class,
        "confiance_cnn": confidence,
        "cluster_id": int(cluster_id),
        "filiere": filiere,
        "binary_conf": float(binary_conf),
        "cluster_conf": float(cluster_conf),
        "state_id": state_id,
        "state_libelle": state_info["libelle"],
        "action": decision["action"],
        "action_libelle": decision["libelle"],
        "v_star": decision["V_star"],
        "justification": decision["justification"],
        "temps_ms": inf_time_ms,
        "probabilites": all_probs.tolist(),
        "embedding": embedding.tolist(),
    }
    return result



@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Initialisation de l'API. Utilisation de : {device}")

    try:
        model, scaler, kmeans, cluster_to_filiere, class_names_list, engine = load_resources(device)
        ml_resources["model"] = model
        ml_resources["scaler"] = scaler
        ml_resources["kmeans"] = kmeans
        ml_resources["cluster_to_filiere"] = cluster_to_filiere
        ml_resources["class_names_list"] = class_names_list
        ml_resources["engine"] = engine
        ml_resources["device"] = device
        print("✅ Tous les modèles sont chargés en mémoire avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles : {e}")
        raise e

    yield

    print(" Arrêt de l'API. Libération des ressources.")
    ml_resources.clear()


app = FastAPI(title="BananaVisionModel API", lifespan=lifespan)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image valide.")

    try:
        image_bytes = io.BytesIO(await file.read())

        result = predict_full_pipeline(
            image_bytes=image_bytes,
            image_filename=file.filename,
            model=ml_resources["model"],
            scaler=ml_resources["scaler"],
            kmeans=ml_resources["kmeans"],
            class_names_list=ml_resources["class_names_list"],
            engine=ml_resources["engine"],
            device=ml_resources["device"],
            cluster_to_filiere=ml_resources["cluster_to_filiere"]
        )
        log_result(result, "banana_vision_log.csv")

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image : {str(e)}")


@app.get("/dashboard")
async def dashboard():
    import pandas as pd
    try:
        df = pd.read_csv("banana_vision_log.csv")
    except FileNotFoundError:
        return {"total": 0, "message": "Aucune donnée pour le moment."}

    if df.empty:
        return {"total": 0, "message": "Aucune analyse effectuée."}

    # Distribution des actions (camembert)
    action_counts = df["action_libelle"].value_counts().to_dict()

    # Distribution des filières (optionnel)
    filiere_counts = df["filiere"].value_counts().to_dict()

    # Évolution temporelle (courbe du nombre d'analyses par minute)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp_dt", inplace=True)
    # Regroupement par minute
    time_series = df.resample("1min").size().fillna(0).to_dict()
    # Conversion des clés en string pour JSON
    time_series_str = {str(k): v for k, v in time_series.items()}

    # Confiance moyenne
    confiance_moyenne = df["confiance_cnn"].astype(float).mean()

    # Temps d'inférence moyen et distribution
    temps_moyen = df["temps_ms"].astype(float).mean()
    temps_hist = df["temps_ms"].astype(float).tolist()  # pour un histogramme côté frontend

    # Dernières 10 analyses
    derniers = df.tail(10).reset_index().to_dict(orient="records")
    # Convertir les Timestamp en string pour JSON
    for d in derniers:
        d["timestamp"] = str(d["timestamp"])

    stats = {
        "total": len(df),
        "action_distribution": action_counts,
        "filiere_distribution": filiere_counts,
        "time_series": time_series_str,
        "confiance_moyenne": round(confiance_moyenne, 4),
        "temps_moyen_ms": round(temps_moyen, 2),
        "temps_histogramme": temps_hist,
        "dernieres_analyses": derniers,
    }
    return stats