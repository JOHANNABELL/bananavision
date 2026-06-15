# banana_vision_app.py
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import joblib
import gradio as gr
from PIL import Image
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from module_c.mdp_engine import MDPEngine
from module_c.state_builder import build_state

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BINARY_MODEL_PATH = "outputs/module_a/best_efficientnet_b0_binary.pth"
MULTICLASS_MODEL_PATH = "outputs/module_a/best_efficientnet_b0_multiclass.pth"
SCALER_PATH = "scaler.joblib"
KMEANS_PATH = "kmeans_model.joblib"
CLUSTER_INFO_PATH = "cluster_info.json"
CORRECTIONS_FILE = "user_corrections.json"
TRACE_FILE = "quality_trace.json"
MDP_POLICY_PATH = "module_c/politique_optimale.json"
MODULE_D_MD_PATH = "docs/module_d_ethique.md"

MULTICLASS_CLASS_NAMES = [
    "mure_malade",
    "mure_saint",
    "tropmure_malade",
    "tropmure_saint",
    "vert_malade",
    "vert_sain",
]

MULTICLASS_DISPLAY = {
    "mure_malade": {
        "nom": "Mûr malade",
        "defaut": "maladie_fongique",
        "description": "Banane mûre présentant des signes de maladie ou d'altération.",
    },
    "mure_saint": {
        "nom": "Mûr sain",
        "defaut": "maturite_non_export",
        "description": "Banane mûre saine, mais hors cible export verte définie pour ce prototype.",
    },
    "tropmure_malade": {
        "nom": "Trop mûr malade",
        "defaut": "maladie_fongique",
        "description": "Banane trop mûre avec signes de dégradation.",
    },
    "tropmure_saint": {
        "nom": "Trop mûr sain",
        "defaut": "surmaturite",
        "description": "Banane trop mûre saine, à rediriger hors export vert.",
    },
    "vert_malade": {
        "nom": "Vert malade",
        "defaut": "maladie_fongique",
        "description": "Banane verte présentant des signes de maladie.",
    },
    "vert_sain": {
        "nom": "Vert sain",
        "defaut": None,
        "description": "Banane verte saine, cible export du système BananaVision.",
    },
}

CLUSTER_COLORS = {
    "mure_malade": "#c2410c",
    "mure_saint": "#eab308",
    "tropmure_malade": "#b45309",
    "tropmure_saint": "#f97316",
    "vert_malade": "#d97706",
    "vert_sain": "#22c55e",
}

CLUSTER_INFO = {
    0: {"nom": "Vert malade", "couleur": "#d97706", "qualite": "rejetee", "defaut": "maladie_fongique", "description": "Banane verte présentant des signes de dégradation fongique"},
    1: {"nom": "Trop mûr malade", "couleur": "#b45309", "qualite": "rejetee", "defaut": "maladie_fongique", "description": "Banane à maturité avancée avec atteintes fongiques"},
    2: {"nom": "Trop mûr sain", "couleur": "#f97316", "qualite": "rejetee", "defaut": "surmaturite", "description": "Banane très mûre, saine mais hors cible export verte"},
    3: {"nom": "Mûr sain", "couleur": "#eab308", "qualite": "rejetee", "defaut": "maturite_non_export", "description": "Banane mûre saine, non exportée par la règle vert sain"},
    4: {"nom": "Mûr malade", "couleur": "#c2410c", "qualite": "rejetee", "defaut": "maladie_fongique", "description": "Banane mûre présentant des infections de surface"},
    5: {"nom": "Vert sain", "couleur": "#22c55e", "qualite": "export", "defaut": None, "description": "Banane verte saine, parfaite pour le transport long"}
}


def load_cluster_info():
    if not os.path.exists(CLUSTER_INFO_PATH):
        return CLUSTER_INFO

    with open(CLUSTER_INFO_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    cluster_map = {}
    for cluster_id, label_key in metadata.get("cluster_to_label", {}).items():
        info = MULTICLASS_DISPLAY.get(label_key)
        if info is None:
            continue
        cluster_map[int(cluster_id)] = {
            "label_key": label_key,
            "nom": info["nom"],
            "couleur": CLUSTER_COLORS.get(label_key, "#64748b"),
            "qualite": "export" if label_key == "vert_sain" else "rejetee",
            "defaut": info["defaut"],
            "description": info["description"],
        }
    return cluster_map or CLUSTER_INFO


CLUSTER_INFO = load_cluster_info()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Modèles
# -------------------------------
def build_binary_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1)
    )
    model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def build_multiclass_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 6)
    )
    model.load_state_dict(torch.load(MULTICLASS_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def build_feature_extractor():
    base_model = models.efficientnet_b0(weights=None)
    extractor = FeatureExtractor(base_model)
    state_dict = torch.load(MULTICLASS_MODEL_PATH, map_location=device)
    filtered_state = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    extractor.load_state_dict(filtered_state, strict=False)
    extractor.to(device)
    extractor.eval()
    return extractor

binary_model = build_binary_model()
multiclass_model = build_multiclass_model()
feature_extractor = build_feature_extractor()
scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)
mdp_engine = MDPEngine(MDP_POLICY_PATH)

# -------------------------------
# Prédiction
# -------------------------------
def predict_binary(img_tensor):
    with torch.no_grad():
        logit_bin = binary_model(img_tensor).squeeze()
        # ImageFolder entraîne l'index 1 comme "bananes_rejetees".
        # La sigmoid représente donc P(rejet), pas P(export).
        prob_rejet = torch.sigmoid(logit_bin).item()
    prob_export = 1.0 - prob_rejet
    decision_quality = "export" if prob_export >= 0.5 else "rejetee"
    return {
        "decision_quality": decision_quality,
        "prob_export": prob_export,
        "prob_rejet": prob_rejet,
        "binary_conf": prob_export if decision_quality == "export" else prob_rejet,
        "binary_label": "Banane verte saine exportable"
        if decision_quality == "export"
        else "Non exportable",
    }


def predict_multiclass(img_tensor):
    with torch.no_grad():
        logits_multi = multiclass_model(img_tensor).squeeze()
        probs_multi = torch.softmax(logits_multi, dim=0).cpu().numpy()
    pred_class_idx = np.argmax(probs_multi)
    pred_class_key = MULTICLASS_CLASS_NAMES[pred_class_idx]
    pred_class_info = MULTICLASS_DISPLAY[pred_class_key]
    return {
        "multiclass_key": pred_class_key,
        "multiclass_label": pred_class_info["nom"],
        "multiclass_description": pred_class_info["description"],
        "multiclass_defaut": pred_class_info["defaut"],
        "multiclass_probs": {
            MULTICLASS_DISPLAY[name]["nom"]: float(probs_multi[i])
            for i, name in enumerate(MULTICLASS_CLASS_NAMES)
        },
    }


def predict_cluster(img_tensor):
    with torch.no_grad():
        embedding = feature_extractor(img_tensor).cpu().numpy().reshape(1, -1)
    embedding = embedding.astype(np.float32)
    embedding_scaled = scaler.transform(embedding)
    embedding_scaled = embedding_scaled.astype(kmeans.cluster_centers_.dtype, copy=False)
    cluster_id = kmeans.predict(embedding_scaled)[0]
    centroid = kmeans.cluster_centers_[cluster_id]
    dist = np.linalg.norm(embedding_scaled - centroid)
    cluster_conf = 1.0 / (1.0 + dist)
    cluster_info = CLUSTER_INFO.get(cluster_id, {
        "nom": f"Cluster {cluster_id}",
        "couleur": "#64748b",
        "qualite": "rejetee",
        "defaut": "profil_inconnu",
        "description": "Profil non supervisé détecté par K-Means.",
    })
    return {
        "cluster_id": int(cluster_id),
        "cluster_label_key": cluster_info.get("label_key"),
        "cluster_nom": cluster_info["nom"],
        "cluster_couleur": cluster_info["couleur"],
        "cluster_description": cluster_info["description"],
        "cluster_qualite": cluster_info["qualite"],
        "cluster_defaut": cluster_info["defaut"],
        "cluster_confiance": cluster_conf,
    }


def predict_mdp(pred):
    state = build_state(pred)
    action = mdp_engine.get_optimal_action(state["state_id"])
    return {
        "mdp_state_id": state["state_id"],
        "mdp_group": state["group"],
        "mdp_confidence": state["confidence"],
        "mdp_alert": state["alert"],
        "mdp_score_global": state["score_global"],
        "mdp_state_label": state["libelle"],
        "mdp_action": action["action"],
        "mdp_action_code": action["code_action"],
        "mdp_action_label": action["libelle"],
        "mdp_action_index": action["action_index"],
        "mdp_v_star": action["V_star"],
        "mdp_q_star": action["Q_star"],
        "mdp_justification": action["justification"],
        "mdp_suspendre": action["suspendre"],
    }


def predict_pipeline(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    binary_pred = predict_binary(img_tensor)

    if binary_pred["decision_quality"] == "export":
        quality_display = "EXPORT"
        result = {
            "pipeline_steps": "Étape 1 : CNN binaire -> export. Arrêt du pipeline.",
            "final_label": "Vert sain",
            "final_label_key": "vert_sain",
            "cluster_id": None,
            "cluster_label_key": "vert_sain",
            "cluster_nom": "Non utilisé",
            "cluster_couleur": "#22c55e",
            "cluster_description": "Le K-Means n'est pas appelé quand le filtre binaire confirme une banane exportable.",
            "cluster_confiance": 1.0,
            "quality_display": quality_display,
            "badge_bg": "#22c55e",
            "defect_display": "Aucun défaut détecté",
            "binary_label": binary_pred["binary_label"],
            "binary_conf": binary_pred["binary_conf"],
            "prob_export": binary_pred["prob_export"],
            "prob_rejet": binary_pred["prob_rejet"],
            "multiclass_label": "Vert sain",
            "multiclass_description": "Le modèle multiclasse est ignoré : la décision binaire suffit pour exporter.",
            "sequence_summary": "CNN binaire : banane verte saine exportable. Aucun traitement complémentaire nécessaire.",
            "multiclass_probs": {"Vert sain": 1.0},
        }
        result.update(predict_mdp(result))
        return result

    multiclass_pred = predict_multiclass(img_tensor)
    cluster_pred = predict_cluster(img_tensor)

    decision_quality = cluster_pred["cluster_qualite"]
    quality_display = "EXPORT" if decision_quality == "export" else "REJETÉE"
    badge_bg = "#22c55e" if decision_quality == "export" else "#ef4444"
    decision_defaut = cluster_pred["cluster_defaut"] or multiclass_pred["multiclass_defaut"]
    defect_display = decision_defaut if decision_defaut else "Aucun défaut détecté"

    result = {
        "pipeline_steps": "Étape 1 : CNN binaire -> rejet. Étape 2 : CNN multiclasse. Étape 3 : K-Means -> réponse finale.",
        "final_label": cluster_pred["cluster_nom"],
        "final_label_key": cluster_pred["cluster_label_key"] or multiclass_pred["multiclass_key"],
        "cluster_id": cluster_pred["cluster_id"],
        "cluster_label_key": cluster_pred["cluster_label_key"],
        "cluster_nom": cluster_pred["cluster_nom"],
        "cluster_couleur": cluster_pred["cluster_couleur"],
        "cluster_description": cluster_pred["cluster_description"],
        "cluster_confiance": cluster_pred["cluster_confiance"],
        "quality_display": quality_display,
        "badge_bg": badge_bg,
        "defect_display": defect_display,
        "binary_label": binary_pred["binary_label"],
        "binary_conf": binary_pred["binary_conf"],
        "prob_export": binary_pred["prob_export"],
        "prob_rejet": binary_pred["prob_rejet"],
        "multiclass_label": multiclass_pred["multiclass_label"],
        "multiclass_description": multiclass_pred["multiclass_description"],
        "sequence_summary": f"Réponse finale affinée par K-Means : {cluster_pred['cluster_nom']}.",
        "multiclass_probs": multiclass_pred["multiclass_probs"],
    }
    result.update(predict_mdp(result))
    return result

# -------------------------------
# Sauvegarde des corrections
# -------------------------------
def save_correction(image, predicted_quality, predicted_defect, user_feedback,
                    correct_quality, correct_defect):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "predicted_quality": predicted_quality,
        "predicted_defect": predicted_defect,
        "user_feedback": user_feedback,
        "correct_quality": correct_quality if user_feedback == "invalid" else predicted_quality,
        "correct_defect": correct_defect if user_feedback == "invalid" else predicted_defect
    }
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "r") as f:
            corrections = json.load(f)
    else:
        corrections = []
    corrections.append(entry)
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)
    return "Système mis à jour."


def load_trace_entries():
    if not os.path.exists(TRACE_FILE):
        return []
    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return entries if isinstance(entries, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def write_trace_entries(entries):
    with open(TRACE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def save_trace(pred, feedback=None, corrections=None, trace_id=None):
    entries = load_trace_entries()
    now = datetime.now().isoformat(timespec="seconds")

    if trace_id:
        for entry in entries:
            if entry.get("trace_id") == trace_id:
                entry["feedback"] = feedback or entry.get("feedback")
                entry["feedback_timestamp"] = now
                entry["corrections"] = corrections or entry.get("corrections")
                write_trace_entries(entries)
                return trace_id

    trace_id = str(uuid.uuid4())
    entry = {
        "trace_id": trace_id,
        "timestamp": now,
        "decision_finale": pred["quality_display"],
        "classe_finale": pred["final_label"],
        "defaut": pred["defect_display"],
        "prob_export": round(float(pred["prob_export"]), 6),
        "prob_rejet": round(float(pred["prob_rejet"]), 6),
        "classe_multiclasse": pred["multiclass_label"],
        "cluster": pred["cluster_nom"],
        "cluster_confiance": round(float(pred["cluster_confiance"]), 6),
        "pipeline": pred["pipeline_steps"],
        "mdp_state_id": pred.get("mdp_state_id"),
        "mdp_state_label": pred.get("mdp_state_label"),
        "mdp_action": pred.get("mdp_action"),
        "mdp_action_label": pred.get("mdp_action_label"),
        "mdp_justification": pred.get("mdp_justification"),
        "mdp_v_star": pred.get("mdp_v_star"),
        "mdp_suspendre": pred.get("mdp_suspendre"),
        "feedback": feedback,
        "feedback_timestamp": None,
        "corrections": corrections,
    }
    entries.append(entry)
    write_trace_entries(entries)
    return trace_id


def _pct(part, total):
    return (100.0 * part / total) if total else 0.0


def _distribution_html(title, counter, total):
    if not counter:
        return f"""
        <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:16px;">
            <h3 style="margin:0 0 10px 0; font-size:15px; color:#111827;">{title}</h3>
            <p style="margin:0; color:#64748b; font-size:13px;">Aucune donnée disponible.</p>
        </div>
        """

    rows = []
    for label, count in counter.most_common():
        percent = _pct(count, total)
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#374151; margin-bottom:4px;">
                <span>{label}</span><strong>{count} ({percent:.1f}%)</strong>
            </div>
            <div style="height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden;">
                <div style="height:100%; width:{percent:.1f}%; background:#2563eb;"></div>
            </div>
        </div>
        """)

    return f"""
    <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:16px;">
        <h3 style="margin:0 0 12px 0; font-size:15px; color:#111827;">{title}</h3>
        {''.join(rows)}
    </div>
    """


def build_dashboard():
    entries = load_trace_entries()
    total = len(entries)
    exports = sum(1 for e in entries if e.get("decision_finale") == "EXPORT")
    rejects = sum(1 for e in entries if e.get("decision_finale") == "REJETÉE")
    validations = sum(1 for e in entries if e.get("feedback") == "valid")
    corrections = sum(1 for e in entries if e.get("feedback") == "invalid")
    suspensions = sum(1 for e in entries if e.get("mdp_suspendre") is True)
    v_values = [float(e["mdp_v_star"]) for e in entries if e.get("mdp_v_star") is not None]
    v_mean = sum(v_values) / len(v_values) if v_values else 0.0
    correction_rate = _pct(corrections, validations + corrections)
    reject_rate = _pct(rejects, total)

    if total == 0:
        alert_label = "En attente"
        alert_text = "Aucune analyse enregistrée pour le moment."
        alert_color = "#64748b"
    elif correction_rate >= 20:
        alert_label = "Vigilance"
        alert_text = "Le taux de corrections humaines est élevé."
        alert_color = "#dc2626"
    elif reject_rate >= 70:
        alert_label = "Attention"
        alert_text = "Le taux de rejet est élevé sur les dernières analyses."
        alert_color = "#d97706"
    else:
        alert_label = "Stable"
        alert_text = "Les indicateurs qualité sont cohérents pour la démo."
        alert_color = "#16a34a"

    kpi_html = f"""
    <div style="font-family:system-ui, -apple-system, sans-serif;">
        <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:14px; margin-bottom:14px;">
            <strong style="font-size:14px; color:#1e293b;">Gouvernance et traçabilité - Module D</strong>
            <p style="margin:6px 0 0 0; color:#64748b; font-size:13px;">
                Ce dashboard matérialise le suivi éthique demandé dans le Module D :
                journal des décisions, corrections humaines, contrôles MDP, taux de rejet et distribution des actions métier.
            </p>
        </div>
        <div style="display:grid; grid-template-columns:repeat(6, minmax(130px, 1fr)); gap:12px; margin-bottom:14px;">
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Images analysées</p>
                <strong style="font-size:28px; color:#111827;">{total}</strong>
            </div>
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Taux export</p>
                <strong style="font-size:28px; color:#16a34a;">{_pct(exports, total):.1f}%</strong>
            </div>
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Taux rejet</p>
                <strong style="font-size:28px; color:#dc2626;">{reject_rate:.1f}%</strong>
            </div>
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Feedback humain</p>
                <strong style="font-size:28px; color:#111827;">{validations + corrections}</strong>
            </div>
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Contrôles manuels MDP</p>
                <strong style="font-size:28px; color:#b91c1c;">{suspensions}</strong>
            </div>
            <div style="background:#ffffff; border:1px solid {alert_color}; border-radius:8px; padding:14px;">
                <p style="margin:0; color:#64748b; font-size:12px;">Statut qualité</p>
                <strong style="font-size:22px; color:{alert_color};">{alert_label}</strong>
                <p style="margin:4px 0 0 0; color:#64748b; font-size:12px;">{alert_text}</p>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4, minmax(160px, 1fr)); gap:12px;">
            <div style="background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:12px;">
                <span style="font-size:13px; color:#475569;">Validations</span>
                <strong style="float:right; color:#16a34a;">{validations}</strong>
            </div>
            <div style="background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:12px;">
                <span style="font-size:13px; color:#475569;">Corrections</span>
                <strong style="float:right; color:#dc2626;">{corrections}</strong>
            </div>
            <div style="background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:12px;">
                <span style="font-size:13px; color:#475569;">Taux correction</span>
                <strong style="float:right; color:#111827;">{correction_rate:.1f}%</strong>
            </div>
            <div style="background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:12px;">
                <span style="font-size:13px; color:#475569;">V* moyen MDP</span>
                <strong style="float:right; color:#111827;">{v_mean:.1f} FCFA</strong>
            </div>
        </div>
    </div>
    """

    class_counter = Counter(e.get("classe_finale", "Non renseigné") for e in entries)
    defect_counter = Counter(e.get("defaut", "Non renseigné") for e in entries)
    action_counter = Counter(e.get("mdp_action", "Non renseigné") for e in entries)
    distribution_html = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px;">
        {_distribution_html("Distribution des classes finales", class_counter, total)}
        {_distribution_html("Distribution des défauts", defect_counter, total)}
        {_distribution_html("Distribution des actions MDP", action_counter, total)}
    </div>
    """

    recent_rows = []
    for entry in list(reversed(entries))[:20]:
        recent_rows.append([
            entry.get("timestamp", ""),
            entry.get("decision_finale", ""),
            entry.get("classe_finale", ""),
            entry.get("defaut", ""),
            f"{entry.get('prob_export', 0):.1%}",
            entry.get("cluster", ""),
            f"{entry.get('cluster_confiance', 0):.1%}",
            entry.get("mdp_action") or "",
            entry.get("mdp_state_id") if entry.get("mdp_state_id") is not None else "",
            f"{entry.get('mdp_v_star', 0):.1f}" if entry.get("mdp_v_star") is not None else "",
            entry.get("feedback") or "en attente",
        ])

    return kpi_html, distribution_html, recent_rows


def load_module_d_markdown():
    path = Path(MODULE_D_MD_PATH)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return """
# Module D - Analyse ethique et sociale

Le document Module D est attendu dans `docs/module_d_ethique.md`.

Points suivis dans l'application :

- emploi et supervision humaine ;
- biais dataset et audit des lots ;
- gouvernance des erreurs ;
- tracabilite par journal qualite, feedback humain et decisions MDP.
"""

# -------------------------------
# Interface Gradio Logic
# -------------------------------
def process_and_validate(image, feedback, correct_quality, correct_defect, trace_id):
    if image is None:
        default_html = """
        <div style='text-align: center; padding: 50px; color: #6b7280; border: 2px dashed #e5e7eb; border-radius: 12px;'>
            <p style='font-size: 16px;'>📸 En attente d'une image pour lancer l'analyse.</p>
        </div>
        """
        return default_html, gr.update(visible=False), trace_id

    pred = predict_pipeline(image)
    if feedback is None:
        trace_id = save_trace(pred)
    elif trace_id is None:
        trace_id = save_trace(pred)
    conf_pct = int(pred['cluster_confiance'] * 100)

    # Construction des lignes de probabilité pour l'accordéon
    probs_lines = "".join([
        f"""<div style='display:flex; justify-content:space-between; margin-bottom:4px; font-size:13px; color:#4b5563;'>
            <span>{name}</span><strong>{prob:.1%}</strong>
        </div>""" for name, prob in pred['multiclass_probs'].items()
    ])

    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #ffffff; border-radius: 16px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); padding: 24px; border-top: 8px solid {pred['cluster_couleur']};">
        
        <div style="margin-bottom: 20px;">
            <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {pred['cluster_couleur']};">Décision séquentielle détectée</span>
            <h2 style="margin: 4px 0; font-size: 24px; font-weight: 800; color: #111827;">{pred['final_label']}</h2>
            <p style="margin: 0; font-size: 14px; color: #4b5563;">{pred['sequence_summary']}</p>
        </div>

        <div style="margin-bottom: 24px; background: #f3f4f6; border-radius: 9999px; height: 8px; overflow: hidden;">
            <div style="background: {pred['cluster_couleur']}; width: {conf_pct}%; height: 100%; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: -20px; margin-bottom: 24px; font-size: 12px; color: #6b7280;">
            <span>Indice de correspondance K-Means complémentaire</span>
            <strong>{conf_pct}%</strong>
        </div>

        <div style="background: #f9fafb; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #f3f4f6; margin-bottom: 20px;">
            <span style="background: {pred['badge_bg']}; color: #ffffff; padding: 6px 16px; border-radius: 9999px; font-size: 14px; font-weight: 700; letter-spacing: 0.05em;">
                {pred['quality_display']}
            </span>
            <h3 style="margin: 16px 0 4px 0; font-size: 15px; color: #6b7280; font-weight: 500;">Diagnostic multiclasse</h3>
            <p style="margin: 0; font-size: 18px; font-weight: 700; color: #1f2937;">{pred['defect_display']}</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #64748b;">{pred['multiclass_description']}</p>
        </div>

        <div style="background: #fff7ed; border-radius: 12px; padding: 18px; border: 1px solid #fed7aa; margin-bottom: 20px;">
            <span style="font-size: 12px; font-weight: 800; color: #c2410c; text-transform: uppercase;">Module C - Décision MDP</span>
            <h3 style="margin: 6px 0 6px 0; font-size: 18px; color: #111827;">{pred['mdp_action_label']}</h3>
            <p style="margin: 0 0 8px 0; font-size: 13px; color: #475569;">État S{pred['mdp_state_id']} - {pred['mdp_state_label']}</p>
            <p style="margin: 0; font-size: 13px; color: #7c2d12;">{pred['mdp_justification']}</p>
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;">
                <span style="font-size:12px; background:#ffedd5; color:#9a3412; padding:5px 8px; border-radius:6px;">Score global {pred['mdp_score_global']:.1%}</span>
                <span style="font-size:12px; background:#ffedd5; color:#9a3412; padding:5px 8px; border-radius:6px;">Confiance {pred['mdp_confidence']}</span>
                <span style="font-size:12px; background:#ffedd5; color:#9a3412; padding:5px 8px; border-radius:6px;">V* {pred['mdp_v_star']:.1f} FCFA</span>
            </div>
        </div>

        <details style="border-top: 1px solid #e5e7eb; padding-top: 12px; cursor: pointer;">
            <summary style="font-size: 13px; color: #374151; font-weight: 600; list-style: none; display: flex; justify-content: space-between; align-items: center;">
                <span>📊 Métriques Deep Learning complémentaires</span>
                <span style="font-size: 10px; color:#9ca3af;">▼ Déplier</span>
            </summary>
            <div style="margin-top: 12px; background: #f9fafb; padding: 12px; border-radius: 8px;">
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #1f2937;"><strong>Pipeline :</strong> {pred['pipeline_steps']}</p>
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #1f2937;"><strong>Étape 1 - Filtre binaire :</strong> {pred['binary_label']} ({pred['binary_conf']:.1%})</p>
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #4b5563;">P(export vert sain) = {pred['prob_export']:.1%} | P(non exportable) = {pred['prob_rejet']:.1%}</p>
                <div style="height: 1px; background: #e5e7eb; margin: 8px 0;"></div>
                <p style="margin: 0 0 6px 0; font-size: 13px; color: #1f2937;"><strong>Étape 2 - Distribution multiclasse explicative :</strong></p>
                {probs_lines}
                <div style="height: 1px; background: #e5e7eb; margin: 8px 0;"></div>
                <p style="margin: 0; font-size: 13px; color: #4b5563;"><strong>Cluster K-Means :</strong> {pred['cluster_nom']} - {pred['cluster_description']}</p>
            </div>
        </details>
    </div>
    """

    if feedback == "Valider":
        save_correction(image, pred['quality_display'], pred['defect_display'], "valid", "", "")
        save_trace(pred, feedback="valid", trace_id=trace_id)
        return html + "<div style='background:#f0fdf4; color:#16a34a; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #bbf7d0;'>✅ Décision validée avec succès.</div>", gr.update(visible=False), trace_id
    elif feedback == "Invalider":
        if not correct_quality or (correct_quality == "export" and correct_defect != "Aucun"):
            return html + "<div style='background:#fef2f2; color:#dc2626; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #fecaca;'>⚠️ Erreur de cohérence : une banane 'export' ne peut avoir de défaut.</div>", gr.update(visible=True), trace_id
        save_correction(image, pred['quality_display'], pred['defect_display'], "invalid", correct_quality, correct_defect if correct_defect != "Aucun" else None)
        save_trace(
            pred,
            feedback="invalid",
            corrections={
                "correct_quality": correct_quality,
                "correct_defect": correct_defect if correct_defect != "Aucun" else None,
            },
            trace_id=trace_id,
        )
        return html + f"<div style='background:#eff6ff; color:#2563eb; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #bfdbfe;'>🔄 Correction enregistrée ! Re-classification forcée.</div>", gr.update(visible=False), trace_id
    else:
        return html, gr.update(visible=True), trace_id

# -------------------------------
# Construction de l'Interface Web
# -------------------------------
default_html_view = """
<div style='text-align: center; padding: 80px 20px; color: #9ca3af; border: 2px dashed #e5e7eb; border-radius: 16px; background: #ffffff;'>
    <p style='font-size: 24px; margin-bottom: 8px;'>📸</p>
    <p style='font-size: 15px; margin: 0; font-weight: 500;'>Chargez une image sur la gauche pour générer le rapport analytique de tri.</p>
</div>
"""

dashboard_html, dashboard_dist_html, dashboard_rows = build_dashboard()

# Correction Gradio 6.0 : Le paramètre CSS a été retiré de gr.Blocks()
with gr.Blocks(title="BananaVision Enterprise") as demo:
    current_trace_id = gr.State(None)

    with gr.Row():
        gr.HTML("""
        <div style="text-align: left; margin: 16px 0 10px 0;">
            <h1 style="font-size: 28px; font-weight: 800; color: #1e293b; margin-bottom: 4px;">BananaVision Enterprise</h1>
            <p style="font-size: 14px; color: #64748b; margin: 0;">Contrôle qualité séquentiel et traçabilité quasi temps réel pour le tri export.</p>
        </div>
        """)

    with gr.Tabs():
        with gr.Tab("Diagnostic"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=320):
                    image_input = gr.Image(type="pil", label="Flux caméra / Image source", elem_id="img_input")
                    analyze_btn = gr.Button("Lancer le diagnostic", variant="primary", size="lg")

                with gr.Column(scale=1, min_width=420):
                    output_html = gr.HTML(value=default_html_view, label="Résultat de l'analyse")

                    with gr.Group(visible=False) as feedback_row:
                        gr.HTML("<div style='padding: 10px 0;'><p style='font-size:13px; font-weight:700; color:#475569; margin: 0 0 10px 0;'>Supervision humaine - Confirmer la décision du système ?</p></div>")
                        feedback_choice = gr.Radio(choices=["Valider", "Invalider"], label=None, show_label=False)

                        with gr.Group() as correction_box:
                            gr.HTML("<p style='font-size:12px; font-weight:600; color:#64748b; margin: 10px 0 5px 0;'>Si invalide, spécifier les valeurs réelles :</p>")
                            correct_quality = gr.Radio(choices=["export", "rejetee"], label="Qualité terrain")
                            correct_defect = gr.Radio(choices=["Aucun", "defaut_mecanique", "maladie_fongique", "maturite_non_export", "surmaturite"], label="Type de défaut constaté")

                        submit_feedback_btn = gr.Button("Enregistrer le feedback", variant="secondary", size="sm")

        with gr.Tab("Dashboard Qualité"):
            with gr.Row():
                refresh_dashboard_btn = gr.Button("Rafraîchir dashboard", variant="primary")
            dashboard_kpis = gr.HTML(value=dashboard_html)
            dashboard_distributions = gr.HTML(value=dashboard_dist_html)
            dashboard_table = gr.Dataframe(
                headers=[
                    "Horodatage",
                    "Décision",
                    "Classe finale",
                    "Défaut",
                    "P(export)",
                    "Cluster",
                    "Conf. cluster",
                    "Action MDP",
                    "État MDP",
                    "V* FCFA",
                    "Feedback",
                ],
                value=dashboard_rows,
                interactive=False,
                wrap=True,
                label="Dernières analyses",
            )

        with gr.Tab("Module D - Éthique"):
            gr.Markdown(load_module_d_markdown())

    analyze_btn.click(
        fn=process_and_validate,
        inputs=[image_input, gr.State(None), gr.State(None), gr.State(None), current_trace_id],
        outputs=[output_html, feedback_row, current_trace_id]
    )

    submit_feedback_btn.click(
        fn=process_and_validate,
        inputs=[image_input, feedback_choice, correct_quality, correct_defect, current_trace_id],
        outputs=[output_html, feedback_row, current_trace_id]
    )

    refresh_dashboard_btn.click(
        fn=build_dashboard,
        inputs=[],
        outputs=[dashboard_kpis, dashboard_distributions, dashboard_table]
    )

if __name__ == "__main__":
    # Injection du thème ET du style de fond CSS ici pour être 100% conforme à Gradio 6
    demo.launch(
        theme=gr.themes.Soft(), 
        css=".gradio-container {background-color: #f8fafc;}"
    )
