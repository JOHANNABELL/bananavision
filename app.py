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
from datetime import datetime

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
        "cluster_nom": cluster_info["nom"],
        "cluster_couleur": cluster_info["couleur"],
        "cluster_description": cluster_info["description"],
        "cluster_qualite": cluster_info["qualite"],
        "cluster_defaut": cluster_info["defaut"],
        "cluster_confiance": cluster_conf,
    }


def predict_pipeline(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    binary_pred = predict_binary(img_tensor)

    if binary_pred["decision_quality"] == "export":
        quality_display = "EXPORT"
        return {
            "pipeline_steps": "Étape 1 : CNN binaire -> export. Arrêt du pipeline.",
            "final_label": "Vert sain",
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

    multiclass_pred = predict_multiclass(img_tensor)
    cluster_pred = predict_cluster(img_tensor)

    decision_quality = cluster_pred["cluster_qualite"]
    quality_display = "EXPORT" if decision_quality == "export" else "REJETÉE"
    badge_bg = "#22c55e" if decision_quality == "export" else "#ef4444"
    decision_defaut = cluster_pred["cluster_defaut"] or multiclass_pred["multiclass_defaut"]
    defect_display = decision_defaut if decision_defaut else "Aucun défaut détecté"

    return {
        "pipeline_steps": "Étape 1 : CNN binaire -> rejet. Étape 2 : CNN multiclasse. Étape 3 : K-Means -> réponse finale.",
        "final_label": cluster_pred["cluster_nom"],
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

# -------------------------------
# Interface Gradio Logic
# -------------------------------
def process_and_validate(image, feedback, correct_quality, correct_defect):
    if image is None:
        default_html = """
        <div style='text-align: center; padding: 50px; color: #6b7280; border: 2px dashed #e5e7eb; border-radius: 12px;'>
            <p style='font-size: 16px;'>📸 En attente d'une image pour lancer l'analyse.</p>
        </div>
        """
        return default_html, gr.update(visible=False)

    pred = predict_pipeline(image)
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
        return html + "<div style='background:#f0fdf4; color:#16a34a; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #bbf7d0;'>✅ Décision validée avec succès.</div>", gr.update(visible=False)
    elif feedback == "Invalider":
        if not correct_quality or (correct_quality == "export" and correct_defect != "Aucun"):
            return html + "<div style='background:#fef2f2; color:#dc2626; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #fecaca;'>⚠️ Erreur de cohérence : une banane 'export' ne peut avoir de défaut.</div>", gr.update(visible=True)
        save_correction(image, pred['quality_display'], pred['defect_display'], "invalid", correct_quality, correct_defect if correct_defect != "Aucun" else None)
        return html + f"<div style='background:#eff6ff; color:#2563eb; padding:12px; border-radius:8px; text-align:center; margin-top:12px; font-weight:600; border:1px solid #bfdbfe;'>🔄 Correction enregistrée ! Re-classification forcée.</div>", gr.update(visible=False)
    else:
        return html, gr.update(visible=True)

# -------------------------------
# Construction de l'Interface Web
# -------------------------------
default_html_view = """
<div style='text-align: center; padding: 80px 20px; color: #9ca3af; border: 2px dashed #e5e7eb; border-radius: 16px; background: #ffffff;'>
    <p style='font-size: 24px; margin-bottom: 8px;'>📸</p>
    <p style='font-size: 15px; margin: 0; font-weight: 500;'>Chargez une image sur la gauche pour générer le rapport analytique de tri.</p>
</div>
"""

# Correction Gradio 6.0 : Le paramètre CSS a été retiré de gr.Blocks()
with gr.Blocks(title="BananaVision Enterprise") as demo:
    
    # Header minimaliste et pro
    with gr.Row():
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0 10px 0;">
            <h1 style="font-size: 28px; font-weight: 800; color: #1e293b; margin-bottom: 4px;">🍌 BananaVision Enterprise</h1>
            <p style="font-size: 14px; color: #64748b; margin: 0;">Système d'évaluation de la qualité d'exportation par vision artificielle et clustering</p>
        </div>
        """)

    # Grid principale divisée proprement en 2 colonnes parfaitement symétriques
    with gr.Row(equal_height=True):
        
        # Colonne de Gauche : Input
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Flux Caméra / Image Source", elem_id="img_input")
            analyze_btn = gr.Button("🔍 Lancer le Diagnostic", variant="primary", size="lg")
        
        # Colonne de Droite : Output
        with gr.Column(scale=1):
            output_html = gr.HTML(value=default_html_view, label="Résultat de l'analyse")
            
            # Correction : Remplacement de gr.Box par gr.Group (mieux adapté et compatible Gradio 6)
            with gr.Group(visible=False) as feedback_row:
                gr.HTML("<div style='padding: 10px 0;'><p style='font-size:13px; font-weight:700; color:#475569; margin: 0 0 10px 0;'>🛠️ Supervision humaine – Confirmer la décision du système ?</p></div>")
                feedback_choice = gr.Radio(choices=["Valider", "Invalider"], label=None, show_label=False)
                
                with gr.Group() as correction_box:
                    gr.HTML("<p style='font-size:12px; font-weight:600; color:#64748b; margin: 10px 0 5px 0;'>Si invalide, spécifier les valeurs réelles :</p>")
                    correct_quality = gr.Radio(choices=["export", "rejetee"], label="Qualité Terrain")
                    correct_defect = gr.Radio(choices=["Aucun", "defaut_mecanique", "maladie_fongique", "mure"], label="Type de Défaut constaté")
                
                submit_feedback_btn = gr.Button("Enregistrer le Feedback", variant="secondary", size="sm")

    # Événements
    analyze_btn.click(
        fn=process_and_validate,
        inputs=[image_input, gr.State(None), gr.State(None), gr.State(None)],
        outputs=[output_html, feedback_row]
    )
    
    submit_feedback_btn.click(
        fn=process_and_validate,
        inputs=[image_input, feedback_choice, correct_quality, correct_defect],
        outputs=[output_html, feedback_row]
    )

if __name__ == "__main__":
    # Injection du thème ET du style de fond CSS ici pour être 100% conforme à Gradio 6
    demo.launch(
        theme=gr.themes.Soft(), 
        css=".gradio-container {background-color: #f8fafc;}"
    )
