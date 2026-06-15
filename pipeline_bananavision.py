import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np

# ---------- Module A : CNN extracteur d'embeddings ----------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

def load_cnn(weights_path="best_efficientnet_b0_binary.pth"):
    cnn = FeatureExtractor()
    # Charger les poids du Module A (entraîné)
    state_dict = torch.load(weights_path, map_location='cpu')
    # On filtre les clés du classifier qui n'existent pas dans FeatureExtractor
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    cnn.load_state_dict(state_dict, strict=False)
    cnn.eval()
    return cnn

def get_embedding(image, cnn, transform):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        emb = cnn(img_tensor).numpy().flatten()
    return emb

# ---------- Module B : clustering ----------
def load_clustering_models():
    scaler = joblib.load("scaler_clustering.joblib")
    kmeans = joblib.load("kmeans_model.joblib")
    return scaler, kmeans

def predict_cluster(embedding, scaler, kmeans):
    emb_scaled = scaler.transform([embedding])
    return kmeans.predict(emb_scaled)[0]

# ---------- Module C : politique de décision (MDP) ----------
# Ici, vous intégrez votre politique optimale. Exemple simpliste :
def decision_from_cluster(cluster_id, confidence=None):
    # À remplacer par votre vraie politique (Value Iteration)
    mapping = {
        0: "Accepter Export",
        1: "Accepter Catégorie II",
        2: "Rediriger Transformation",
        3: "Suspendre contrôle manuel",
    }
    return mapping.get(cluster_id, "Suspendre contrôle manuel")

# ---------- Pipeline complet ----------
def analyze_image(image_path):
    # 1. Charger les modèles (à mettre en cache si utilisé plusieurs fois)
    cnn = load_cnn()
    scaler, kmeans = load_clustering_models()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Charger l'image
    image = Image.open(image_path).convert('RGB')

    # 3. Extraire l'embedding (Module A)
    emb = get_embedding(image, cnn, transform)

    # 4. Prédire le cluster (Module B)
    cluster = predict_cluster(emb, scaler, kmeans)

    # 5. Décision finale (Module C)
    decision = decision_from_cluster(cluster)

    return {
        "cluster": cluster,
        "decision": decision,
        "embedding": emb  # optionnel
    }

if __name__ == "__main__":
    # Test
    result = analyze_image("test_banane.jpg")
    print(f"Cluster : {result['cluster']}")
    print(f"Décision : {result['decision']}")