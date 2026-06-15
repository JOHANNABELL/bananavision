"""
module_c/state_builder.py  [v2 — corrigé]
BananaVision | PHP Plantations du Haut-Penja | UCAC-ICAM 2026

CORRECTION PROBLÈME 5 : cluster_id inconnu lève maintenant ValueError
au lieu de silencieusement tomber sur "malade".
"""

from __future__ import annotations
import unicodedata

# ── Mapping cluster → groupe métier ──────────────────────────────────────────
# Source : matrice de correspondance classes réelles vs clusters (Module B)
#   Cluster 0 = vert_malade       → malade
#   Cluster 1 = tropmure_malade   → malade
#   Cluster 2 = tropmure_sain     → tropmure_sain
#   Cluster 3 = mure_sain         → mure_sain
#   Cluster 4 = mure_malade       → malade
#   Cluster 5 = vert_sain         → vert_sain
CLUSTER_TO_GROUP: dict[int, str] = {
    0: "malade",
    1: "malade",
    2: "tropmure_sain",
    3: "mure_sain",
    4: "malade",
    5: "vert_sain",
}

LABEL_TO_GROUP: dict[str, str] = {
    "mure_malade": "malade",
    "mur_malade": "malade",
    "muremalade": "malade",
    "murmalade": "malade",
    "tropmure_malade": "malade",
    "tropmur_malade": "malade",
    "tropmuremalade": "malade",
    "tropmurmalade": "malade",
    "vert_malade": "malade",
    "vertmalade": "malade",
    "mure_sain": "mure_sain",
    "mur_sain": "mure_sain",
    "mure_saint": "mure_sain",
    "mur_saint": "mure_sain",
    "muresain": "mure_sain",
    "mursain": "mure_sain",
    "muresaint": "mure_sain",
    "mursaint": "mure_sain",
    "tropmure_sain": "tropmure_sain",
    "tropmur_sain": "tropmure_sain",
    "tropmure_saint": "tropmure_sain",
    "tropmur_saint": "tropmure_sain",
    "tropmuresain": "tropmure_sain",
    "tropmursain": "tropmure_sain",
    "tropmuresaint": "tropmure_sain",
    "tropmursaint": "tropmure_sain",
    "vert_sain": "vert_sain",
    "vert_saint": "vert_sain",
    "vertsain": "vert_sain",
    "vertsaint": "vert_sain",
}

GROUPS             = ["malade", "vert_sain", "mure_sain", "tropmure_sain"]
CONFIDENCE_LEVELS  = ["faible", "moyen", "fort"]
ALERT_VALUES       = [0, 1]

# ── Combinaisons impossibles (documentées) ────────────────────────────────────
# (malade, fort, 1)        : CNN très sûr + K-Means ambigu simultanément
# (vert_sain, faible, 0)   : CNN peu sûr sans aucune alerte — non observé
# (mure_sain, faible, 0)   : idem
# (tropmure_sain, fort, 1) : CNN très sûr + K-Means ambigu — contradiction
IMPOSSIBLE = {
    ("malade",        "fort",   1),
    ("vert_sain",     "faible", 0),
    ("mure_sain",     "faible", 0),
    ("tropmure_sain", "fort",   1),
}

_all_combos = [
    (g, c, a)
    for g in GROUPS
    for c in CONFIDENCE_LEVELS
    for a in ALERT_VALUES
    if (g, c, a) not in IMPOSSIBLE
]
assert len(_all_combos) == 20, f"Attendu 20 états, obtenu {len(_all_combos)}"

STATE_MAPPING: dict[tuple, int] = {combo: idx for idx, combo in enumerate(_all_combos)}
STATE_INDEX:   dict[int, tuple] = {idx: combo for combo, idx in STATE_MAPPING.items()}
N_STATES  = 20
N_ACTIONS = 4

GROUP_LABELS = {
    "malade":        "Fruit malade (toute maturité)",
    "vert_sain":     "Fruit vert sain — Export",
    "mure_sain":     "Fruit mûr sain — Marché local",
    "tropmure_sain": "Fruit trop mûr sain — Transformation",
}


def describe_state(sid: int) -> str:
    g, c, a = STATE_INDEX[sid]
    return f"{GROUP_LABELS[g]} — confiance {c} — {'⚠ alerte' if a else 'sans alerte'}"


def get_confidence_level(binary_conf: float, cluster_confiance: float) -> str:
    """score_global = 0.6×binary_conf + 0.4×cluster_confiance"""
    score = 0.6 * binary_conf + 0.4 * cluster_confiance
    if score < 0.60:   return "faible"
    elif score < 0.80: return "moyen"
    else:              return "fort"


def get_alert(binary_conf: float, cluster_confiance: float) -> int:
    return 1 if (binary_conf < 0.55 or cluster_confiance < 0.50) else 0


def _normalize_label(value: str | None) -> str | None:
    if not value:
        return None
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    normalized = normalized.replace("û", "u").replace("é", "e").replace("è", "e")
    normalized = normalized.replace(" ", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _group_from_prediction(pred: dict) -> str:
    """
    Adaptation BananaVision globale :
    - le Module B/K-Means actuel exporte déjà une classe finale métier ;
    - on préfère donc cette classe au mapping historique cluster_id -> groupe ;
    - l'ancien mapping reste en fallback pour préserver le travail initial.
    """
    for key in ("cluster_label_key", "final_label_key", "multiclass_key"):
        label = _normalize_label(pred.get(key))
        if label in LABEL_TO_GROUP:
            return LABEL_TO_GROUP[label]

    for key in ("classe_finale", "final_label", "cluster_nom", "multiclass_label"):
        label = _normalize_label(pred.get(key))
        if label in LABEL_TO_GROUP:
            return LABEL_TO_GROUP[label]

    if "cluster_id" in pred and pred.get("cluster_id") is not None:
        cluster_id = int(pred.get("cluster_id"))
        if cluster_id not in CLUSTER_TO_GROUP:
            raise ValueError(
                f"cluster_id={cluster_id} absent de CLUSTER_TO_GROUP "
                f"(clusters connus : {sorted(CLUSTER_TO_GROUP.keys())}). "
                "Fournir cluster_label_key/final_label pour utiliser le mapping recalibré."
            )
        return CLUSTER_TO_GROUP[cluster_id]

    raise ValueError(
        "Impossible de construire le groupe MDP : prédiction sans classe finale "
        "ni cluster_id exploitable."
    )


def _fallback_state_id(group: str, confidence: str, alert: int) -> int:
    for key in [
        (group, confidence, alert),
        (group, confidence, 0),
        (group, "moyen", 0),
    ]:
        if key in STATE_MAPPING:
            return STATE_MAPPING[key]
    for (g, c, a), sid in STATE_MAPPING.items():
        if g == group:
            return sid
    raise ValueError(f"Aucun état valide pour groupe={group}")


def build_state(pred: dict) -> dict:
    """
    Convertit la sortie de predict_pipeline() en état MDP.

    PROBLÈME 5 CORRIGÉ : cluster_id inconnu lève ValueError explicite.
    """
    binary_conf       = float(pred.get("binary_conf",       0.5))
    cluster_confiance = float(pred.get("cluster_confiance", 0.5))

    group      = _group_from_prediction(pred)
    confidence = get_confidence_level(binary_conf, cluster_confiance)
    alert      = get_alert(binary_conf, cluster_confiance)
    state_id   = _fallback_state_id(group, confidence, alert)

    return {
        "state_id":    state_id,
        "group":       group,
        "confidence":  confidence,
        "alert":       alert,
        "score_global": round(0.6 * binary_conf + 0.4 * cluster_confiance, 4),
        "libelle":     describe_state(state_id),
    }


if __name__ == "__main__":
    print(f"N_STATES = {N_STATES}")
    print(f"\nTable des 20 états :")
    for sid, (g, c, a) in STATE_INDEX.items():
        print(f"  S{sid:>2}  {g:>15}  {c:>6}  alerte={a}  {describe_state(sid)}")
    print("\n✅ state_builder.py v2 — OK")
