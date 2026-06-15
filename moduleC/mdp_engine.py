"""
module_c/mdp_engine.py  [v2 — Problème 6 corrigé]
BananaVision | PHP Plantations du Haut-Penja | UCAC-ICAM 2026

CORRECTION PROBLÈME 6 :
  - Type hint Q_star : dict[str, float | None]
  - Gestion explicite de None dans get_optimal_action()
    (action masquée ne devrait jamais être optimale → exception)
"""

from __future__ import annotations
import json
from pathlib import Path

ACTION_LABELS_FR = {
    "export_cat1":    "✅ Export Catégorie I (marché européen)",
    "local_cat2":     "🟡 Marché local Catégorie II (Douala / Mungo)",
    "transformation": "🟠 Transformation industrielle (farine / jus)",
    "suspend":        "🔴 Suspension — contrôle manuel requis",
}
ACTION_CODES = {"export_cat1":"A1","local_cat2":"A2","transformation":"A3","suspend":"A4"}


class MDPEngine:
    """
    Agent de décision production.
    Chargé une fois, répond en O(1) par appel à get_optimal_action().
    """

    def __init__(self, policy_path: str | Path):
        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(
                f"politique_optimale.json introuvable : {policy_path}\n"
                "Exécuter d'abord : python module_c/train_mdp.py"
            )
        with open(policy_path, encoding="utf-8") as f:
            data = json.load(f)

        meta, etats = data["_meta"], data["etats"]
        self.gamma     = meta["gamma"]
        self.algorithm = meta["algorithme"]
        self.N_STATES  = meta["N_STATES"]
        self.V_mean    = meta["V_mean_fcfa"]

        # PROBLÈME 6 : type hint corrigé — Q_star peut contenir None (action masquée)
        self.pi_star: dict[int, str]                      = {}
        self.V_star:  dict[int, float]                    = {}
        self.Q_star:  dict[int, dict[str, float | None]]  = {}
        self.labels:  dict[int, str]                      = {}
        self.groups:  dict[int, str]                      = {}

        for sid_str, sd in etats.items():
            sid = int(sid_str)
            self.pi_star[sid] = sd["action_optimale"]
            self.V_star[sid]  = sd["V_star"]
            self.Q_star[sid]  = sd["Q_star"]
            self.labels[sid]  = sd["libelle_php"]
            self.groups[sid]  = sd["group"]

        print(f"[MDPEngine] ✅ {self.N_STATES} états | algo={self.algorithm} | γ={self.gamma} | V̄*={self.V_mean:.1f} FCFA")

    def get_optimal_action(self, state_id: int) -> dict:
        """
        Retourne la décision optimale pour state_id ∈ [0, N_STATES[.

        PROBLÈME 6 CORRIGÉ :
        - Si Q_star[action] is None (action masquée), lève une exception
          explicite plutôt que de propager None dans la sortie.
        """
        if state_id not in self.pi_star:
            raise ValueError(f"state_id={state_id} invalide (plage : 0–{self.N_STATES-1})")

        action_name = self.pi_star[state_id]
        q_val       = self.Q_star[state_id].get(action_name)

        # PROBLÈME 6 : incohérence détectée → exception explicite
        if q_val is None:
            raise RuntimeError(
                f"Incohérence : l'action optimale '{action_name}' à l'état {state_id} "
                f"a une Q-value None (action masquée). "
                f"Vérifier build_action_masks() et la construction de Q_star dans export_policy()."
            )

        action_index = {"export_cat1":0,"local_cat2":1,"transformation":2,"suspend":3}[action_name]

        return {
            "action":        action_name,
            "action_index":  action_index,
            "code_action":   ACTION_CODES[action_name],
            "libelle":       ACTION_LABELS_FR[action_name],
            "V_star":        self.V_star[state_id],
            "Q_star":        self.Q_star[state_id],   # dict[str, float | None]
            "justification": self._justification(state_id, action_name),
            "suspendre":     action_name == "suspend",
        }

    def _justification(self, sid: int, action: str) -> str:
        lbl = self.labels.get(sid, "")
        j = {
            "export_cat1":    f"{lbl}. Fruit conforme UE 1333/2011 — export premium.",
            "local_cat2":     f"{lbl}. Déclassement local Douala/Mungo — optimal pour ce groupe.",
            "transformation": f"{lbl}. Valorisation industrielle farine/jus.",
            "suspend":        f"{lbl}. Ambiguïté ou risque phytosanitaire — contrôle manuel.",
        }
        return j.get(action, lbl)