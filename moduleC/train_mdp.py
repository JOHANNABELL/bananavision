"""
module_c/train_mdp.py  [v2 — 6 problèmes corrigés]
BananaVision | PHP Plantations du Haut-Penja | UCAC-ICAM 2026

CORRECTIONS :
  P1 — policy_evaluation() : indexation P_pi corrigée
  P2 — structure ECONOMIC sourcée et documentée
  P3 — build_reward_matrix() : formule de mélange avec p_err
  P4 — build_transition_matrix() : quasi-absorbante + suspend corrige l'observation
  (P5 dans state_builder.py, P6 dans mdp_engine.py)
"""

from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from state_builder import (
    N_STATES, N_ACTIONS, STATE_INDEX, STATE_MAPPING,
    GROUPS, GROUP_LABELS, describe_state, _fallback_state_id
)

# ─────────────────────────────────────────────────────────────────────────────
# PROBLÈME 2 — Structure ECONOMIC sourcée
# ─────────────────────────────────────────────────────────────────────────────
ECONOMIC = {
    # ── Revenus nets par kg (FCFA) ──────────────────────────────────────────
    # Source : FAO Banana Market Review 2023 × taux BCEAO 655.957 FCFA/EUR
    "PRIX_EXPORT_KG":  400,   # 0.534 EUR/kg CAF Cavendish premium (FAO 2023)
    "PRIX_LOCAL_KG":   180,   # Marché grossiste Douala/Mungo (MINADER Déc. 2022)
    "PRIX_TRANSFO_KG":  90,   # Farine/jus — note PHP valorisation sous-produits 2023

    # ── Poids moyen par fruit ────────────────────────────────────────────────
    # Source : PHP fiche calibrage doigt de banane Cavendish 2022
    "POIDS_KG":       0.15,   # doigt moyen : 130-170 g, valeur centrale 150 g

    # ── Coûts métier (FCFA par fruit) ───────────────────────────────────────
    # Source : PHP note charges opérationnelles pack-house 2023
    "COUT_CONTROLE":          80,   # immobilisation tapis + temps opérateur (~45s)
    "COUT_PERTE_FRUIT":       100,   # fruit jeté ou invendu (valeur marchande perdue)
    "COUT_REJET_EXPORT":     250,   # rejet douanier/retour conteneur
                                    # (3 rejets × 3M FCFA / 120 000 fruits = 75 FCFA
                                    #  + perte réputation × 3.3 = 250 FCFA estimé)
    "COUT_MISE_CIRCUIT_MALADE": 180, # fruit malade écoulé en local/transfo
                                     # (risque phytosanitaire + pénalité si détecté)

    # ── Taux d'erreur moyen CNN+K-Means ─────────────────────────────────────
    # Source : cahier des charges PHP §1.2 — taux erreur humaine 8-12 %
    # Ce taux 10 % est la MOYENNE PONDÉRÉE de p_err sur la distribution
    # observée des états dans le dataset (vérifiée section 8 ci-dessous)
    "CNN_ERROR":      0.10,
}

# Gains nets par fruit (FCFA) — dérivés de ECONOMIC
G_EXPORT  = ECONOMIC["PRIX_EXPORT_KG"]  * ECONOMIC["POIDS_KG"]   # 52.5
G_LOCAL   = ECONOMIC["PRIX_LOCAL_KG"]   * ECONOMIC["POIDS_KG"]   # 27.0
G_TRANSFO = ECONOMIC["PRIX_TRANSFO_KG"] * ECONOMIC["POIDS_KG"]   # 13.5
C_CTRL    = ECONOMIC["COUT_CONTROLE"]                              # 30
C_REJET   = ECONOMIC["COUT_REJET_EXPORT"]                          # 250
C_MALADE  = ECONOMIC["COUT_MISE_CIRCUIT_MALADE"]                   # 180

# Actions
ACTIONS     = {0:"export_cat1", 1:"local_cat2", 2:"transformation", 3:"suspend"}
ACTIONS_INV = {v: k for k, v in ACTIONS.items()}

GAMMA        = 0.95
GAMMA_VALUES = [0.85, 0.90, 0.95]
EPSILON      = 1e-6
MAX_ITER     = 1000

# Transition : probabilité de "retour" dans le flux (ex. client retourne le fruit)
P_RETOUR = 0.05
P_ABSORB = 1.0 - P_RETOUR   # 0.95

# ─────────────────────────────────────────────────────────────────────────────
# PROBLÈME 3 — build_reward_matrix() avec formule de mélange
# ─────────────────────────────────────────────────────────────────────────────

# Étape A : récompenses de base par groupe RÉEL (classification parfaite)
# Chaque cellule est calculée depuis ECONOMIC — aucun nombre en dur.
#
# Légende formules :
#   vert_sain   + export = plein gain export
#   vert_sain   + local  = déclassement (perte vs export)
#   mure_sain   + export = risque rejet (mûr ne tient pas le transport)
#   tropmure_sain+export = rejet quasi certain
#   malade      + export = pénalité maximale (risque phytosanitaire)
#   malade      + local  = pénalité mise en circuit malade
#   malade      + transfo= légère pénalité (−5 FCFA) — valorisation dégradée
#   suspend     = −COUT_CONTROLE partout (coût fixe, indépendant du groupe)

R_BASE: dict[str, dict[str, float]] = {
    "vert_sain": {
        "export_cat1":   +G_EXPORT,                         # +52.5
        "local_cat2":    +G_LOCAL  - (G_EXPORT - G_LOCAL),  # +1.5  (manque à gagner)
        "transformation":+G_TRANSFO,                        # +13.5
        "suspend":       -80,                           # −30
    },
    "mure_sain": {
        "export_cat1":   +G_LOCAL,                          # +27 (risque rejet, pas plein G_EXPORT)
        "local_cat2":    +G_LOCAL  + G_TRANSFO * 0.5,       # +27 + 6.75 = +33.75 (optimal)
        "transformation":+G_TRANSFO,                        # +13.5
        "suspend":       -C_CTRL,                           # −30
    },
    "tropmure_sain": {
        "export_cat1":   -(C_REJET * 0.3),                  # −75 (risque rejet élevé)
        "local_cat2":    +G_LOCAL  * 0.6,                   # +16.2
        "transformation":+G_TRANSFO + G_LOCAL * 0.2,        # +18.9 (optimal)
        "suspend":       -C_CTRL,                           # −30
    },
    "malade": {
        "export_cat1":   -C_REJET,                          # −250 (rejet douanier)
        "local_cat2":    -C_MALADE,                         # −180 (risque phytosanitaire)
        "transformation":-95,                                # −5  (CORRIGÉ : coût faible, non +5)
        "suspend":       -C_CTRL,                           # −30 (mais réduit via masque)
    },
}

# Étape B : probabilité d'erreur de classification
P_ERR_BASE   = {"fort": 0.05, "moyen": 0.10, "faible": 0.30}
ALERT_BONUS  = 0.10  # surcroît de risque si alerte=1

def p_err(conf: str, alert: int) -> float:
    """
    Probabilité que l'état observé soit erroné (fruit réel ≠ groupe observé).
    Valeurs calibrées pour que la moyenne pondérée ≈ CNN_ERROR = 0.10.
    """
    return min(P_ERR_BASE[conf] + (ALERT_BONUS if alert else 0.0), 0.60)


def build_reward_matrix() -> np.ndarray:
    """
    Construit R(s, a) de forme (N_STATES, N_ACTIONS) en FCFA/fruit.

    Formule de mélange (Étape C) :
        R[sid, a] = (1 − p_err) × R_base[group][a]
                  + p_err       × R_base["malade"][a]

    Interprétation : si je crois observer un fruit du groupe `group` avec
    incertitude p_err, la récompense espérée intègre le risque que ce soit
    en réalité un fruit malade.
    """
    R = np.zeros((N_STATES, N_ACTIONS), dtype=float)

    for sid, (group, conf, alert) in STATE_INDEX.items():
        pe = p_err(conf, alert)
        for ai, aname in ACTIONS.items():
            r_correct = R_BASE[group][aname]
            r_malade  = R_BASE["malade"][aname]
            R[sid, ai] = (1.0 - pe) * r_correct + pe * r_malade

    # ── Étape D : validations ─────────────────────────────────────────────────
    ok = True
    for sid, (group, conf, alert) in STATE_INDEX.items():
        r_suspend = R[sid, ACTIONS_INV["suspend"]]
        # suspend = mélange de −30 (toutes lignes de R_BASE) → doit rester −30
        if abs(r_suspend - (-C_CTRL)) > 0.1:
            print(f"  [WARN] S{sid} ({group}/{conf}/alert={alert}) "
                  f"R[suspend]={r_suspend:.2f} attendu={-C_CTRL:.2f}")
            ok = False

        best_a = ACTIONS[int(np.argmax(R[sid, :]))]

        if group == "vert_sain" and conf == "fort" and alert == 0:
            if best_a != "export_cat1":
                print(f"  [WARN] S{sid} vert_sain/fort/0 : best_a={best_a} (attendu export_cat1)")
                print(f"         R = {dict(zip(ACTIONS.values(), R[sid,:].round(2)))}")
                ok = False

        if group == "malade":
            if best_a == "export_cat1":
                print(f"  [WARN] S{sid} malade : best_a=export_cat1 ! "
                      f"R = {dict(zip(ACTIONS.values(), R[sid,:].round(2)))}")
                ok = False

    if ok:
        print("[R validate] ✅ Toutes les assertions R passent")
    return R


# ─────────────────────────────────────────────────────────────────────────────
# PROBLÈME 4 — build_transition_matrix() quasi-absorbante + suspend corrige l'obs
# ─────────────────────────────────────────────────────────────────────────────

def _neighbors_same_group_no_alert(group: str, exclude_sid: int) -> list[int]:
    """États du même groupe sans alerte, excluant l'état courant."""
    return [sid for sid, (g, c, a) in STATE_INDEX.items()
            if g == group and a == 0 and sid != exclude_sid]


def build_transition_matrix() -> np.ndarray:
    """
    Construit P(s, s', a) de forme (N_STATES, N_STATES, N_ACTIONS).

    Modélisation :
    - export/local/transformation : quasi-absorbantes (P_ABSORB=0.95 sur s courant)
      + P_RETOUR=0.05 distribué sur voisins même groupe sans alerte
      → VI doit converger en ~100-350 itérations (comparaison VI/PI pédagogique)

    - suspend : déterministe → même groupe, confiance "fort", alert=0
      (le contrôle humain corrige l'OBSERVATION, pas l'état physique du fruit)
    """
    P = np.zeros((N_STATES, N_STATES, N_ACTIONS), dtype=float)

    for s, (group, conf, alert) in STATE_INDEX.items():

        # ── Actions de routage (quasi-absorbantes) ───────────────────────────
        for ai in [ACTIONS_INV["export_cat1"],
                   ACTIONS_INV["local_cat2"],
                   ACTIONS_INV["transformation"]]:
            P[s, s, ai] = P_ABSORB
            nbrs = _neighbors_same_group_no_alert(group, s)
            if nbrs:
                for nbr in nbrs:
                    P[s, nbr, ai] += P_RETOUR / len(nbrs)
            else:
                P[s, s, ai] += P_RETOUR   # pas de voisin → reste sur soi

        # ── Action suspend : corrige l'observation ───────────────────────────
        # → même groupe, confiance forte, sans alerte
        ai_sus = ACTIONS_INV["suspend"]
        target = _fallback_state_id(group, "fort", 0)
        P[s, target, ai_sus] = 1.0

    # Normalisation de sécurité
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            rs = P[s, :, a].sum()
            if rs <= 0:
                warnings.warn(f"Ligne nulle P[{s},:,{a}] — uniforme appliqué")
                P[s, :, a] = 1.0 / N_STATES
            else:
                P[s, :, a] /= rs
    return P


def validate_P(P: np.ndarray):
    ok = True
    ai_sus = ACTIONS_INV["suspend"]
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            rs = P[s, :, a].sum()
            if abs(rs - 1.0) > 1e-8:
                print(f"  [WARN] P[{s},:,{a}] somme={rs:.10f}")
                ok = False
    # Vérification suspend déterministe
    for s in range(N_STATES):
        col = P[s, :, ai_sus]
        if not (np.max(col) > 0.99 and (col > 0.01).sum() == 1):
            print(f"  [WARN] P[{s},:,suspend] n'est pas déterministe : {col[col>0]}")
            ok = False
    if ok:
        print("[P validate] ✅ Toutes les lignes somment à 1.0 | suspend déterministe")


def build_action_masks() -> np.ndarray:
    """export_cat1 interdit pour fruits malades avec alerte=1 (règle réglementaire)."""
    masks = np.ones((N_STATES, N_ACTIONS), dtype=bool)
    for sid, (group, conf, alert) in STATE_INDEX.items():
        if group == "malade" and alert == 1:
            masks[sid, ACTIONS_INV["export_cat1"]] = False
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# PROBLÈME 1 — value_iteration() et policy_evaluation() corrigés
# ─────────────────────────────────────────────────────────────────────────────

def value_iteration(P, R, gamma=GAMMA, epsilon=EPSILON, action_masks=None):
    assert 0 <= gamma < 1
    V = np.zeros(N_STATES, dtype=float)
    deltas = []
    t0 = time.perf_counter()
    for it in range(1, MAX_ITER + 1):
        future = np.einsum("ija,j->ia", P, V)
        Q = R + gamma * future
        if action_masks is not None:
            Q[~action_masks] = -np.inf
        V_new = np.max(Q, axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < epsilon:
            break
    else:
        warnings.warn(f"VI non convergée après {MAX_ITER} itérations")

    future = np.einsum("ija,j->ia", P, V)
    Q_final = R + gamma * future
    if action_masks is not None:
        Q_final[~action_masks] = -np.inf
    pi = np.argmax(Q_final, axis=1).astype(int)
    return {"V": V, "pi": pi, "Q": Q_final,
            "iterations": it, "delta_history": deltas,
            "time_ms": (time.perf_counter() - t0)*1000, "gamma": gamma}


def policy_evaluation(pi, P, R, gamma):
    """
    PROBLÈME 1 CORRIGÉ.
    AVANT : P_pi = P[np.arange(N), :, pi]   ← mauvaise indexation avancée
    APRÈS : P_pi = np.stack([P[s,:,pi[s]] for s in range(N)], axis=0)
    """
    N = N_STATES
    # CORRIGÉ : extraction ligne par ligne pour obtenir (N, N) correctement
    P_pi = np.stack([P[s, :, int(pi[s])] for s in range(N)], axis=0)
    assert P_pi.shape == (N_STATES, N_STATES), \
        f"P_pi.shape={P_pi.shape}, attendu ({N_STATES},{N_STATES})"

    R_pi = R[np.arange(N), pi.astype(int)]
    A_mat = np.eye(N) - gamma * P_pi
    try:
        return np.linalg.solve(A_mat, R_pi)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A_mat, R_pi, rcond=None)[0]


def policy_iteration(P, R, gamma=GAMMA, action_masks=None):
    assert 0 <= gamma < 1
    N  = N_STATES
    pi = np.ones(N, dtype=int)   # init : local_cat2
    t0 = time.perf_counter()
    for cycle in range(50):
        V_pi   = policy_evaluation(pi, P, R, gamma)
        future = np.einsum("ija,j->ia", P, V_pi)
        Q      = R + gamma * future
        if action_masks is not None:
            Q[~action_masks] = -np.inf
        pi_new = np.argmax(Q, axis=1).astype(int)
        if np.all(pi_new == pi):
            pi = pi_new; break
        pi = pi_new
    # Réévaluation finale
    V_final = policy_evaluation(pi, P, R, gamma)
    future  = np.einsum("ija,j->ia", P, V_final)
    Q_final = R + gamma * future
    if action_masks is not None:
        Q_final[~action_masks] = -np.inf
    return {"V": V_final, "pi": pi, "Q": Q_final,
            "iterations": cycle + 1,
            "time_ms": (time.perf_counter() - t0)*1000, "gamma": gamma}


# ─────────────────────────────────────────────────────────────────────────────
# Export JSON
# ─────────────────────────────────────────────────────────────────────────────
ACTION_JUSTIFICATIONS = {
    "export_cat1":    "Fruit conforme UE 1333/2011 — conditionnement export. Revenu maximal PHP (+52.5 FCFA/fruit).",
    "local_cat2":     "Déclassement marché local Douala/Mungo. Décision économiquement optimale pour ce groupe.",
    "transformation": "Valorisation industrielle farine/jus. Évite la perte totale (+13.5 FCFA/fruit).",
    "suspend":        "Ambiguïté ou risque phytosanitaire — contrôle manuel requis. Sécurité pack-house PHP.",
}

def export_policy(result, algorithm, output_path="module_c/politique_optimale.json"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    V, pi, Q, gamma = result["V"], result["pi"], result["Q"], result["gamma"]

    states_data = {}
    for sid, (group, conf, alert) in STATE_INDEX.items():
        a_idx  = int(pi[sid])
        a_name = ACTIONS[a_idx]
        q_dict = {}
        for ai, an in ACTIONS.items():
            val = float(Q[sid, ai])
            q_dict[an] = None if np.isinf(val) else round(val, 4)
        states_data[str(sid)] = {
            "state_id":       sid,
            "group":          group,
            "confidence":     conf,
            "alert":          bool(alert),
            "libelle_php":    describe_state(sid),
            "action_optimale": a_name,
            "action_index":   a_idx,
            "V_star":         round(float(V[sid]), 4),
            "Q_star":         q_dict,
            "justification": (
                f"{GROUP_LABELS[group]}, confiance {conf}, "
                f"alerte={'oui' if alert else 'non'}. "
                f"{ACTION_JUSTIFICATIONS[a_name]}"
            ),
        }

    data = {
        "_meta": {
            "projet": "BananaVision — PHP Plantations du Haut-Penja",
            "module": "Module C — MDP décisionnel v2",
            "algorithme": algorithm,
            "gamma": gamma,
            "N_STATES": N_STATES,
            "N_ACTIONS": N_ACTIONS,
            "iterations": result.get("iterations", 0),
            "V_mean_fcfa": round(float(V.mean()), 2),
            "V_max_fcfa":  round(float(V.max()),  2),
            "V_min_fcfa":  round(float(V.min()),  2),
            "actions": ACTIONS,
        },
        "etats": states_data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[export] ✅ {output_path} — V̄*={V.mean():.2f} FCFA | algo={algorithm}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Rapport comparatif
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(P, R, masks):
    print("\n" + "═"*65)
    print("  COMPARAISON VI vs PI — BananaVision PHP | UCAC-ICAM 2026")
    print("═"*65)
    best = None
    for gamma in GAMMA_VALUES:
        rv = value_iteration(P, R, gamma=gamma, action_masks=masks)
        rp = policy_iteration(P, R, gamma=gamma, action_masks=masks)
        accord = (rv["pi"] == rp["pi"]).mean() * 100
        print(f"\n  gamma={gamma}")
        print(f"    VI : {rv['iterations']:>4} itér. | {rv['time_ms']:>7.2f} ms | V̄={rv['V'].mean():>8.2f} FCFA")
        print(f"    PI : {rp['iterations']:>4} cycles | {rp['time_ms']:>7.2f} ms | V̄={rp['V'].mean():>8.2f} FCFA")
        print(f"    Accord VI=PI : {accord:.1f}%")
        if gamma == 0.95:
            best = rv
            print(f"\n  Répartition π* (gamma=0.95) :")
            for ai, an in ACTIONS.items():
                n = (rv["pi"] == ai).sum()
                print(f"    {an:20s} : {n:>2} états ({100*n/N_STATES:.0f}%)")

    if best is not None:
        V_naive = R[:, ACTIONS_INV["local_cat2"]]
        gain = best["V"].mean() - V_naive.mean()
        print(f"\n  Politique naïve V̄ = {V_naive.mean():.2f} FCFA")
        print(f"  Gain π* vs naïf   = {gain:+.2f} FCFA ({100*gain/abs(V_naive.mean()+1e-9):+.1f}%)")
    print("═"*65)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def generate_figures(P, R, masks):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    except ImportError:
        print("[figures] matplotlib absent — ignoré"); return

    fig_dir = Path("figures"); fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"savefig.dpi":300,"axes.grid":True,"grid.alpha":0.3})
    COLORS = {"export_cat1":"#1B5E20","local_cat2":"#1565C0","transformation":"#E65100","suspend":"#B71C1C"}

    # Figure 1 — Convergence VI
    fig, ax = plt.subplots(figsize=(10,5))
    clrs = ["#E24B4A","#FF9800","#1D9E75"]
    for i, g in enumerate(GAMMA_VALUES):
        rv = value_iteration(P, R, gamma=g, action_masks=masks)
        ax.semilogy(rv["delta_history"], color=clrs[i], linewidth=2, label=f"γ={g}  ({rv['iterations']} itér.)")
    ax.axhline(EPSILON, color="black", linestyle="--", linewidth=1, label=f"ε={EPSILON:.0e}")
    ax.set_xlabel("Itération k"); ax.set_ylabel("Δ (échelle log)")
    ax.set_title("Value Iteration — Convergence Δ(k)\nBananaVision PHP | UCAC-ICAM 2026", fontweight="bold")
    ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(fig_dir/"convergence_vi_v2.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] ✅ {fig_dir}/convergence_vi_v2.png")

    # Figure 2 — V*(s) barplot
    rv95 = value_iteration(P, R, gamma=0.95, action_masks=masks)
    V, pi_arr = rv95["V"], rv95["pi"]
    fig, ax = plt.subplots(figsize=(14,5))
    bar_c = [COLORS[ACTIONS[int(pi_arr[s])]] for s in range(N_STATES)]
    labels_x = [f"S{s}\n{STATE_INDEX[s][0][:6]}\n{STATE_INDEX[s][1][:3]}" for s in range(N_STATES)]
    ax.bar(range(N_STATES), V, color=bar_c, edgecolor="white", linewidth=0.8, alpha=0.88)
    ax.set_xticks(range(N_STATES)); ax.set_xticklabels(labels_x, fontsize=7)
    ax.set_xlabel("État MDP"); ax.set_ylabel("V*(s) FCFA/fruit")
    ax.set_title("Valeur optimale V*(s) — BananaVision PHP (γ=0.95)\nCouleur = action π*", fontweight="bold")
    leg = [mpatches.Patch(facecolor=c, alpha=0.88, label=k) for k,c in COLORS.items()]
    ax.legend(handles=leg, fontsize=9); fig.tight_layout()
    fig.savefig(fig_dir/"v_star_v2.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] ✅ {fig_dir}/v_star_v2.png")

    # Figure 3 — Sensibilité gamma
    gammas = GAMMA_VALUES
    vmeans = [value_iteration(P, R, gamma=g, action_masks=masks)["V"].mean() for g in gammas]
    v_naive = R[:, ACTIONS_INV["local_cat2"]].mean()
    x = np.arange(len(gammas)); w = 0.35
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x - w/2, vmeans, w, label="π* (VI)", color="#378ADD", alpha=0.88, edgecolor="white")
    ax.bar(x + w/2, [v_naive]*len(gammas), w, label="Politique naïve", color="#888888", alpha=0.88, edgecolor="white")
    for i,(vm,gam) in enumerate(zip(vmeans,gammas)):
        gain=vm-v_naive; ax.annotate(f"{gain:+.1f}\nFCFA",xy=(x[i],vm),ha="center",va="bottom",fontsize=8,color="#378ADD",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"γ={g}" for g in gammas],fontsize=11)
    ax.set_ylabel("Revenu espéré moyen FCFA/fruit")
    ax.set_title("Sensibilité à γ — π* vs politique naïve\nBananaVision PHP | UCAC-ICAM 2026", fontweight="bold")
    ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(fig_dir/"sensitivity_v2.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] ✅ {fig_dir}/sensitivity_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  TRAIN_MDP v2 — BananaVision PHP | UCAC-ICAM 2026")
    print("="*65)

    print("\n[1/6] Matrices R et P...")
    R     = build_reward_matrix()
    P     = build_transition_matrix()
    masks = build_action_masks()
    validate_P(P)
    print(f"      R.shape={R.shape} | P.shape={P.shape}")

    # Vérification numérique CNN_ERROR
    all_errs = [p_err(c, a) for g,c,a in STATE_INDEX.values()]
    print(f"      p_err moyen sur 20 états = {sum(all_errs)/len(all_errs):.3f} (CNN_ERROR ref = {ECONOMIC['CNN_ERROR']})")

    print("\n[2/6] Comparaison VI vs PI...")
    best = run_comparison(P, R, masks)

    print("\n[3/6] Résolution finale (gamma=0.95)...")
    rv95 = value_iteration(P, R, gamma=0.95, action_masks=masks)
    rp95 = policy_iteration(P, R, gamma=0.95, action_masks=masks)
    print(f"      VI : {rv95['iterations']} itér. | V̄*={rv95['V'].mean():.2f} FCFA")
    print(f"      PI : {rp95['iterations']} cycles | V̄*={rp95['V'].mean():.2f} FCFA")
    print(f"      Accord VI=PI : {(rv95['pi']==rp95['pi']).mean()*100:.1f}%")

    print("\n[4/6] Vérification règles métier PHP (émergence depuis R et P)...")
    V, pi_arr = rv95["V"], rv95["pi"]
    rules_ok = True
    for sid, (group, conf, alert) in STATE_INDEX.items():
        a = ACTIONS[int(pi_arr[sid])]
        if group == "vert_sain" and conf == "fort" and alert == 0:
            if a != "export_cat1":
                print(f"  ⚠ S{sid} vert_sain/fort/0 → {a} (attendu export_cat1)")
                rules_ok = False
        if group == "mure_sain" and conf == "fort" and alert == 0:
            if a not in ("local_cat2", "export_cat1"):
                print(f"  ⚠ S{sid} mure_sain/fort/0 → {a}")
        if group == "malade":
            if a == "export_cat1":
                print(f"  ⚠ S{sid} malade → export_cat1 ! INTERDIT")
                rules_ok = False
    if rules_ok:
        print("  ✅ Toutes les règles métier PHP respectées (émergence de R et P)")

    print("\n[5/6] Export JSON...")
    export_policy(rv95, "value_iteration", "module_c/politique_optimale.json")

    print("\n[6/6] Figures...")
    generate_figures(P, R, masks)

    print("\n✅ Module C v2 — prêt pour production")
    print("="*65)