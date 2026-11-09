"""
Preuve de `porte_sequentielle` — avec mutation. Lancer : .venv/bin/python <ce fichier>.

Chaque check dit ce qu'il refuse. Le dernier casse la porte exprès (signe des gains inversé)
et vérifie que les checks s'en aperçoivent : une preuve qui n'a jamais rien refusé est une
décoration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from porte_sequentielle import (
    PorteConfig,
    departs_utilisables,
    jouer_porte,
    taille_pour_risque,
)


def _echec(msg: str) -> NoReturn:
    print(f"  ✗ {msg}")
    sys.exit(1)


def _gains(edge: float, n: int, seed: int, taille: float = 500.0) -> np.ndarray:
    alea = np.random.default_rng(seed)
    return np.where(alea.random(n) < edge, 1.0, -1.0) * taille


def check_1_un_decideur_parfait_passe_toujours() -> None:
    cfg = PorteConfig(budget=250)
    r = jouer_porte(np.full(2000, 500.0), departs_utilisables(2000, 250, 10, None), cfg)
    if r.n_passes != r.n_chemins or r.n_censures or r.n_casses:
        _echec(f"edge 1,0 : {r}")
    print(f"  ✓ 1. edge 1,0 → {r.n_passes}/{r.n_chemins} passent, cible 1 en 10 pas, cible 2 en 5")


def check_2_un_decideur_toujours_faux_casse_toujours() -> None:
    cfg = PorteConfig(budget=250)
    r = jouer_porte(np.full(2000, -500.0), departs_utilisables(2000, 250, 10, None), cfg)
    if r.n_casses != r.n_chemins:
        _echec(f"edge 0 : {r}")
    print(f"  ✓ 2. edge 0 → {r.n_casses}/{r.n_chemins} cassent")


def check_3_budget_trop_court_censure_tout() -> None:
    # +500 par pas, cible 1 = 5 000 → 10 pas, cible 2 = 2 500 → 5 pas : 15 pas nécessaires.
    cfg = PorteConfig(budget=14)
    r = jouer_porte(np.full(500, 500.0), departs_utilisables(500, 14, 1, None), cfg)
    if r.n_censures != r.n_chemins:
        _echec(f"budget 14 : {r}")
    cfg = PorteConfig(budget=15)
    r = jouer_porte(np.full(500, 500.0), departs_utilisables(500, 15, 1, None), cfg)
    if r.n_passes != r.n_chemins:
        _echec(f"budget 15 : {r}")
    print("  ✓ 3. budget 14 → tout censuré ; budget 15 → tout passe (frontière exacte)")


def check_4_la_phase_2_compte_vraiment() -> None:
    # 10 gains puis 20 pertes : la phase 1 passe (10 x 500 = cible), la phase 2 casse
    # (-10 x 500 = perte max depuis le solde d'entrée en phase 2), budget 40.
    g = np.concatenate([np.full(10, 500.0), np.full(30, -500.0)])
    r = jouer_porte(g, np.array([0]), PorteConfig(budget=40))
    if (r.n_passes, r.n_casses, r.n_censures) != (0, 1, 0):
        _echec(f"phase 2 : {r}")
    # 10 gains, 9 pertes, puis 14 gains : la phase 2 touche -4 500 (pas la perte max), puis
    # remonte à +2 500 net → passe.
    g = np.concatenate([np.full(10, 500.0), np.full(9, -500.0), np.full(14, 500.0)])
    r = jouer_porte(g, np.array([0]), PorteConfig(budget=33))
    if (r.n_passes, r.n_casses, r.n_censures) != (1, 0, 0):
        _echec(f"phase 2 bis : {r}")
    print("  ✓ 4. la phase 2 repart du solde d'entrée : casse à -10 %, passe à +5 %")


def check_5_le_null_est_symetrique_et_les_seeds_reproductibles() -> None:
    cfg = PorteConfig(budget=250)
    dep = departs_utilisables(20_000, 250, 20, None)
    passes, casses = [], []
    for seed in range(20):
        r = jouer_porte(_gains(0.5, 20_000, seed), dep, cfg)
        passes.append(r.n_passes)
        casses.append(r.n_casses)
    p, c = sum(passes), sum(casses)
    # Deux phases : passer demande +10 % PUIS +5 %, casser n'en demande qu'une. Le null passe
    # donc moins souvent qu'il ne casse : on vérifie l'ordre, pas l'égalité.
    if not (0.15 < p / (p + c) < 0.45):
        _echec(f"null : passes {p}, casses {c}")
    r1 = jouer_porte(_gains(0.5, 20_000, 7), dep, cfg)
    r2 = jouer_porte(_gains(0.5, 20_000, 7), dep, cfg)
    if r1 != r2:
        _echec("même seed, résultats différents")
    print(f"  ✓ 5. null : {p} passes contre {c} casses (deux phases), seeds reproductibles")


def check_6_la_taille_risque_bien_le_risque_demande() -> None:
    enjeux = np.array([100.0, 200.0, 300.0, 400.0, 1000.0])
    t = taille_pour_risque(enjeux, 0.01, 50_000.0)
    if abs(t * 300.0 - 500.0) > 1e-9:
        _echec(f"taille {t}")
    print("  ✓ 6. une décision d'enjeu médian risque exactement 1 % du capital")


def check_7_departs_tronques_par_la_fin() -> None:
    d = departs_utilisables(1000, 250, 1, None)
    if d[0] != 0 or d[-1] != 749 or len(d) != 750:
        _echec(f"départs complets : {d[:3]} … {d[-3:]}")
    d = departs_utilisables(1000, 250, 1, 100)
    if d[0] != 650 or len(d) != 100:
        _echec(f"départs tronqués : {d[:3]} … {d[-3:]}")
    if len(departs_utilisables(200, 250, 1, None)) != 0:
        _echec("flux plus court que le budget : départs non vides")
    print("  ✓ 7. départs : budget entier devant, troncature par la fin, flux trop court → vide")


def check_99_mutation_le_signe_inverse_est_refuse() -> None:
    import porte_sequentielle as ps

    original = ps._premier_franchissement

    def mute(cum, haut, bas):  # type: ignore[no-untyped-def]
        return original(-cum, haut, bas)

    ps._premier_franchissement = mute
    try:
        for check in (check_1_un_decideur_parfait_passe_toujours,):
            try:
                check()
            except SystemExit:
                print("  ✓ 99. mutation (cumul inversé) refusée par le check 1")
                return
        _echec("la mutation du signe est passée inaperçue")
    finally:
        ps._premier_franchissement = original


if __name__ == "__main__":
    print("porte_sequentielle — preuve")
    for f in [v for k, v in sorted(globals().items()) if k.startswith("check_")]:
        f()
    print("OK")
