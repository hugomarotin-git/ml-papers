"""
Paper 1 — driver de campagne du contrôle positif (puissance d'une chaîne de validation).

Reconstruit le 2026-11-09 d'après le journal du 2026-09-15 (le driver d'origine vivait dans un
scratchpad et a été perdu). Il ne calcule rien de nouveau : il appelle `mesurer_controle` cellule
par cellule (cadence x risque), en parallèle, et écrit tout ce qu'il faut pour recalculer : la
grille, les seeds, le hash git, la durée, et chaque point (edge, p50, p95, barre du null, censures).

Ne consomme aucun essai : propriété de l'instrument, pas résultat de modèle.

Lancer (depuis la racine du dépôt) :
  .venv/bin/python campagnes/paper1/driver.py --tag fumee --cadences 1 --risques 0.01 \
      --edges 0.55 0.70 --seeds 5 --workers 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
for d in ("juge", "src"):
    p = str(RACINE / d)
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
from aimd import AIMDConfig  # noqa: E402
from controle_positif import mesurer_controle, plus_petit_edge_detectable  # noqa: E402
from regles_ftmo import ReglesFTMO  # noqa: E402

from ftmo_cleaner import DATA_DIR  # noqa: E402
from label_triple_barriere import load_labelle  # noqa: E402
from periodes import assert_dev_seulement, restreindre_au_dev  # noqa: E402

PANEL = DATA_DIR / "v2" / "nas100_h1_pondere.csv"


def config_aimd(risque_base: float) -> AIMDConfig:
    """Les centres AIMD de référence, sans airbag (G12) — copie de `juge/verdict_modele.config_aimd`."""
    return AIMDConfig(
        risque_base=risque_base,
        hausse_additive=0.25,
        seuil_journees_perdantes=3,
        facteur_coupe=0.5,
        m_initial=1.0,
        m_min=0.25,
        m_max=1.25,
        reset_au_passage_de_phase=True,
        airbag_n=None,
    )


def charger_panel(annees: int | None) -> pd.DataFrame:
    panel = restreindre_au_dev(load_labelle(PANEL)).sort_values("DateTime").reset_index(drop=True)
    assert_dev_seulement(panel, "la campagne paper 1")
    if annees is not None:
        # Tronque par journées FTMO entières, en gardant la FIN du dev (le régime le plus récent).
        jours = sorted(pd.Series(panel["jour_ftmo"]).unique())
        garder = list(jours[-round(annees * 250) :])
        panel = panel[panel["jour_ftmo"].isin(garder)].reset_index(drop=True)
    return pd.DataFrame(panel)


def jouer_cellule(args: tuple) -> dict:
    panel, cadence, risque, edges, seeds, seed_offset, seed_cal, pas = args
    t0 = time.time()
    points = mesurer_controle(
        panel,
        cadence,
        config_aimd(risque),
        edges,
        list(range(seed_offset, seed_offset + seeds)),
        seed_cal,
        pas=pas,
        regles=ReglesFTMO(),
    )
    ppe = plus_petit_edge_detectable(points)
    return {
        "cadence": cadence,
        "risque": risque,
        "duree_s": round(time.time() - t0, 1),
        "plus_petit_edge_detectable": ppe,
        "points": [
            {**asdict(p), "interpretable": p.interpretable, "detectable": p.detectable}
            for p in points
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cadences", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--risques", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--edges", type=float, nargs="+", default=[0.52, 0.55, 0.58, 0.62, 0.70])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-calendrier", type=int, default=0)
    ap.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="premier seed de direction (pour retirer le null)",
    )
    ap.add_argument("--pas", type=int, default=3)
    ap.add_argument(
        "--annees", type=float, default=None, help="tronquer le dev aux N dernières années"
    )
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()

    t0 = time.time()
    panel = charger_panel(a.annees)
    journees = int(pd.Series(panel["jour_ftmo"]).nunique())
    labels = int(pd.Series(panel["issue"]).notna().sum())
    print(f"panel : {len(panel)} barres, {journees} journées FTMO, {labels} labels", flush=True)

    cellules = [
        (panel, c, r, a.edges, a.seeds, a.seed_offset, a.seed_calendrier, a.pas)
        for c in a.cadences
        for r in a.risques
    ]
    print(
        f"{len(cellules)} cellules x {len(a.edges) + 1} edges x {a.seeds} seeds, {a.workers} workers",
        flush=True,
    )
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            resultats = list(ex.map(jouer_cellule, cellules))
    else:
        resultats = [jouer_cellule(c) for c in cellules]

    horodatage = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=RACINE, capture_output=True, text=True
    ).stdout.strip()
    sortie = RACINE / "campagnes" / "paper1" / "sorties" / f"{horodatage}_{a.tag}"
    sortie.mkdir(parents=True)
    meta = {
        "tag": a.tag,
        "horodatage_utc": horodatage,
        "commit": commit,
        "panel": str(PANEL.relative_to(RACINE.parent)),
        "n_barres": len(panel),
        "n_journees": journees,
        "n_labels": labels,
        "annees": a.annees,
        "grille": {
            "cadences": a.cadences,
            "risques": a.risques,
            "edges": a.edges,
            "seeds": a.seeds,
            "seed_offset": a.seed_offset,
            "seed_calendrier": a.seed_calendrier,
            "pas": a.pas,
        },
        "aimd": "centres AIMD de référence sans airbag (G12)",
        "duree_totale_s": round(time.time() - t0, 1),
    }
    (sortie / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (sortie / "cellules.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2))
    lignes = [
        {"cadence": c["cadence"], "risque": c["risque"], **p}
        for c in resultats
        for p in c["points"]
    ]
    pd.DataFrame(lignes).to_csv(sortie / "points.csv", index=False)

    print(f"\n{'cadence':>7} {'risque':>6} | {'censure null':>12} {'barre p95':>9} | edges")
    for c in resultats:
        null = next(p for p in c["points"] if p["edge"] == 0.5)
        reste = " ".join(
            f"{p['edge']:.2f}:{p['p50']:.3f}{'✓' if p['detectable'] else ('·' if p['interpretable'] else 'ni')}"
            for p in c["points"]
            if p["edge"] != 0.5
        )
        print(
            f"{c['cadence']:>7} {c['risque']:>6.3f} | {null['censure_mediane']:>12.3f} {null['barre_null_p95']:>9.3f} | {reste}  "
            f"→ ppe {c['plus_petit_edge_detectable']}  ({c['duree_s']} s)"
        )
    print(
        f"\nécrit dans {sortie.relative_to(RACINE)} · {meta['duree_totale_s']} s · commit {commit}"
    )


if __name__ == "__main__":
    main()
