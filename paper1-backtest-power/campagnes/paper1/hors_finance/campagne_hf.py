"""
Paper 1 — driver de campagne hors finance (porte séquentielle sur les réservations d'hôtel).

Même forme de sortie que `campagnes/paper1/driver.py` : un dossier daté par campagne avec
meta.json (grille, commit, flux), cellules.json et points.csv.

Exemples :
  .venv/bin/python campagnes/paper1/hors_finance/campagne_hf.py --tag HF-A
  .venv/bin/python campagnes/paper1/hors_finance/campagne_hf.py --tag HF-C-n2000 --budgets 250 \
      --risques 0.01 --edges 0.55 0.58 0.62 --n-utilisable 2000 --pas 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

from porte_sequentielle import charger_flux, mesurer_porte, point_en_dict  # noqa: E402

RACINE = ICI.parents[2]
FLUX = RACINE.parent / "data" / "hors_finance" / "hotels.csv"
SORTIES = ICI.parent / "sorties"


def commit_courant() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RACINE, text=True
        ).strip()
    except Exception:
        return "inconnu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[125, 250, 500])
    ap.add_argument("--risques", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument(
        "--edges", type=float, nargs="+", default=[0.52, 0.54, 0.55, 0.56, 0.58, 0.62, 0.70]
    )
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--pas", type=int, default=100)
    ap.add_argument("--n-utilisable", type=int, default=None)
    ap.add_argument("--enjeux-unitaires", action="store_true")
    a = ap.parse_args()

    flux = charger_flux(FLUX)
    seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
    debut = time.time()
    horodatage = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    lignes = []
    print(
        f"flux : {len(flux)} réservations, {flux['date'].min().date()} → {flux['date'].max().date()}"
    )
    print(
        f"{len(a.budgets) * len(a.risques)} cellules x {len(a.edges) + 1} edges x {a.seeds} seeds"
    )
    for b in a.budgets:
        for r in a.risques:
            t0 = time.time()
            pts = mesurer_porte(
                flux, b, r, a.edges, seeds, a.pas, a.n_utilisable, a.enjeux_unitaires
            )
            lignes.extend(point_en_dict(p) for p in pts)
            null = next(p for p in pts if p.edge == 0.5)
            txt = " ".join(
                f"{p.edge:.2f}:{p.p50:.3f}{'●' if p.detectable else '·'}"
                for p in pts
                if p.edge != 0.5
            )
            print(
                f"budget {b:4d} risque {r:.3f} | chemins {null.n_chemins:5d} censure null "
                f"{null.censure_null_mediane:.3f} barre {null.barre_null_p95:.3f} | {txt} "
                f"({time.time() - t0:.1f} s)"
            )
    dossier = SORTIES / f"{horodatage}_{a.tag}"
    dossier.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(lignes).to_csv(dossier / "points.csv", index=False)
    (dossier / "cellules.json").write_text(json.dumps(lignes, indent=1, ensure_ascii=False))
    meta = {
        "tag": a.tag,
        "horodatage_utc": horodatage,
        "commit": commit_courant(),
        "flux": str(FLUX.relative_to(RACINE.parent)),
        "n_reservations": len(flux),
        "n_utilisable": a.n_utilisable,
        "enjeux_unitaires": a.enjeux_unitaires,
        "grille": {
            "budgets": a.budgets,
            "risques": a.risques,
            "edges": a.edges,
            "seeds": a.seeds,
            "seed_offset": a.seed_offset,
            "pas": a.pas,
        },
        "duree_s": round(time.time() - debut, 1),
    }
    (dossier / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(
        f"écrit dans {dossier.relative_to(RACINE)} · {meta['duree_s']} s · commit {meta['commit']}"
    )


if __name__ == "__main__":
    main()
