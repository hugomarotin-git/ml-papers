"""
Paper 1 — analyse croisée de plusieurs campagnes (même grille, seeds différentes).

Pour chaque cellule (cadence x risque) :
  - la barre du null (p95) dans chaque tirage, sa moyenne et son écart-type = bruit de la barre ;
  - pour chaque force plantée présente dans la campagne de référence : marge = p50 - barre ;
    verdict « nettement » si la marge dépasse l'écart-type de la barre (critère écrit le 09/11
    avant les résultats), « détectable » si marge > 0 et cellule interprétable, sinon « noyé ».

Lancer : .venv/bin/python campagnes/paper1/analyse.py <dossier_reference> <dossier_2> [<dossier_3> ...]
Écrit <dossier_reference>/analyse_croisee.csv et affiche la table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(dossiers: list[Path]) -> None:
    ref = pd.read_csv(dossiers[0] / "points.csv")
    nulls = []
    for d in dossiers:
        p = pd.read_csv(d / "points.csv")
        n = p[p["edge"] == 0.5][
            ["cadence", "risque", "barre_null_p95", "censure_null_mediane"]
        ].copy()
        n["tirage"] = d.name
        nulls.append(n)
    nulls = pd.concat(nulls)
    bruit = (
        nulls.groupby(["cadence", "risque"])["barre_null_p95"]
        .agg(barre_moyenne="mean", barre_ecart_type="std", n_tirages="count")
        .reset_index()
    )

    lignes = []
    points = ref.to_dict("records")
    for b in bruit.to_dict("records"):
        c, r = float(b["cadence"]), float(b["risque"])
        cell = [x for x in points if float(x["cadence"]) == c and float(x["risque"]) == r]
        null = next(x for x in cell if float(x["edge"]) == 0.5)
        ecart = float(b["barre_ecart_type"])
        ecart = 0.0 if np.isnan(ecart) else ecart
        for p in sorted(
            (x for x in cell if float(x["edge"]) > 0.5), key=lambda x: float(x["edge"])
        ):
            marge = float(p["p50"]) - float(p["barre_null_p95"])
            if not bool(p["interpretable"]):
                verdict = "non interprétable"
            elif marge > ecart:
                verdict = "nettement"
            elif marge > 0:
                verdict = "détectable (marge sous le bruit)"
            else:
                verdict = "noyé"
            lignes.append(
                {
                    "cadence": c,
                    "risque": r,
                    "edge": float(p["edge"]),
                    "p50": round(float(p["p50"]), 3),
                    "barre_ref": round(float(p["barre_null_p95"]), 3),
                    "barre_moyenne": round(float(b["barre_moyenne"]), 3),
                    "barre_ecart_type": round(ecart, 3),
                    "n_tirages": int(b["n_tirages"]),
                    "marge": round(marge, 3),
                    "censure_null": round(float(null["censure_null_mediane"]), 3),
                    "censure_edge": round(float(p["censure_mediane"]), 3),
                    "verdict": verdict,
                }
            )
    out = pd.DataFrame(lignes)
    out.to_csv(dossiers[0] / "analyse_croisee.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nbruit de la barre du null par cellule :")
    print(bruit.to_string(index=False))


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
