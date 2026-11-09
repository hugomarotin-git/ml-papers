"""
Paper 1 — puissance en fonction de n (campagne C).

Prend des dossiers de sortie du driver (dev tronqué à N années, ou complet) et, pour chaque
cellule (cadence x risque) et chaque force plantée, met côte à côte la médiane du signal, la
barre du null (p95), la marge et les censures, en fonction du nombre de journées et de labels
du panel (lus dans meta.json). Le n qui compte est `journees_utilisables` = journées du panel
moins le budget (250) : seules ces journées peuvent ouvrir un chemin complet (à 1 an, 0 journée
utilisable, un seul chemin par seed, point structurellement vide). Produit fig3_puissance_n.png :
marge en fonction des journées utilisables, une courbe par force, un panneau par cellule ; les
points non interprétables sont évidés.

Lancer : .venv/bin/python campagnes/paper1/analyse_n.py <sortie> <dossier_1> [<dossier_2> ...]
Écrit <sortie>/puissance_n.csv et <sortie>/fig3_puissance_n.png.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

#: = calibration.BUDGET_UN_AN (juge/calibration.py), le budget de journées FTMO d'un chemin.
BUDGET_UN_AN = 250

EN = os.environ.get("PAPER1_EN") == "1"
#: Les figures en anglais vont dans le dossier du paper, les françaises restent avec la campagne.
FIG_EN_DIR = Path(__file__).resolve().parents[0] / "paper" / "figures"


def lib(fr: str, en: str) -> str:
    """Libellé français par défaut, anglais si PAPER1_EN=1 (figures du paper)."""
    return en if EN else fr


ROUGE = "#7A1C24"
STYLES = {0.55: ":", 0.58: "--", 0.62: "-"}


def charger(d: Path) -> pd.DataFrame:
    meta = json.loads((d / "meta.json").read_text())
    p = pd.read_csv(d / "points.csv")
    p = p[p["edge"] != 0.5].copy()
    p["annees"] = meta["annees"] if meta["annees"] is not None else 4.0
    p["n_journees"] = meta["n_journees"]
    p["n_labels"] = meta["n_labels"]
    p["journees_utilisables"] = max(int(meta["n_journees"]) - BUDGET_UN_AN, 0)
    p["dossier"] = d.name
    p["marge"] = p["p50"] - p["barre_null_p95"]
    colonnes = [
        "cadence",
        "risque",
        "edge",
        "annees",
        "n_journees",
        "journees_utilisables",
        "n_labels",
        "p50",
        "barre_null_p95",
        "marge",
        "censure_null_mediane",
        "censure_mediane",
        "interpretable",
        "detectable",
        "dossier",
    ]
    return pd.DataFrame(p[colonnes])


def main(sortie: Path, dossiers: list[Path]) -> None:
    t = pd.concat([charger(d) for d in dossiers]).sort_values(
        ["cadence", "risque", "edge", "annees"]
    )
    t.to_csv(sortie / "puissance_n.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.round(3).to_string(index=False))

    # Tracé : seulement les cellules jouées à plusieurs n, et sans le point à 0 journée
    # utilisable (un seul chemin par seed, borne 0 ou 1 : structurellement vide, gardé dans
    # le CSV pour mémoire).
    t = pd.DataFrame(t[t["journees_utilisables"] > 0])
    par_cellule = t.groupby(["cadence", "risque"])["annees"].nunique().reset_index()
    par_cellule = par_cellule[par_cellule["annees"] >= 2]
    cellules = [
        (float(c), float(r))
        for c, r in zip(par_cellule["cadence"], par_cellule["risque"], strict=True)
    ]
    fig, axes = plt.subplots(1, len(cellules), figsize=(5.8 * len(cellules), 4.4), sharey=True)
    if len(cellules) == 1:
        axes = [axes]
    for ax, (c, r) in zip(axes, cellules, strict=True):
        sous = t[(t["cadence"] == c) & (t["risque"] == r)]
        for e in sorted(set(float(x) for x in sous["edge"])):
            g = pd.DataFrame(sous[sous["edge"] == e]).sort_values("journees_utilisables")
            if g["annees"].nunique() < 2:
                continue
            x = g["journees_utilisables"].to_numpy()
            ax.plot(x, g["marge"].to_numpy(), STYLES.get(e, "-"), color=ROUGE, lw=1.6)
            ok = g["interpretable"].to_numpy(dtype=bool)
            ax.plot(
                x[ok],
                g["marge"].to_numpy()[ok],
                "o",
                color=ROUGE,
                ms=6,
                label=lib(f"force {e:.2f}", f"strength {e:.2f}"),
            )
            ax.plot(x[~ok], g["marge"].to_numpy()[~ok], "o", mfc="white", mec=ROUGE, ms=6)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(
            lib(f"{c:g} déc./jour · risque {r * 100:g} %", f"{c:g} dec./day · risk {r * 100:g} %")
        )
        ax.set_xlabel(
            lib(
                "journées utilisables (panel - budget de 250)",
                "usable trading days (window - 250-day budget)",
            )
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(lib("marge = p50(signal) - p95(null)", "margin = signal median - null bar"))
    h, lab = axes[0].get_legend_handles_labels()
    vus: dict[str, object] = {}
    for hh, ll in zip(h, lab, strict=True):
        vus.setdefault(ll, hh)
    axes[0].legend(vus.values(), vus.keys(), fontsize=8, loc="upper left")
    fig.suptitle(
        lib(
            "Contrôle positif — puissance en fonction de n (points évidés : non interprétable)",
            "Power as a function of usable history (hollow points: uninterpretable)",
        )
    )
    fig.tight_layout()
    fig.savefig((FIG_EN_DIR if EN else sortie) / "fig3_puissance_n.png", dpi=150)
    print(f"écrit {sortie / 'puissance_n.csv'} et fig3_puissance_n.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
