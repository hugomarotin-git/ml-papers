"""
Paper 1 — figures depuis les sorties du driver (points.csv).

Figure 1 : par cellule (cadence x risque), P(funded) p50 de chaque edge planté contre la barre
du null (p95) ; cellules non interprétables grisées.
Figure 2 : courbes de puissance : p50 - barre en fonction de l'edge, une courbe par cellule.

Lancer : .venv/bin/python campagnes/paper1/figures.py campagnes/paper1/sorties/<dossier>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

EN = os.environ.get("PAPER1_EN") == "1"
#: Les figures en anglais vont dans le dossier du paper, les françaises restent avec la campagne.
FIG_EN_DIR = Path(__file__).resolve().parents[0] / "paper" / "figures"


def lib(fr: str, en: str) -> str:
    """Libellé français par défaut, anglais si PAPER1_EN=1 (figures du paper)."""
    return en if EN else fr


def main(dossier: Path) -> None:
    pts = pd.read_csv(dossier / "points.csv")
    cadences = sorted(pts["cadence"].unique())
    risques = sorted(pts["risque"].unique())

    fig, axes = plt.subplots(
        len(cadences),
        len(risques),
        figsize=(3.6 * len(risques), 2.8 * len(cadences)),
        sharex=True,
        sharey=True,
    )
    axes = axes.reshape(len(cadences), len(risques))
    for i, c in enumerate(cadences):
        for j, r in enumerate(risques):
            ax = axes[i, j]
            cell = pd.DataFrame(pts[(pts["cadence"] == c) & (pts["risque"] == r)]).sort_values(
                "edge"
            )
            null = cell[cell["edge"] == 0.5].iloc[0]
            sig = cell[cell["edge"] > 0.5]
            interp = bool(null["censure_null_mediane"] <= 0.10)
            ax.axhline(
                null["barre_null_p95"],
                color="k",
                ls="--",
                lw=1,
                label=lib("null p95", "null bar ($p_{95}$)"),
            )
            ax.plot(
                sig["edge"],
                sig["p50"],
                "o-",
                color="#7A1C24" if interp else "grey",
                label=lib("edge p50", "signal median"),
            )
            ax.fill_between(
                sig["edge"],
                sig["p50"],
                sig["p95"],
                color="#7A1C24" if interp else "grey",
                alpha=0.15,
            )
            if not interp:
                ax.text(
                    0.5,
                    0.5,
                    lib("non interprétable", "uninterpretable"),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="grey",
                )
            ax.set_title(
                lib(
                    f"{c:g} déc./jour · risque {100 * r:g} %  (censure null {null['censure_null_mediane']:.2f})",
                    f"{c:g} dec./day · risk {100 * r:g} %  (null censoring {null['censure_null_mediane']:.2f})",
                ),
                fontsize=8,
            )
            ax.set_ylim(0, 1.02)
            if i == len(cadences) - 1:
                ax.set_xlabel(lib("force du signal planté", "planted signal strength"))
            if j == 0:
                ax.set_ylabel(lib("P(funded), borne basse", "$\\hat{P}$ (lower bound)"))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        lib(
            f"Contrôle positif — {dossier.name}",
            f"Planted-signal control — campaign {dossier.name}",
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig((FIG_EN_DIR if EN else dossier) / "fig1_cellules.png", dpi=160)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for c in cadences:
        for r in risques:
            cell = pd.DataFrame(pts[(pts["cadence"] == c) & (pts["risque"] == r)]).sort_values(
                "edge"
            )
            null = cell[cell["edge"] == 0.5].iloc[0]
            if null["censure_null_mediane"] > 0.10:
                continue
            sig = cell[cell["edge"] > 0.5]
            ax2.plot(
                sig["edge"],
                sig["p50"] - sig["barre_null_p95"],
                "o-",
                label=lib(f"{c:g}/j, {100 * r:g} %", f"{c:g}/day, {100 * r:g} %"),
            )
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel(lib("force du signal planté", "planted signal strength"))
    ax2.set_ylabel(lib("p50 de l'edge - barre du null (p95)", "margin = signal median - null bar"))
    ax2.set_title(
        lib(
            "Marge de détection par cellule (interprétables seulement)",
            "Detection margin per cell (interpretable cells only)",
        ),
        fontsize=9,
    )
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig((FIG_EN_DIR if EN else dossier) / "fig2_puissance.png", dpi=160)
    print(f"figures écrites dans {dossier}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
