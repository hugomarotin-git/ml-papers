"""
Paper 1 — analyse des campagnes hors finance (porte séquentielle, réservations d'hôtel).

Lancer : .venv/bin/python campagnes/paper1/hors_finance/analyse_hf.py <sortie> \
             --a <HF-A> --unitaire <HF-A-unitaire> --b <HF-B1> <HF-B2> --c <HF-C-n...> ...

Écrit dans <sortie> :
  analyse_croisee_hf.csv  : par cellule (budget x risque) et force, p50, barre, bruit de la barre
                            (écart-type du p95 du null sur A, B1, B2), verdict ; enjeux réels
                            contre enjeux unitaires côte à côte.
  puissance_n_hf.csv      : par force, marge en fonction des départs utilisables (HF-C).
  fig4_cellules_hf.png    : 9 cellules, p50 du signal (réel en plein, unitaire en pointillé)
                            contre la barre du null ; cellules non interprétables grisées.
  fig5_puissance_n_hf.png : marge en fonction des départs utilisables, une courbe par force,
                            et la barre du null en tirets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
EN = os.environ.get("PAPER1_EN") == "1"
#: Les figures en anglais vont dans le dossier du paper, les françaises restent avec la campagne.
FIG_EN_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"


def lib(fr: str, en: str) -> str:
    """Libellé français par défaut, anglais si PAPER1_EN=1 (figures du paper)."""
    return en if EN else fr


ROUGE = "#7A1C24"
STYLES = {0.55: ":", 0.58: "--", 0.62: "-"}


def lire(d: Path) -> list[dict[str, Any]]:
    """Les points d'une campagne, en dictionnaires simples (un par ligne de points.csv)."""
    return pd.read_csv(d / "points.csv").to_dict("records")


def meta(d: Path) -> dict[str, Any]:
    return json.loads((d / "meta.json").read_text())


def _cellule(points: list[dict[str, Any]], b: int, r: float) -> list[dict[str, Any]]:
    return sorted(
        (p for p in points if int(p["budget"]) == b and abs(float(p["risque"]) - r) < 1e-12),
        key=lambda p: float(p["edge"]),
    )


def _null(cell: list[dict[str, Any]]) -> dict[str, Any]:
    return next(p for p in cell if float(p["edge"]) == 0.5)


def croisee(a: Path, unitaire: Path, bs: list[Path], sortie: Path) -> pd.DataFrame:
    ref, uni = lire(a), lire(unitaire)
    tirages = [lire(d) for d in [a, *bs]]
    lignes: list[dict[str, Any]] = []
    for p in ref:
        e = float(p["edge"])
        if e <= 0.5:
            continue
        b, r = int(p["budget"]), float(p["risque"])
        barres = np.array([float(_null(_cellule(t, b, r))["barre_null_p95"]) for t in tirages])
        u = next(q for q in _cellule(uni, b, r) if float(q["edge"]) == e)
        marge = float(p["p50"]) - float(p["barre_null_p95"])
        ecart = float(barres.std(ddof=1))
        if not bool(p["interpretable"]):
            verdict = "non interprétable"
        elif marge <= 0:
            verdict = "noyé"
        elif marge > ecart:
            verdict = "nettement"
        else:
            verdict = "détectable (marge sous le bruit)"
        lignes.append(
            {
                "budget": b,
                "risque": r,
                "edge": e,
                "p50": float(p["p50"]),
                "barre_ref": float(p["barre_null_p95"]),
                "barre_ecart_type": ecart,
                "n_tirages": len(barres),
                "marge": marge,
                "censure_null": float(p["censure_null_mediane"]),
                "censure_edge": float(p["censure_mediane"]),
                "verdict": verdict,
                "p50_unitaire": float(u["p50"]),
                "barre_unitaire": float(u["barre_null_p95"]),
                "marge_unitaire": float(u["p50"]) - float(u["barre_null_p95"]),
                "censure_null_unitaire": float(u["censure_null_mediane"]),
            }
        )
    t = pd.DataFrame(lignes)
    t.to_csv(sortie / "analyse_croisee_hf.csv", index=False)
    return t


def fig_cellules(a: Path, unitaire: Path, sortie: Path) -> None:
    ref, uni = lire(a), lire(unitaire)
    budgets = sorted({int(p["budget"]) for p in ref})
    risques = sorted({float(p["risque"]) for p in ref})
    fig, axes = plt.subplots(
        len(budgets),
        len(risques),
        figsize=(3.6 * len(risques), 2.8 * len(budgets)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(budgets), len(risques))
    for i, b in enumerate(budgets):
        for j, r in enumerate(risques):
            ax = axes[i, j]
            cell, cu = _cellule(ref, b, r), _cellule(uni, b, r)
            null = _null(cell)
            sig = [p for p in cell if float(p["edge"]) > 0.5]
            sigu = [p for p in cu if float(p["edge"]) > 0.5]
            interp = float(null["censure_null_mediane"]) <= 0.10
            couleur = ROUGE if interp else "grey"
            x = [float(p["edge"]) for p in sig]
            y50 = [float(p["p50"]) for p in sig]
            y95 = [float(p["p95"]) for p in sig]
            ax.axhline(
                float(null["barre_null_p95"]),
                color="k",
                ls="--",
                lw=1,
                label=lib("null p95 (réel)", "null bar ($p_{95}$), real stakes"),
            )
            ax.plot(
                x,
                y50,
                "o-",
                color=couleur,
                label=lib("signal p50, enjeux réels", "signal median, real stakes"),
            )
            ax.fill_between(x, y50, y95, color=couleur, alpha=0.15)
            ax.plot(
                [float(p["edge"]) for p in sigu],
                [float(p["p50"]) for p in sigu],
                "s:",
                color=couleur,
                ms=4,
                label=lib("signal p50, enjeux unitaires", "signal median, unit stakes"),
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
                    f"budget {b} déc. · risque {100 * r:g} %  ",
                    f"budget {b} dec. · risk {100 * r:g} %  ",
                )
                + lib("(censure null ", "(null censoring ")
                + f"{float(null['censure_null_mediane']):.2f})",
                fontsize=8,
            )
            ax.set_ylim(0, 1.02)
            if i == len(budgets) - 1:
                ax.set_xlabel(lib("force du signal planté", "planted signal strength"))
            if j == 0:
                ax.set_ylabel(lib("P(passe), borne basse", "$\\hat{P}$ (lower bound)"))
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(
        lib(
            "Porte séquentielle hors finance (réservations d'hôtel) — contrôle positif",
            "Sequential gate on hotel bookings — planted-signal control",
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig((FIG_EN_DIR if EN else sortie) / "fig4_cellules_hf.png", dpi=160)


def puissance_n(cs: list[Path], sortie: Path) -> pd.DataFrame:
    lignes: list[dict[str, Any]] = []
    for d in cs:
        m = meta(d)
        for p in lire(d):
            if float(p["edge"]) <= 0.5:
                continue
            lignes.append(
                {
                    "budget": int(p["budget"]),
                    "risque": float(p["risque"]),
                    "edge": float(p["edge"]),
                    "n_utilisable": int(m["n_utilisable"]),
                    "pas": int(m["grille"]["pas"]),
                    "n_chemins": int(p["n_chemins"]),
                    "p50": float(p["p50"]),
                    "barre_null_p95": float(p["barre_null_p95"]),
                    "marge": float(p["p50"]) - float(p["barre_null_p95"]),
                    "censure_null": float(p["censure_null_mediane"]),
                    "interpretable": bool(p["interpretable"]),
                    "detectable": bool(p["detectable"]),
                    "dossier": d.name,
                }
            )
    lignes.sort(key=lambda x: (x["edge"], x["n_utilisable"]))
    t = pd.DataFrame(lignes)
    t.to_csv(sortie / "puissance_n_hf.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for e in sorted({x["edge"] for x in lignes}):
        g = [x for x in lignes if x["edge"] == e]
        ax.plot(
            [x["n_utilisable"] for x in g],
            [x["marge"] for x in g],
            STYLES.get(e, "-") + "o",
            color=ROUGE,
            label=lib(f"force {e:.2f}", f"strength {e:.2f}"),
        )
    barre = sorted({(x["n_utilisable"], x["barre_null_p95"]) for x in lignes})
    ax2 = ax.twinx()
    ax2.plot(
        [b[0] for b in barre],
        [b[1] for b in barre],
        "k--",
        lw=1,
        label=lib("barre du null (p95)", "null bar ($p_{95}$)"),
    )
    ax2.set_ylabel(lib("barre du null (p95)", "null bar ($p_{95}$)"))
    ax2.set_ylim(0, 1)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(
        lib(
            "départs utilisables (fin du flux conservée)",
            "usable starts (most recent part of the stream)",
        )
    )
    ax.set_ylabel(lib("marge = p50(signal) - p95(null)", "margin = signal median - null bar"))
    ax.set_title(
        lib(
            "Hors finance — puissance en fonction de n (budget 250 déc., risque 1 %)",
            "Hotel bookings — power vs usable starts (budget 250, risk 1 %)",
        )
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig((FIG_EN_DIR if EN else sortie) / "fig5_puissance_n_hf.png", dpi=150)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--unitaire", type=Path, required=True)
    ap.add_argument("--b", type=Path, nargs="+", required=True)
    ap.add_argument("--c", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    args.sortie.mkdir(parents=True, exist_ok=True)
    t = croisee(args.a, args.unitaire, args.b, args.sortie)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(t.round(3).to_string(index=False))
    fig_cellules(args.a, args.unitaire, args.sortie)
    n = puissance_n(args.c, args.sortie)
    with pd.option_context("display.width", 250):
        print(
            n[
                [
                    "edge",
                    "n_utilisable",
                    "pas",
                    "n_chemins",
                    "p50",
                    "barre_null_p95",
                    "marge",
                    "detectable",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    print(f"écrit dans {args.sortie}")


if __name__ == "__main__":
    main()
