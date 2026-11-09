"""
Paper 1 — tables LaTeX générées depuis les fichiers de sortie datés. Aucun chiffre du paper
n'est tapé à la main : tout passe par ici.

Lancer : .venv/bin/python campagnes/paper1/tables.py
Écrit campagnes/paper1/paper/tables/*.tex et affiche les chiffres clés cités dans le texte.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ICI = Path(__file__).resolve().parent
S = ICI / "sorties"
OUT = ICI / "paper" / "tables"
A = S / "2026-11-09T120632Z_A-puissance"
C = S / "2026-11-09_C-puissance-n"
HF = S / "2026-11-09_HF-analyse"


def f3(x: float) -> str:
    return f"{x:.3f}"


def table_1() -> None:
    """Finance : par cellule, barre du null, son bruit, censure, plus petit signal détectable."""
    rows = pd.read_csv(A / "analyse_croisee.csv").to_dict("records")
    lignes = []
    for c, r in sorted({(float(x["cadence"]), float(x["risque"])) for x in rows}):
        g = sorted(
            (x for x in rows if float(x["cadence"]) == c and float(x["risque"]) == r),
            key=lambda x: float(x["edge"]),
        )
        interp = g[0]["verdict"] != "non interprétable"
        net = [float(x["edge"]) for x in g if x["verdict"] == "nettement"]
        det = [
            float(x["edge"]) for x in g if str(x["verdict"]).startswith(("nettement", "détectable"))
        ]
        p = {float(x["edge"]): f3(float(x["p50"])) for x in g}
        lignes.append(
            f"{c:g} & {100 * r:g} & {f3(float(g[0]['barre_ref']))} & {f3(float(g[0]['barre_ecart_type']))} & "
            f"{f3(float(g[0]['censure_null']))} & {p[0.55]} & {p[0.58]} & {p[0.70]} & "
            + (
                f"{min(det):.2f} & {min(net):.2f}"
                if interp
                else r"\multicolumn{2}{c}{\emph{uninterpretable}}"
            )
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{tabular}{rrrrrrrrrr}",
            r"\toprule",
            r"Cadence & Risk & Null bar & Bar s.d. & Null cens. & \multicolumn{3}{c}{Signal median $\hat P$} & \multicolumn{2}{c}{Smallest detectable} \\",
            r"\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
            r"(dec./day) & (\%) & ($p_{95}$) & (3 draws) & (median) & $s=0.55$ & $s=0.58$ & $s=0.70$ & any & clear \\",
            r"\midrule",
            *lignes,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table1_cells.tex").write_text(tex + "\n")


def table_2() -> None:
    """Finance : puissance en fonction de n (campagne C + A), deux cellules."""
    rows = [
        x
        for x in pd.read_csv(C / "puissance_n.csv").to_dict("records")
        if int(x["journees_utilisables"]) > 0
    ]
    cellules = [(1.0, 0.02), (2.0, 0.01)]
    ns = sorted({int(x["journees_utilisables"]) for x in rows})
    lignes = []
    for c, r in cellules:
        g = [x for x in rows if float(x["cadence"]) == c and float(x["risque"]) == r]
        barre = {int(x["journees_utilisables"]): float(x["barre_null_p95"]) for x in g}
        lignes.append(
            rf"\multicolumn{{2}}{{l}}{{\emph{{{c:g} dec./day, risk {100 * r:g}\,\%}}}} & "
            + " & ".join(f3(barre[n]) for n in ns)
            + r" \\"
        )
        for e in (0.55, 0.58, 0.62):
            marge = {
                int(x["journees_utilisables"]): float(x["marge"])
                for x in g
                if float(x["edge"]) == e
            }
            cells = []
            for n in ns:
                if n in marge:
                    m = marge[n]
                    cells.append((r"\textbf{" + f"{m:+.3f}" + "}") if m > 0 else f"{m:+.3f}")
                else:
                    cells.append("--")
            lignes.append(rf"& $s={e:.2f}$ & " + " & ".join(cells) + r" \\")
    ns_txt = " & ".join(str(int(n)) for n in ns)
    tex = "\n".join(
        [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"\multicolumn{2}{l}{Usable trading days $n - B$} & " + ns_txt + r" \\",
            r"\multicolumn{2}{l}{Development window (years)} & 1.5 & 2 & 3 & 4 (full) \\",
            r"\midrule",
            *lignes,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table2_power_n.tex").write_text(tex + "\n")


def table_3() -> None:
    """Hors finance : par cellule, barre, censure, plus petit signal, réel contre unitaire."""
    rows = pd.read_csv(HF / "analyse_croisee_hf.csv").to_dict("records")
    lignes = []
    for b, r in sorted({(int(x["budget"]), float(x["risque"])) for x in rows}):
        g = sorted(
            (x for x in rows if int(x["budget"]) == b and float(x["risque"]) == r),
            key=lambda x: float(x["edge"]),
        )
        interp = g[0]["verdict"] != "non interprétable"
        net = [float(x["edge"]) for x in g if x["verdict"] == "nettement"]
        p = {float(x["edge"]): (f3(float(x["p50"])), f3(float(x["p50_unitaire"]))) for x in g}
        lignes.append(
            f"{b} & {100 * r:g} & {f3(float(g[0]['barre_ref']))} & {f3(float(g[0]['censure_null']))} & "
            f"{p[0.55][0]} & {p[0.55][1]} & {p[0.58][0]} & {p[0.58][1]} & "
            + (f"{min(net):.2f}" if interp else r"\emph{uninterp.}")
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{tabular}{rrrrrrrrr}",
            r"\toprule",
            r"Budget & Risk & Null bar & Null cens. & \multicolumn{2}{c}{$\hat P$ at $s=0.55$} & \multicolumn{2}{c}{$\hat P$ at $s=0.58$} & Smallest \\",
            r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"(decisions) & (\%) & ($p_{95}$) & (median) & real & unit & real & unit & clear \\",
            r"\midrule",
            *lignes,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table3_hotel_cells.tex").write_text(tex + "\n")


def table_4() -> None:
    """Hors finance : puissance en fonction du nombre de départs utilisables."""
    rows = pd.read_csv(HF / "puissance_n_hf.csv").to_dict("records")
    ns = sorted({int(x["n_utilisable"]) for x in rows})
    barre = {int(x["n_utilisable"]): float(x["barre_null_p95"]) for x in rows}
    chem = {int(x["n_utilisable"]): int(x["n_chemins"]) for x in rows}
    lignes = [
        r"Paths played & "
        + " & ".join(f"{int(chem[n]):,}".replace(",", r"\,") for n in ns)
        + r" \\",
        r"Null bar ($p_{95}$) & " + " & ".join(f3(barre[n]) for n in ns) + r" \\",
        r"\midrule",
    ]
    for e in (0.55, 0.58, 0.62):
        marge = {int(x["n_utilisable"]): float(x["marge"]) for x in rows if float(x["edge"]) == e}
        cells = []
        for n in ns:
            m = marge[n]
            cells.append((r"\textbf{" + f"{m:+.3f}" + "}") if m > 0 else f"{m:+.3f}")
        lignes.append(rf"Margin, $s={e:.2f}$ & " + " & ".join(cells) + r" \\")
    tex = "\n".join(
        [
            r"\begin{tabular}{l" + "r" * len(ns) + "}",
            r"\toprule",
            r"Usable starts & "
            + " & ".join(f"{int(n):,}".replace(",", r"\,") for n in ns)
            + r" \\",
            r"\midrule",
            *lignes,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table4_hotel_power_n.tex").write_text(tex + "\n")


def chiffres_cles() -> None:
    """Les chiffres cités dans le texte, à vérifier contre le PDF (J6)."""
    a = pd.read_csv(A / "points.csv").to_dict("records")

    def p(c: float, r: float, e: float) -> float:
        return float(
            next(
                x["p50"]
                for x in a
                if float(x["cadence"]) == c and float(x["risque"]) == r and float(x["edge"]) == e
            )
        )

    print(
        "finance, 2/day : s=0.70 ->", f3(p(2, 0.01, 0.70)), "(1 %)", f3(p(2, 0.02, 0.70)), "(2 %)"
    )
    print(
        "finance, 2/day : s=0.62 ->", f3(p(2, 0.01, 0.62)), "(1 %)", f3(p(2, 0.02, 0.62)), "(2 %)"
    )
    print(
        "finance, 2/day : s=0.58 ->", f3(p(2, 0.01, 0.58)), "(1 %)", f3(p(2, 0.02, 0.58)), "(2 %)"
    )
    print(
        "finance, 1/day : s=0.58 ->",
        f3(p(1, 0.01, 0.58)),
        "(1 %, uninterp)",
        f3(p(1, 0.02, 0.58)),
        "(2 %)",
    )
    print("finance, 2/day 0.5 % : s=0.62 ->", f3(p(2, 0.005, 0.62)), "(uninterp)")
    for x in a:
        if float(x["edge"]) == 0.5:
            print(
                f"null {x['cadence']:g}/day {100 * float(x['risque']):g} % : barre "
                f"{f3(float(x['barre_null_p95']))} censure {f3(float(x['censure_null_mediane']))}"
            )
    h = pd.read_csv(HF / "analyse_croisee_hf.csv").to_dict("records")
    for x in h:
        if int(x["budget"]) == 250 and float(x["risque"]) in (0.01, 0.02):
            print(
                f"hotel 250 x {100 * float(x['risque']):g} % s={float(x['edge']):.2f} : réel "
                f"{f3(float(x['p50']))} unitaire {f3(float(x['p50_unitaire']))} barre "
                f"{f3(float(x['barre_ref']))} / {f3(float(x['barre_unitaire']))}"
            )
        if float(x["risque"]) == 0.005 and float(x["edge"]) == 0.55:
            print(
                f"hotel {int(x['budget'])} x 0.5 % : censure null réel {f3(float(x['censure_null']))} "
                f"unitaire {f3(float(x['censure_null_unitaire']))} p50(0.55) {f3(float(x['p50']))}"
            )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    table_1()
    table_2()
    table_3()
    table_4()
    chiffres_cles()
    print("tables écrites dans", OUT)
