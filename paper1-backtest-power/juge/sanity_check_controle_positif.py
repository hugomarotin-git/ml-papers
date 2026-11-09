"""
Preuve du contrôle positif — tâche 7.6, décision A10.

Le contrôle positif protège du **faux NO-GO** : sans lui, « aucun edge trouvé » peut vouloir dire
« pas d'edge » OU « chaîne cassée ». Cette preuve vérifie donc que la chaîne d'évaluation **voit**
un edge quand on en plante un, et **ne le voit pas** quand il n'y en a pas (le null) :

- **calibration** : le taux de réussite réalisé colle à l'edge visé (l'injection ne ment pas) ;
- **edge 0,5 == null** : à même seed, la distribution coïncide avec `twin_null.tirer_directions` ;
- **monotonie** : `P(funded)` croît avec l'edge — le cœur d'A10, « si la chaîne ne voit pas un
  edge planté, elle ne verra jamais un edge réel » ;
- **contrôle négatif intégré** : le null n'est JAMAIS détectable (sinon le comparateur est cassé) ;
- **détectabilité** : un edge fort EST détectable, et `plus_petit_edge_detectable` le retrouve ;
- **interprétabilité** : une censure au-dessus du seuil F13, côté edge OU côté null, rend le point
  non interprétable, donc jamais détectable — la barre voyage avec sa fiabilité.

Lancer :  .venv/bin/python juge/sanity_check_controle_positif.py
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from datetime import date, timedelta
from typing import NoReturn

import numpy as np
import pandas as pd
from aimd import AIMDConfig
from cadence import construire_calendrier
from controle_positif import (
    EDGE_NULL,
    injecter_edge,
    mesurer_controle,
    plus_petit_edge_detectable,
)
from regles_ftmo import ReglesFTMO
from trajectoires import SEUIL_CENSURE_ALARMANT
from twin_null import tirer_directions

from atr import ATR_PERIODE
from label_triple_barriere import construire_labels

DEBUT = pd.Timestamp("2021-03-01 18:00")


def _echec(msg: str) -> NoReturn:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """Vérifie qu'un garde-fou lève — **et pour la bonne raison** (piège n°4)."""
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(f"{quoi} : levée pour une AUTRE raison — « {str(e)[:90]} »")
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def panel_directionnel(n_jours: int = 48, barres_par_jour: int = 8) -> pd.DataFrame:
    """
    Panel labellisé aux issues VARIÉES : journées haussières (issue « haut ») et baissières
    (« bas ») en alternance, pour que l'edge — aligné sur l'issue — ait quelque chose à prédire.

    Une journée haussière : après l'entrée, une barre franchit nettement la barrière haute ; une
    journée baissière, la barrière basse. Amplitude large (±25) pour dépasser `K_ATR x ATR` et
    conclure par barrière, pas seulement par timeout. Coupure de 2 h entre journées (rollover) →
    labels intra-journée, comme les vraies données. Chaque Open distinct (C5).

    ⚠️ **48 journées et un budget de 24, pas 24 et 8** (changé le 2026-09-15) : sur le fixture
    court, le null ne concluait que dans 47 % des cas (censure 0,533) — au-dessus du seuil F13,
    donc **non interprétable** depuis que `PointControle` porte ce flag. Un fixture dont le null
    ne conclut pas ne peut prouver ni la détectabilité ni son garde-fou (piège n°2).
    """
    ohlc: list[tuple[float, float, float, float]] = []
    dts = []
    jours = []
    base_jour = date(2021, 3, 1)
    t = DEBUT
    i = 0
    for j in range(n_jours):
        hausse = j % 2 == 0
        for b in range(barres_par_jour):
            o = 100.0 + 0.01 * i
            if i <= ATR_PERIODE:
                haut, bas = o + 5.0, o - 5.0  # amorce ATR
            elif b == barres_par_jour // 2:
                # la barre qui tranche : franchit franchement une barrière
                haut, bas = (o + 25.0, o - 1.0) if hausse else (o + 1.0, o - 25.0)
            else:
                haut, bas = o + 1.0, o - 1.0  # calme
            ohlc.append((o, haut, bas, o))
            dts.append(t)
            jours.append(base_jour + timedelta(days=1 + j))
            t = t + pd.Timedelta(hours=1)
            i += 1
        t = t + pd.Timedelta(hours=2)  # coupure de session
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", dts)
    df["jour_ftmo"] = jours
    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60
    df["gap_kind"] = np.where(ecart == 60, "contigu", "premiere_barre")
    return construire_labels(df)


CONFIG = AIMDConfig(
    facteur_coupe=0.5,
    m_initial=1.0,
    m_min=0.25,
    reset_au_passage_de_phase=True,
    risque_base=0.02,
    hausse_additive=0.25,
    seuil_journees_perdantes=3,
    airbag_n=None,
    m_max=1.25,
)
#: Barème RÉDUIT (cibles 2 %/1 %, 1 journée min) : sur un fixture court, le barème FTMO complet
#: (cibles 10 %+5 %, 4 jours min par phase → 8 minimum) rend tout financement impossible, et tous
#: les P(funded) vaudraient 0 — la monotonie serait alors invisible. On teste la CHAÎNE, pas le
#: barème, et le barème réel est prouvé ailleurs (regles_ftmo).
REGLES = ReglesFTMO(cible_phase1_pct=0.02, cible_phase2_pct=0.01, jours_minimum=1)
SEEDS = (0, 1, 2, 3, 4)
SEED_CAL = 7
CADENCE = 1.0
BUDGET = 24


def check_1_le_fixture_a_des_issues_variees() -> None:
    """
    Anti-vacuité (piège n°2) : le fixture doit produire des issues « haut » ET « bas », sinon
    l'edge n'a pas de direction à prédire et tous les checks suivants seraient creux.
    """
    issues = pd.Series(PANEL["issue"]).value_counts().to_dict()
    if issues.get("haut", 0) == 0 or issues.get("bas", 0) == 0:
        _echec(f"le fixture doit avoir des issues haut ET bas, obtenu {issues}")
    print(f"  1. fixture aux issues variées : {issues}  ✓")


def check_2_calibration_taux_colle_a_edge() -> None:
    """Le taux de réussite réalisé colle à l'edge visé (moyenné sur seeds) — l'injection ne ment pas."""
    cal = construire_calendrier(PANEL, CADENCE, SEED_CAL)
    for edge in (0.5, 0.7, 0.9):
        taux = [
            injecter_edge(PANEL, cal.candidats, edge, s).taux_reussite_realise for s in range(40)
        ]
        moyenne = float(np.nanmean(taux))
        if abs(moyenne - edge) > 0.03:
            _echec(f"edge visé {edge} : taux réalisé moyen {moyenne:.3f} (écart > 0,03)")
    print("  2. calibration : le taux réalisé colle à l'edge visé (40 seeds)  ✓")


def check_3_edge_null_reproduit_le_twin_null() -> None:
    """
    CONTRÔLE NÉGATIF (A10) — `injecter_edge(0,5)` reproduit la distribution du twin-null : même
    nombre de décisions et même comptabilité d'écartés. C'est ce qui garantit que le null du
    contrôle positif EST le vrai null, pas un cousin qui aurait dérivé.
    """
    cal = construire_calendrier(PANEL, CADENCE, SEED_CAL)
    for s in (0, 1, 2):
        edge0 = injecter_edge(PANEL, cal.candidats, EDGE_NULL, s)
        null = tirer_directions(PANEL, cal.candidats, s)
        if len(edge0.decisions) != len(null.decisions):
            _echec("injecter_edge(0,5) et le twin-null ne produisent pas le même nombre de trades")
        if edge0.ecartes != null.ecartes:
            _echec(f"écartés différents : edge {edge0.ecartes} vs null {null.ecartes}")
    print(
        "  3. (contrôle négatif) edge 0,5 reproduit le twin-null (mêmes trades, mêmes écartés)  ✓"
    )


def check_4_monotonie_pfunded_croit_avec_edge() -> None:
    """
    LE CŒUR D'A10 — `P(funded)` croît avec l'edge. Si la chaîne ne voit pas un edge planté monter,
    elle est aveugle, et tout NO-GO futur serait ininterprétable.
    """
    points = {p.edge: p for p in POINTS}
    ordre = sorted(points)
    p50s = [points[e].p50 for e in ordre]
    # ⚠️ ANTI-VACUITÉ AVANT TOUT : deux points saturés à 1,0 se comparent l'un à l'autre sans que
    # rien ne puisse échouer. C'est exactement ce que faisait ce check avant le 2026-09-20.
    satures = sum(1 for x in p50s if x >= 1.0 - 1e-12)
    if satures > 1:
        _echec(
            f"{satures} points saturés à P(funded) = 1,0 dans la grille {ordre} : leur comparaison "
            "mutuelle ne peut PAS échouer, et cette preuve ne prouverait plus la monotonie"
        )
    # ⚠️ Croissance STRICTE, pas « au sens large » : une grille où tout est égal passerait le
    # sens large sans rien démontrer.
    for (ea, a), (eb, b) in itertools.pairwise(list(zip(ordre, p50s, strict=True))):
        if not b > a + 1e-9:
            _echec(
                f"P(funded) ne croît pas STRICTEMENT de l'edge {ea} à {eb} : {a:.4f} → {b:.4f} "
                f"(grille complète {[round(x, 4) for x in p50s]})"
            )
    if not points[max(ordre)].p50 > points[EDGE_NULL].p50 + 1e-9:
        _echec(
            f"l'edge le plus fort ne bat pas le null : {points[max(ordre)].p50} vs {points[EDGE_NULL].p50}"
        )
    print(
        f"  4. (A10) P(funded) croît avec l'edge : {[round(x, 3) for x in p50s]} (edges {ordre})  ✓"
    )


def check_5_le_null_nest_jamais_detectable() -> None:
    """
    CONTRÔLE NÉGATIF INTÉGRÉ — le null (edge 0,5) ne doit JAMAIS ressortir « détectable » : sa
    médiane ne peut pas dépasser son propre p95. Si c'était le cas, le comparateur serait cassé
    (un faux GO), et le contrôle positif lui-même mentirait.
    """
    null = next(p for p in POINTS if p.edge == EDGE_NULL)
    if null.detectable:
        _echec(
            f"le null est déclaré DÉTECTABLE (p50 {null.p50} > barre {null.barre_null_p95}) — "
            "le comparateur est cassé"
        )
    print("  5. (contrôle négatif) le null n'est jamais détectable  ✓")


def check_6_un_edge_fort_est_detectable() -> None:
    """
    Un edge fort DOIT être détectable, et `plus_petit_edge_detectable` doit le retrouver — sinon
    la chaîne est aveugle même à un signal massif, et rien de ce qui suit n'aurait de sens (A10).
    """
    fort = max(p.edge for p in POINTS)
    point_fort = next(p for p in POINTS if p.edge == fort)
    if not point_fort.detectable:
        _echec(
            f"l'edge le plus fort ({fort}) n'est PAS détectable : p50 {point_fort.p50} vs "
            f"barre {point_fort.barre_null_p95} — la chaîne est aveugle à un edge massif"
        )
    seuil = plus_petit_edge_detectable(POINTS)
    if seuil is None:
        _echec(
            "aucun edge détectable alors qu'un edge fort l'est — plus_petit_edge_detectable rate"
        )
    if seuil <= EDGE_NULL:
        _echec(
            f"plus_petit_edge_detectable rend {seuil} <= null : il ne doit jamais rendre le null"
        )
    # ⚠️ **La VALEUR EXACTE, pas « au-dessus du null »** (P3, revue passe 5). Avec la grille
    # précédente `[0.7, 0.9]`, ce seuil ne pouvait valoir que 0,70 — le minimum de la grille —
    # quelle que soit la vraie sensibilité de la chaîne. Il vaut désormais **0,60**, et un point
    # NON détectable existe en dessous : c'est ce qui rend la réponse informative.
    if abs(seuil - 0.60) > 1e-12:
        _echec(
            f"le plus petit edge détectable vaut {seuil} au lieu de 0,60 — soit la sensibilité de "
            "la chaîne a changé, soit la grille ne l'encadre plus"
        )
    sous_le_seuil = [p for p in POINTS if p.edge < seuil and p.edge != EDGE_NULL]
    if not sous_le_seuil:
        _echec(
            "aucun point de la grille sous le seuil (hors null) : le seuil est alors le minimum "
            "de la grille, pas une mesure de sensibilité"
        )
    if any(p.detectable for p in sous_le_seuil):
        _echec("un point sous le seuil est déclaré détectable — le seuil n'est pas le plus petit")
    print(f"  6. (A10) edge fort détectable, plus petit edge détectable = {seuil:.0%}  ✓")


def check_7_interpretable_impose_le_seuil_de_censure() -> None:
    """
    LA BARRE VOYAGE AVEC SA FIABILITÉ (F13) — un point dont la censure dépasse le seuil, côté
    edge OU côté null, n'est ni interprétable ni détectable, et `plus_petit_edge_detectable` ne
    le rend jamais. Vérifié en faisant basculer un point réellement détectable de chaque côté du
    seuil, via `replace` — comme la preuve de `table_null`. Le fixture doit d'abord être lisible
    lui-même (anti-vacuité) : sinon rien ici ne prouverait quoi que ce soit.
    """
    fort = max(POINTS, key=lambda p: p.edge)
    if not fort.interpretable or not fort.detectable:
        _echec(
            f"anti-vacuité : le point fort du fixture n'est pas lisible (censure edge "
            f"{fort.censure_mediane:.3f}, null {fort.censure_null_mediane:.3f}) — le check est creux"
        )
    seuil = SEUIL_CENSURE_ALARMANT
    au_seuil = replace(fort, censure_mediane=seuil, censure_null_mediane=seuil)
    if not au_seuil.interpretable or not au_seuil.detectable:
        _echec("un point PILE au seuil F13 doit rester interprétable (<=, pas <)")
    edge_trop_censure = replace(fort, censure_mediane=seuil + 1e-9)
    if edge_trop_censure.interpretable or edge_trop_censure.detectable:
        _echec("censure de l'EDGE au-dessus du seuil : le point doit être non interprétable")
    null_trop_censure = replace(fort, censure_null_mediane=seuil + 1e-9)
    if null_trop_censure.interpretable or null_trop_censure.detectable:
        _echec(
            "censure du NULL au-dessus du seuil : la barre vaut une convention, pas une mesure — "
            "le point doit être non interprétable même si l'edge a conclu"
        )
    if "NON INTERPRÉTABLE" not in null_trop_censure.resume():
        _echec("resume() doit dire NON INTERPRÉTABLE, pas « noyé » ni « détectable »")
    autres = [p for p in POINTS if p is not fort]
    if plus_petit_edge_detectable([*autres, null_trop_censure]) == fort.edge:
        _echec("plus_petit_edge_detectable rend un point non interprétable")
    print(
        "  7. (F13) interprétable = censure edge ET null <= seuil ; au-delà, jamais détectable, "
        "jamais rendu  ✓"
    )


def check_8_les_gardes() -> None:
    """Chaque entrée absurde lève pour SA raison (piège n°4)."""
    cal = construire_calendrier(PANEL, CADENCE, SEED_CAL)
    _leve(
        lambda: injecter_edge(PANEL, cal.candidats, 1.5, 0),
        ValueError,
        "edge > 1",
        contient="probabilité",
    )
    _leve(
        lambda: injecter_edge(PANEL, cal.candidats, -0.1, 0),
        ValueError,
        "edge < 0",
        contient="probabilité",
    )
    _leve(
        lambda: injecter_edge(PANEL, cal.candidats, 0.6, -1),
        ValueError,
        "seed < 0",
        contient="seed",
    )
    _leve(
        lambda: mesurer_controle(PANEL, CADENCE, CONFIG, [], SEEDS, SEED_CAL),
        ValueError,
        "aucun edge",
        contient="aucun edge",
    )
    # ⚠️ AUDIT DES GARDES (2026-09-21) : cinq gardes de `injecter_edge` qu'aucune preuve ne
    # voyait refuser. Le contrôle positif est ce qui prouve que la chaîne SAIT VOIR un edge —
    # s'il accepte un panel troué, c'est la calibration de notre détecteur qui devient fausse.
    _leve(
        lambda: injecter_edge(PANEL, list(reversed(cal.candidats)), 0.6, 0),
        ValueError,
        "instants non strictement croissants",
        "croissants",
    )
    _leve(
        lambda: injecter_edge(PANEL.drop(columns=["barriere_basse"]), cal.candidats, 0.6, 0),
        ValueError,
        "colonne absente du panel",
        "absente du panel",
    )
    avec_nat = PANEL.copy()
    avec_nat["DateTime"] = pd.to_datetime(avec_nat["DateTime"])
    avec_nat.loc[avec_nat.index[0], "DateTime"] = pd.NaT
    _leve(
        lambda: injecter_edge(avec_nat, cal.candidats, 0.6, 0),
        ValueError,
        "horodatage manquant",
        "horodatage",
    )
    _leve(
        lambda: injecter_edge(PANEL.iloc[::-1].reset_index(drop=True), cal.candidats, 0.6, 0),
        ValueError,
        "panel non trié",
        "trié",
    )
    sans_entree = PANEL.copy()
    sans_entree["label_entree_dt"] = pd.to_datetime(sans_entree["label_entree_dt"])
    # ⚠️ La barre trouée doit être un CANDIDAT du calendrier et porter ses barrières : sinon elle
    # est écartée en amont (« barrières indisponibles ») et la garde n'est jamais atteinte.
    dts_panel = pd.to_datetime(pd.Series(sans_entree["DateTime"]))
    vise = next(
        i
        for i, dt in enumerate(dts_panel)
        if dt in set(pd.to_datetime(pd.Series(list(cal.candidats))))
        and np.isfinite(float(sans_entree["barriere_haute"].iloc[i]))
    )
    sans_entree.loc[sans_entree.index[vise], "label_entree_dt"] = pd.NaT
    _leve(
        lambda: injecter_edge(sans_entree, cal.candidats, 0.6, 0),
        ValueError,
        "candidat sans date d'entrée",
        "C5",
    )
    # anti-vacuité : sur le panel intact, l'injection produit toujours des trades
    if not injecter_edge(PANEL, cal.candidats, 0.6, 0).decisions:
        _echec("anti-vacuité : aucune injection possible sur le panel intact")
    print(
        "  8. gardes : edge hors [0,1], seed négative, edges vide, instants non croissants, "
        "colonne, NaT, tri, entrée non datée — chacune sa raison  ✓"
    )


PANEL = panel_directionnel()
POINTS = mesurer_controle(
    PANEL,
    CADENCE,
    CONFIG,
    # ⚠️⚠️ **Grille descendue SOUS LA SATURATION le 2026-09-20 (revue de la phase 8, passe 5).**
    # Avant : `[0.7, 0.9]` — les deux points donnaient `p50 = 1,0000` pile. La comparaison de
    # monotonie entre eux comparait donc **1,0 à 1,0** : ce `_echec` ne pouvait pas se déclencher,
    # et `plus_petit_edge_detectable` ne pouvait rendre que 70 %, faute d'autre point dans la
    # grille. **Le module dont le rôle unique est de prouver que la chaîne VOIT un edge (A10)
    # passait au vert par saturation.**
    # Mesuré sur ce décor : null 0,6087 · 0,55 → 0,7391 · 0,60 → 0,8696 · 0,65 → 1,0000, barre
    # 0,8522. La grille exerce donc **trois marches réelles** et le plus petit edge détectable
    # vaut **0,60** — une vraie réponse, pas le minimum de la grille.
    edges=[0.55, 0.60, 0.65],
    seeds_directions=SEEDS,
    seed_calendrier=SEED_CAL,
    pas=1,
    regles=REGLES,
    budget_journees=BUDGET,
    unites_minimum=0.0,
)


if __name__ == "__main__":
    print("Preuve du contrôle positif (7.6) :")
    checks = [v for k, v in sorted(globals().items()) if k.startswith("check_")]
    for chk in checks:
        chk()
    print(f"\n{len(checks)}/{len(checks)} — le contrôle positif est prouvé.")
    for p in POINTS:
        print(f"    {p.resume()}")
    sys.exit(0)
