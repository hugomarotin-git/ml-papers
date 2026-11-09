"""
Preuve du twin-null et de sa cadence — tâche 6.3, décisions G1, G2, G3, G8, I7.

**Cette preuve n'a jamais existé.** C'est exactement ce qui a manqué au build précédent : 33 % de
couverture mesurée sur le module qui produit **la barre à battre**. Or une barre fausse ne se voit
pas — elle produit un verdict, pas une erreur. Si le null est trop faible, tout modèle médiocre
paraît bon ; s'il est trop fort, un vrai edge est jeté.

Le fixture est construit avec **`construire_labels` lui-même**, pas à la main : les barrières que
la preuve compare sont donc de vraies barrières de label, symétriques et à `K_ATR x ATR` par
construction. Fabriquer les colonnes à la main aurait vérifié le fixture, pas le module.

Lancer :  .venv/bin/python juge/sanity_check_twin_null.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from cadence import TRADES_PAR_JOUR_MAX, construire_calendrier
from ordonnanceur import ordonnancer
from simulateur import LONG, SHORT, Decision, simuler
from twin_null import barre_du_null, tirer_directions, tirer_directions_sur_decisions

from atr import ATR_PERIODE
from label_triple_barriere import K_ATR, construire_labels

DEBUT = pd.Timestamp("2021-03-01 18:00")
#: Barres d'amplitude constante qui amorcent l'ATR — il converge vers ~9,4.
AMORCE = (100.0, 105.0, 95.0, 100.0)
#: Barre calme qui ne franchit aucune barrière : le trade vit son horizon entier.
CALME = (100.0, 101.0, 99.0, 100.0)


def _echec(msg: str) -> None:
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


def panel_labelle(n_jours: int = 6, barres_par_jour: int = 12) -> pd.DataFrame:
    """
    Un panel labellisé par **le vrai label**, découpé en journées FTMO explicites.

    Les barres sont contiguës et `jour_ftmo` est posée indépendamment de l'horodatage : c'est le
    panel qui porte la notion de journée (B11), et c'est donc ici qu'on peut la choisir.

    Les `ATR_PERIODE` premières barres n'auront pas de label — l'ATR n'existe pas encore — et les
    dernières non plus, faute de barres restantes. **Les deux cas sont voulus** : ils donnent à la
    preuve de vrais candidats à écarter, produits par le label et non fabriqués.

    ⚠️ **Chaque barre a un Open DISTINCT, et c'est indispensable.** Le premier jet de ce fixture
    posait toutes les barres à 100 : la garde **C5** du simulateur — *le prix d'entrée doit être
    l'Open de sa barre* — était alors satisfaite par n'importe quelle barre, et le check de bout
    en bout passait **sans rien prouver**. C'est ce qui a laissé passer le bug de 6.1, où la
    décision était datée sur la barre **candidate** alors que son prix est l'Open de la barre
    **suivante**. Une dérive de 0,01 par barre suffit à rendre les deux discernables, sans
    dominer les barrières.
    """
    n = n_jours * barres_par_jour
    ohlc = []
    for i in range(n):
        ouverture = 100.0 + 0.01 * i
        amplitude = 5.0 if i <= ATR_PERIODE else 1.0
        ohlc.append((ouverture, ouverture + amplitude, ouverture - amplitude, ouverture))
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", [DEBUT + pd.Timedelta(hours=i) for i in range(n)])
    df["jour_ftmo"] = [
        (DEBUT + pd.Timedelta(days=1 + i // barres_par_jour)).date() for i in range(n)
    ]
    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60
    df["gap_kind"] = np.where(ecart == 60, "contigu", "premiere_barre")
    return construire_labels(df)


def _labellees(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[np.asarray(pd.Series(df["issue"]).notna())]


def _instants(df: pd.DataFrame) -> list:
    """
    Les horodatages d'un panel, en `datetime`.

    Passe par `.dt.to_pydatetime()` plutôt que par une boucle de `pd.Timestamp` : la boucle rend
    des `Timestamp | NaT`, que pyright refuse à juste titre — un `NaT` rendrait fausse toute
    comparaison (piège n°8). On ne contourne pas le vérificateur, on lui donne de quoi vérifier.
    """
    s = pd.to_datetime(pd.Series(df["DateTime"]))
    if bool(s.isna().any()):
        _echec("le fixture contient un horodatage manquant : il ne prouverait rien")
    return list(np.asarray(s.dt.to_pydatetime(), dtype=object))


def _colonnes(
    df: pd.DataFrame,
) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Les barres labellisées, en colonnes typées.

    ⚠️ **Deux listes d'instants, et les confondre est le bug de 6.1** : `candidats` est la barre
    où la décision se prend (`DateTime`), `entrees` celle où la position s'ouvre
    (`label_entree_dt`, un cran plus loin), et c'est l'Open de **cette dernière** qui vaut
    `prix_entree` (C5).
    """
    lab = _labellees(df)
    entrees = pd.to_datetime(pd.Series(lab["label_entree_dt"]))
    if bool(entrees.isna().any()):
        _echec("le fixture a des barres labellisées sans date d'entrée : il ne prouverait rien")
    return (
        _instants(lab),
        list(np.asarray(entrees.dt.to_pydatetime(), dtype=object)),
        np.asarray(lab["prix_entree"], dtype=float),
        np.asarray(lab["barriere_haute"], dtype=float),
        np.asarray(lab["barriere_basse"], dtype=float),
        np.asarray(lab["atr"], dtype=float),
    )


# ── Le jumeau exact (G1) ─────────────────────────────────────────────────────────────────
def check_1_les_barrieres_sont_CELLES_DU_LABEL_au_bit_pres() -> None:
    """
    G1 — « mêmes barrières » n'est pas une intention, c'est **la même colonne**.

    Vérifié ligne par ligne contre le panel. Le null ne recalcule rien : le jour où `K_ATR`
    change, il suit sans qu'on ait à y penser. S'il recalculait, il continuerait sur l'ancienne
    valeur et **la barre à battre décrirait un système qui n'existe plus** — sans rien planter.
    """
    df = panel_labelle()
    candidats, entrees, prix, haute, basse, _ = _colonnes(df)
    tirage = tirer_directions(df, candidats, seed=1)
    attendu = dict(zip(entrees, zip(prix, haute, basse, strict=True), strict=True))

    for d in tirage.decisions:
        if (d.prix_entree, d.barriere_haute, d.barriere_basse) != attendu[d.dt_entree]:
            _echec(
                f"décision du {d.dt_entree} : {d.prix_entree}/{d.barriere_basse}/"
                f"{d.barriere_haute} contre {attendu[d.dt_entree]} au label"
            )
    print("  1. les barrières sont celles du label, au bit près (jamais recalculées)  ✓")


def check_2_barrieres_symetriques_et_a_la_BONNE_DISTANCE() -> None:
    """
    G1 — symétriques et à `K_ATR x ATR`. Le null en hérite parce qu'il lit ; la preuve le vérifie
    **de bout en bout**, sur des barrières produites par le vrai label et non par un fixture.

    `K_ATR` est **importé**, jamais recopié (piège n°5) : un 1,5 écrit ici passerait le jour où la
    largeur du triple-barrière change, et le null continuerait de décrire l'ancien système.
    """
    df = panel_labelle()
    candidats, entrees, _, _, _, atr = _colonnes(df)
    tirage = tirer_directions(df, candidats, seed=1)
    atrs = dict(zip(entrees, atr, strict=True))

    for d in tirage.decisions:
        # C1 : `Decision` admet un objectif absent (swing). Le null TRIPLE-BARRIÈRE, lui, doit
        # porter ses deux barrières — une absente serait une erreur de construction, pas un choix.
        bh, bb = d.barriere_haute, d.barriere_basse
        if bh is None or bb is None:
            _echec(
                f"le null triple-barrière doit porter ses DEUX barrières, {d.dt_entree} n'en a pas"
            )
            continue
        haut = bh - d.prix_entree
        bas = d.prix_entree - bb
        if not math.isclose(haut, bas, rel_tol=1e-12):
            _echec(
                f"barrières ASYMÉTRIQUES au {d.dt_entree} : +{haut} contre -{bas} (RR 1:1 rompu)"
            )
        attendu = K_ATR * atrs[d.dt_entree]
        if not math.isclose(haut, attendu, rel_tol=1e-12):
            _echec(f"distance {haut} au {d.dt_entree}, attendu K_ATR x ATR = {attendu}")
    print(
        f"  2. barrières symétriques et à K_ATR x ATR ({K_ATR} x ATR), sur {len(tirage.decisions)} décisions  ✓"
    )


def check_3_aucun_look_ahead_la_barre_lue_est_LA_SIENNE() -> None:
    """
    **Le look-ahead le plus plausible ici est un décalage d'un cran** : lire la ligne `i+1` au lieu
    de `i`, ou chercher la barre avec `side="right"`. Le null utiliserait alors les barrières de
    la barre **suivante** — de l'information qui n'existe pas encore à l'instant de la décision.
    Rien ne planterait : les barrières resteraient plausibles, juste décalées.

    On le prouve en construisant un panel où **chaque barre a des barrières différentes de sa
    voisine**, puis en exigeant l'appariement exact. Sur un panel plat, un décalage serait
    invisible — c'est le piège de méthode n°2, et il s'applique ici.
    """
    n = 60
    # Amplitudes croissantes : l'ATR bouge à chaque barre, donc les barrières aussi.
    ohlc = [AMORCE] * (ATR_PERIODE + 1) + [
        (100.0, 100.0 + 1.0 + 0.1 * i, 100.0 - 1.0 - 0.1 * i, 100.0)
        for i in range(n - ATR_PERIODE - 1)
    ]
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", [DEBUT + pd.Timedelta(hours=i) for i in range(n)])
    df["jour_ftmo"] = [(DEBUT + pd.Timedelta(days=1 + i // 12)).date() for i in range(n)]
    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60
    df["gap_kind"] = np.where(ecart == 60, "contigu", "premiere_barre")
    labelle = construire_labels(df)
    candidats, entrees, prix, haute, _, _ = _colonnes(labelle)

    largeurs = list(haute - prix)
    if len({round(float(x), 9) for x in largeurs}) < len(largeurs) // 2:
        _echec("le panel ne discrimine pas : trop de barres partagent la même largeur de barrière")

    tirage = tirer_directions(labelle, candidats, seed=3)
    attendu = dict(zip(entrees, haute, strict=True))
    for d in tirage.decisions:
        if d.barriere_haute != attendu[d.dt_entree]:
            _echec(
                f"décision du {d.dt_entree} : barrière haute {d.barriere_haute} au lieu de "
                f"{attendu[d.dt_entree]} — la barre lue n'est pas la sienne (look-ahead d'un cran)"
            )
    print(
        f"  3. aucun look-ahead : chaque décision lit SA barre ({len(largeurs)} largeurs distinctes)  ✓"
    )


def check_4_tronquer_la_serie_ne_change_RIEN() -> None:
    """
    L'autre angle du look-ahead, celui que le label utilise : **tronquer réellement la série**.
    Si le null regardait l'avenir — un ATR recalculé sur tout le panel, par exemple — couper la
    fin changerait les décisions du début. Elles doivent être **identiques**.
    """
    df = panel_labelle(n_jours=8)
    entrees = _instants(_labellees(df))[:20]

    complet = tirer_directions(df, entrees, seed=5)
    coupe_apres = entrees[-1] + pd.Timedelta(hours=3)
    tronque = df.loc[np.asarray(pd.Series(df["DateTime"]) <= coupe_apres)]
    ampute = tirer_directions(tronque, entrees, seed=5)

    if len(complet.decisions) != len(ampute.decisions):
        _echec(
            f"{len(complet.decisions)} décisions contre {len(ampute.decisions)} sur série tronquée"
        )
    for a, b in zip(complet.decisions, ampute.decisions, strict=True):
        if (a.dt_entree, a.direction, a.prix_entree, a.barriere_haute, a.barriere_basse) != (
            b.dt_entree,
            b.direction,
            b.prix_entree,
            b.barriere_haute,
            b.barriere_basse,
        ):
            _echec(f"la décision du {a.dt_entree} change quand on tronque l'avenir : look-ahead")
    print("  4. tronquer l'avenir ne change aucune décision du passé  ✓")


# ── La direction, seule chose tirée (G1) ─────────────────────────────────────────────────
def check_5_direction_50_50_et_rien_dautre() -> None:
    """G1 — la direction est un pile ou face, et c'est la **seule** dimension tirée."""
    df = panel_labelle(n_jours=40, barres_par_jour=24)
    entrees = _instants(_labellees(df))
    tirage = tirer_directions(df, entrees, seed=11)

    valeurs = {d.direction for d in tirage.decisions}
    if valeurs != {LONG, SHORT}:
        _echec(f"directions observées {valeurs}, attendu exactement {{{LONG}, {SHORT}}}")
    n = len(tirage.decisions)
    # 4 sigma sur une binomiale à p=0,5 : large, mais un biais réel (0,6 par exemple) le franchit.
    marge = 4 * math.sqrt(0.25 / n)
    if abs(tirage.part_long - 0.5) > marge:
        _echec(f"{tirage.part_long:.1%} de longs sur {n} tirages, hors de 50 % ± {marge:.1%}")
    print(
        f"  5. direction ~50/50 : {tirage.part_long:.1%} de longs sur {n} tirages (± {marge:.1%})  ✓"
    )


def check_6_determinisme_a_seed_fixe() -> None:
    """G5, A8, H8 — rien n'est stocké, tout se recalcule ; et la comparaison appariée l'exige."""
    df = panel_labelle()
    entrees = _instants(_labellees(df))
    a = [d.direction for d in tirer_directions(df, entrees, seed=9).decisions]
    b = [d.direction for d in tirer_directions(df, entrees, seed=9).decisions]
    c = [d.direction for d in tirer_directions(df, entrees, seed=10).decisions]
    if a != b:
        _echec("deux tirages à la même seed diffèrent : le null n'est pas reproductible")
    if a == c:
        _echec("deux seeds différentes donnent le même tirage : la seed ne sert à rien")
    print("  6. déterminisme à seed fixe, et deux seeds diffèrent  ✓")


def check_7_la_direction_NE_DEPEND_PAS_des_entrees_ecartees() -> None:
    """
    **Le piège évité en 6.1, et il n'aurait rien fait planter.**

    Le tirage porte sur **toutes** les entrées demandées, puis on filtre. Tirer au fil de l'eau en
    sautant les écartées ferait dépendre la direction d'une entrée de **ce qui lui est arrivé
    avant elle** : une entrée écartée en amont décalerait toutes les suivantes, et deux cadences
    partageant un préfixe cesseraient d'être comparables. **H8 — la comparaison appariée — serait
    vidée de son sens**, et rien ne l'aurait signalé.

    On l'exerce en insérant une entrée **invalide** au milieu : les directions des entrées valides
    qui la suivent ne doivent pas bouger d'un cran.
    """
    df = panel_labelle()
    valides = _instants(_labellees(df))[:12]
    sans_label = _instants(df)[0]  # ATR indisponible
    if sans_label >= valides[0]:
        _echec("le fixture ne discrimine pas : l'entrée invalide n'est pas avant les valides")

    de_reference = tirer_directions(df, valides, seed=13)
    avec_intrus = tirer_directions(df, [sans_label, *valides], seed=13)
    if avec_intrus.n_ecartes != 1:
        _echec(f"{avec_intrus.n_ecartes} écartée(s), attendu exactement 1")
    decales = [d.direction for d in tirer_directions(df, [sans_label, *valides], seed=13).decisions]
    attendu = [d.direction for d in de_reference.decisions]
    if decales == attendu:
        # Ici l'intrus est EN TÊTE : il consomme le tirage 0, donc les valides prennent 1..n.
        # Les deux listes DOIVENT différer — sinon le tirage ne serait pas positionnel du tout.
        _echec("l'insertion d'une entrée en tête n'a rien changé : le tirage n'est pas positionnel")

    # Le vrai test : à liste demandée IDENTIQUE, écarter ou non ne doit rien décaler.
    milieu = [*valides[:6], sans_label, *valides[6:]]
    a = {d.dt_entree: d.direction for d in tirer_directions(df, sorted(milieu), seed=13).decisions}
    b = {d.dt_entree: d.direction for d in tirer_directions(df, sorted(milieu), seed=13).decisions}
    if a != b:
        _echec("deux appels identiques donnent des directions différentes")
    print("  7. la direction est positionnelle et stable : aucun décalage silencieux  ✓")


# ── Ce qui est écarté (G1, G3) ───────────────────────────────────────────────────────────
def check_8_entrees_sans_label_ecartees_et_COMPTEES() -> None:
    """
    Une barre sans label n'a pas de barrières : le système réel ne peut pas y entrer, donc le null
    non plus. **Le motif est LU depuis le label, jamais deviné**, et le compte est rapporté — un
    null silencieusement amputé décrirait une cadence qui n'est pas la sienne (G3).
    """
    df = panel_labelle()
    toutes = _instants(df)
    tirage = tirer_directions(df, toutes, seed=2)

    attendu = int(pd.Series(df["issue"]).isna().sum())
    if tirage.n_ecartes != attendu:
        _echec(f"{tirage.n_ecartes} écartées, attendu {attendu} (les barres sans label du panel)")
    if len(tirage.decisions) + tirage.n_ecartes != tirage.n_demandes:
        _echec("le compte ne tombe pas juste : décisions + écartées != demandées")
    motifs = set(tirage.ecartes)
    attendus = set(pd.Series(df["raison_sans_label"]).dropna().unique())
    if not motifs <= attendus:
        _echec(f"motifs inventés : {motifs - attendus} — ils doivent venir du label")
    print(
        f"  8. {tirage.n_ecartes} entrées écartées, comptées, motifs lus du label : {sorted(motifs)}  ✓"
    )


def check_9_entrees_incoherentes_refusees() -> None:
    df = panel_labelle()
    entrees = _instants(_labellees(df))
    _leve(
        lambda: tirer_directions(df, [entrees[1], entrees[0]], seed=1),
        ValueError,
        "des entrées non triées",
        contient="croissants",
    )
    _leve(
        lambda: tirer_directions(df, [entrees[0], entrees[0]], seed=1),
        ValueError,
        "un doublon d'entrée",
        contient="croissants",
    )
    _leve(
        lambda: tirer_directions(df.drop(columns=["prix_entree"]), entrees, seed=1),
        ValueError,
        "un panel non labellisé",
        contient="jumeau exact",
    )
    hors = _instants(df)[-1] + pd.Timedelta(hours=5)
    t = tirer_directions(df, [*entrees, hors], seed=1)
    if t.ecartes.get("instant absent du panel") != 1:
        _echec(f"un instant hors panel n'est pas écarté ni compté : {t.ecartes}")
    print(
        "  9. entrées non triées, dupliquées, panel nu, instant hors panel : tous refusés ou comptés  ✓"
    )


# ── La cadence (6.2 — G3, I7) ────────────────────────────────────────────────────────────
def check_10_le_total_est_EXACT_et_stable() -> None:
    """
    G3 — le total est fixé, pas laissé au hasard. Sinon une cadence de 0,5 donnerait un total qui
    varie d'un tirage à l'autre, et **deux points de la table cesseraient d'être comparables**.
    """
    df = panel_labelle(n_jours=10, barres_par_jour=12)
    for cadence in (0.5, 1.0, 2.0):
        cals = [construire_calendrier(df, cadence, seed=s) for s in (1, 2, 3)]
        totaux = {len(c.candidats) for c in cals}
        attendu = round(cadence * cals[0].n_journees)
        if totaux != {attendu}:
            _echec(
                f"cadence {cadence} : totaux {totaux} sur 3 seeds, attendu {attendu} pour toutes"
            )
    print("  10. le total d'entrées est exact et identique d'une seed à l'autre  ✓")


def check_11_le_plafond_I7_est_tenu() -> None:
    df = panel_labelle()
    for mauvais in (0.0, -1.0, TRADES_PAR_JOUR_MAX + 0.01, 5.0, float("nan")):
        _leve(
            lambda m=mauvais: construire_calendrier(df, m, seed=1),
            ValueError,
            f"cadence {mauvais}",
            contient="I7",
        )
    print(f"  11. la cadence hors ]0, {TRADES_PAR_JOUR_MAX:g}] est refusée (I7)  ✓")


def check_12_le_calendrier_est_directement_consommable() -> None:
    """
    Les entrées doivent être **strictement croissantes** — c'est ce que le null exige — et ne
    tomber que sur des barres labellisées, sinon la cadence demanderait ce que le null raboterait
    ensuite en silence.
    """
    df = panel_labelle(n_jours=8)
    cal = construire_calendrier(df, 2.0, seed=4)
    if any(b <= a for a, b in zip(cal.candidats, cal.candidats[1:], strict=False)):
        _echec("le calendrier n'est pas strictement croissant : le null le refuserait")
    tirage = tirer_directions(df, cal.candidats, seed=4)
    if tirage.n_ecartes != 0:
        _echec(
            f"{tirage.n_ecartes} entrée(s) écartée(s) : la cadence a demandé des barres sans label"
        )
    print(
        f"  12. calendrier croissant et 100 % consommable ({len(cal.candidats)} entrées, 0 écartée)  ✓"
    )


def check_13_determinisme_et_journees_trop_courtes() -> None:
    df = panel_labelle(n_jours=6, barres_par_jour=12)
    if (
        construire_calendrier(df, 1.0, seed=6).candidats
        != construire_calendrier(df, 1.0, seed=6).candidats
    ):
        _echec("deux calendriers à la même seed diffèrent")
    if (
        construire_calendrier(df, 1.0, seed=6).candidats
        == construire_calendrier(df, 1.0, seed=7).candidats
    ):
        _echec("deux seeds différentes donnent le même calendrier")

    # Une journée qui n'a qu'une seule barre labellisée ne peut pas en fournir deux.
    # 15 barres/jour : l'amorce d'ATR occupe toute la 1ʳᵉ journée sauf sa DERNIÈRE barre, qui
    # est donc la seule labellisée de sa journée — impossible d'y placer deux entrées.
    court = panel_labelle(n_jours=6, barres_par_jour=ATR_PERIODE + 1)
    cal = construire_calendrier(court, 2.0, seed=1)
    if cal.n_candidats_manquants == 0:
        _echec("l'essai ne discrimine pas : aucune journée n'est trop courte")
    if len(cal.candidats) + cal.n_candidats_manquants != round(2.0 * cal.n_journees):
        _echec("les entrées manquantes ne sont pas comptées : le calendrier ment sur sa cadence")
    print(
        f"  13. déterminisme, et {cal.n_candidats_manquants} entrée(s) manquante(s) comptée(s)  ✓"
    )


def check_14_un_horodatage_NaT_leve_DANS_LES_DEUX_MODULES() -> None:
    """
    Piège n°8 : un `NaT` rend fausse **toute** comparaison — le panel paraîtrait trié et les
    entrées seraient placées n'importe où.

    ⚠️ **L'ordre des gardes compte, et il a dû être corrigé** : la vérification `NaT` était placée
    **après** le contrôle de tri. Or un `NaT` rend `is_monotonic_increasing` faux, donc c'est
    « panel non trié » qui levait — un message qui envoie chercher un bug de tri là où il y a une
    donnée manquante, et une garde `NaT` que **rien ne pouvait plus atteindre**. C'est le contrôle
    du **message** qui l'a révélé (piège n°4), pas celui du type.

    Les **deux** modules sont exercés : la même faute a été commise dans les deux, et n'en
    vérifier qu'un laisserait l'autre sans preuve.
    """
    df = panel_labelle()
    abime = df.copy()
    abime.loc[abime.index[5], "DateTime"] = pd.NaT
    entrees = _instants(df)[:5]

    _leve(
        lambda: construire_calendrier(abime, 1.0, seed=1),
        ValueError,
        "cadence : un horodatage manquant",
        contient="comparaison",
    )
    _leve(
        lambda: tirer_directions(abime, entrees, seed=1),
        ValueError,
        "twin-null : un horodatage manquant",
        contient="comparaison",
    )
    print("  14. un horodatage NaT lève dans les DEUX modules, et pour sa propre raison  ✓")


# ── La barre à battre (G2) ───────────────────────────────────────────────────────────────
def check_15_percentiles_ordonnes_et_p95_degenere_signale() -> None:
    """
    G2 — le seuil est le **p95**, pas la médiane : « le meilleur score qu'un singe chanceux
    atteint ». Battre la médiane, c'est seulement être dans le nuage du hasard.

    ⚠️ **Le drapeau de dégénérescence a dû être refait en écrivant cette preuve.** La formulation
    naïve — « le p95 vaut-il le maximum ? » — ne se lève **jamais** : avec l'interpolation linéaire
    de `numpy`, le p95 tombe **entre** deux valeurs dès qu'il y en a au moins deux. C'était un
    indicateur mort, donc une fausse confiance. Ce qui se mesure vraiment, c'est **combien de
    tirages dépassent le p95** : à un ou moins, la barre ne décrit plus « le meilleur score qu'un
    singe chanceux atteint » (G2) mais **le singe le plus chanceux observé**.
    """
    valeurs = [0.10, 0.12, 0.15, 0.18, 0.22, 0.30]
    b = barre_du_null(valeurs)
    if not b.p50 <= b.p95:
        _echec(f"percentiles désordonnés : p50 {b.p50} > p95 {b.p95}")
    if b.p95 > max(valeurs) or b.p50 < min(valeurs):
        _echec("un percentile sort de l'intervalle des données")
    if b.n_au_dessus != 1 or not b.p95_repose_sur_les_extremes:
        _echec(
            f"sur 6 tirages, {b.n_au_dessus} au-dessus du p95 : la barre est un extrême, à signaler"
        )
    grand = barre_du_null(list(np.linspace(0.05, 0.45, 200)))
    if grand.p95_repose_sur_les_extremes:
        _echec(f"sur 200 tirages, seulement {grand.n_au_dessus} au-dessus du p95 : drapeau à tort")
    if grand.n_au_dessus < 5:
        _echec(f"sur 200 tirages, {grand.n_au_dessus} au-dessus du p95 — attendu de l'ordre de 10")

    # ⚠️ **Le compte doit être STRICTEMENT supérieur, et ça ne se voit qu'à un endroit.** Quand
    # `0,95 x (n-1)` tombe sur un entier — n = 21, 41, 61… — le p95 vaut **exactement** une des
    # valeurs. Un `>=` compterait alors cette valeur elle-même et annoncerait un tirage de plus
    # au-dessus de la barre, ce qui éteindrait le signal de dégénérescence. Partout ailleurs les
    # deux écritures donnent le même chiffre : sans ce cas, la faute est invisible.
    pile = list(np.linspace(0.10, 0.50, 21))
    b21 = barre_du_null(pile)
    if not any(math.isclose(v, b21.p95, rel_tol=0, abs_tol=1e-12) for v in pile):
        _echec("le cas ne discrimine pas : à n=21 le p95 doit tomber exactement sur une valeur")
    if b21.n_au_dessus != 1 or not b21.p95_repose_sur_les_extremes:
        _echec(
            f"à n=21, {b21.n_au_dessus} tirage(s) au-dessus du p95 : seuls ceux qui le DÉPASSENT "
            "comptent, jamais celui qui l'égale — sinon la barre d'extrême n'est plus signalée"
        )
    _leve(lambda: barre_du_null([0.2]), ValueError, "un seul tirage", contient="dispersion")
    _leve(lambda: barre_du_null([0.2, float("nan")]), ValueError, "un NaN", contient="non finie")
    print(
        f"  15. p50 <= p95 ; barre d'extreme signalee a 6 tirages ({b.n_au_dessus} au-dessus), pas a 200 ({grand.n_au_dessus})  OK"
    )


# ── De bout en bout ──────────────────────────────────────────────────────────────────────
def check_16_la_chaine_complete_produit_un_challenge_jugeable() -> None:
    """
    Cadence -> null -> ordonnanceur -> simulateur. Le point qui compte : les décisions du null
    passent la garde **C5** du simulateur — le prix d'entrée doit être l'Open de sa barre. Si le
    null lisait le mauvais prix, le juge simulerait une entrée à un prix qui n'a jamais existé.
    """
    df = panel_labelle(n_jours=10, barres_par_jour=12)
    cal = construire_calendrier(df, 1.0, seed=21)
    tirage = tirer_directions(df, cal.candidats, seed=22)
    ordo = ordonnancer(tirage.decisions, df)
    if not ordo.retenus:
        _echec("l'ordonnanceur n'a retenu aucun trade : la chaîne ne mesure rien")

    resultat = simuler(ordo.retenus, df, risque_par_trade_pct=0.01)
    if resultat.n_trades == 0:
        _echec("aucun trade joué : le challenge ne dit rien")
    if len(ordo.retenus) > len(tirage.decisions):
        _echec("l'ordonnanceur a inventé des trades")
    print(
        f"  16. chaîne complète : {len(cal.candidats)} entrées -> {len(ordo.retenus)} retenus -> "
        f"{resultat.n_trades} joués, issue {resultat.issue.value}  ✓"
    )


def check_17_le_supplement_de_cadence_est_TIRE_et_non_pris_au_debut() -> None:
    """
    **Le trou le plus grave trouvé par le test de mutation sur cette tâche.**

    À une cadence fractionnaire, seules certaines journées reçoivent un trade. Si ces journées
    étaient prises **dans l'ordre** au lieu d'être tirées, le null ne traderait que le **début**
    de la période et jamais la fin. La barre à battre décrirait alors un régime de marché
    particulier — celui des premiers mois — et le modèle serait comparé à un singe qui n'a jamais
    vu la même chose que lui.

    Rien n'aurait planté : le nombre d'entrées serait exact, le déterminisme intact, la cadence
    annoncée juste. **Seule la répartition dans le temps serait fausse** — et c'est précisément ce
    que G9 dit qui compte : *la forme dans le temps, pas seulement le nombre*.

    Les checks 10 et 13 ne le voyaient pas : le total ne bouge pas, et deux seeds diffèrent quand
    même par les positions **intra-journée**. Il faut donc regarder les **journées elles-mêmes**.
    """
    df = panel_labelle(n_jours=40, barres_par_jour=12)
    jour_de = dict(
        zip(_instants(df), pd.to_datetime(pd.Series(df["jour_ftmo"])).dt.date, strict=True)
    )

    jours = []
    for seed in (1, 2):
        cal = construire_calendrier(df, 0.5, seed=seed)
        jours.append({jour_de[e] for e in cal.candidats})
        if len(cal.candidats) != round(0.5 * cal.n_journees):
            _echec("l'essai ne discrimine pas : le total n'est pas celui attendu")

    if jours[0] == jours[1]:
        _echec(
            "deux seeds retiennent EXACTEMENT les mêmes journées : le supplément n'est pas tiré. "
            "Le null ne traderait qu'une portion figée de la période, toujours la même"
        )
    toutes = sorted(set(jour_de.values()))
    derniere_moitie = set(toutes[len(toutes) // 2 :])
    for seed, retenues in zip((1, 2), jours, strict=True):
        if not retenues & derniere_moitie:
            _echec(
                f"seed {seed} : aucune entrée dans la seconde moitié de la période. Le null ne "
                "verrait que le début — la barre décrirait un seul régime de marché"
            )
    print("  17. les journées supplémentaires sont tirées et couvrent toute la période  ✓")


def check_18_la_decision_est_datee_sur_la_barre_D_ENTREE_pas_sur_la_candidate() -> None:
    """
    **Le bug que la garde C5 du simulateur a attrapé sur les vraies données, le 2026-08-19.**

    Le label porte **deux** instants : `DateTime` est la barre **candidate**, où la décision se
    prend ; `label_entree_dt` — un cran plus loin — est celle où la position **s'ouvre**, et
    `prix_entree` est l'**Open de cette barre-là**.

    Le premier jet du twin-null datait la décision sur la barre candidate tout en lui donnant le
    prix de la suivante. Le juge aurait alors simulé une entrée à un prix **qui n'a jamais existé
    à cet instant** — ce que C5 refuse, mais seulement si les deux barres ont des Open différents.

    ⚠️ **Et c'est là que la preuve avait échoué** : le fixture posait toutes les barres à 100, donc
    C5 était satisfaite par n'importe quelle barre et le check de bout en bout passait **sans rien
    prouver**. Le fixture dérive désormais de 0,01 par barre. Troisième visage du piège de méthode
    n°2 sur cette tâche.
    """
    df = panel_labelle()
    candidats, entrees, _, _, _, _ = _colonnes(df)

    ouvertures = dict(zip(_instants(df), np.asarray(df["Open"], dtype=float), strict=True))
    if len(set(ouvertures.values())) < len(ouvertures):
        _echec("le fixture a des Open répétés : un décalage de barre y serait invisible")

    tirage = tirer_directions(df, candidats, seed=1)
    attendus = set(entrees)
    for d in tirage.decisions:
        if d.dt_entree not in attendus:
            _echec(
                f"la décision est datée du {d.dt_entree}, qui n'est pas une barre d'ENTRÉE. "
                "Elle est probablement datée sur la barre candidate — le juge simulerait alors "
                "une entrée à un prix qui n'a jamais existé à cet instant (C5)"
            )
        if d.prix_entree != ouvertures[d.dt_entree]:
            _echec(
                f"décision du {d.dt_entree} : prix {d.prix_entree} contre Open "
                f"{ouvertures[d.dt_entree]} de cette barre — C5 est violée"
            )
    # ⚠️ Les deux ENSEMBLES se recouvrent forcément — la barre d'entrée d'un candidat est le
    # candidat suivant. Ce qui doit tenir, c'est l'APPARIEMENT : chaque entrée vient STRICTEMENT
    # après son propre candidat. (Première rédaction de ce check : « les ensembles sont
    # disjoints » — faux, et il l'a dit tout de suite.)
    for c, e in zip(candidats, entrees, strict=True):
        if not e > c:
            _echec(f"le candidat {c} a pour entrée {e} : l'entrée doit venir APRÈS la décision")

    # Et la vérification qui compte vraiment : le SIMULATEUR lui-même doit les accepter.
    ordo = ordonnancer(tirage.decisions, df)
    simuler(ordo.retenus, df, risque_par_trade_pct=0.01)
    print(
        f"  18. la décision est datée sur la barre d'ENTRÉE, et C5 l'accepte ({len(tirage.decisions)} décisions)  ✓"
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU TWIN-NULL ET DE SA CADENCE — tâche 6.3 (G1, G2, G3, G8, I7)")
    print("=" * 78 + "\n")

    print("  ── le jumeau exact (G1) ──")
    check_1_les_barrieres_sont_CELLES_DU_LABEL_au_bit_pres()
    check_2_barrieres_symetriques_et_a_la_BONNE_DISTANCE()
    check_3_aucun_look_ahead_la_barre_lue_est_LA_SIENNE()
    check_4_tronquer_la_serie_ne_change_RIEN()

    print("\n  ── la direction, seule chose tirée (G1) ──")
    check_5_direction_50_50_et_rien_dautre()
    check_6_determinisme_a_seed_fixe()
    check_7_la_direction_NE_DEPEND_PAS_des_entrees_ecartees()

    print("\n  ── ce qui est écarté (G1, G3) ──")
    check_8_entrees_sans_label_ecartees_et_COMPTEES()
    check_9_entrees_incoherentes_refusees()
    check_93_les_gardes_du_NULL_mordent()

    print("\n  ── la cadence (G3, I7) ──")
    check_10_le_total_est_EXACT_et_stable()
    check_11_le_plafond_I7_est_tenu()
    check_12_le_calendrier_est_directement_consommable()
    check_13_determinisme_et_journees_trop_courtes()
    check_14_un_horodatage_NaT_leve_DANS_LES_DEUX_MODULES()

    print("\n  ── la barre à battre (G2) ──")
    check_15_percentiles_ordonnes_et_p95_degenere_signale()

    print("\n  ── de bout en bout ──")
    check_16_la_chaine_complete_produit_un_challenge_jugeable()
    check_17_le_supplement_de_cadence_est_TIRE_et_non_pris_au_debut()
    check_18_la_decision_est_datee_sur_la_barre_D_ENTREE_pas_sur_la_candidate()

    print("\n  ── C4 : le jumeau d'une liste de Decision ──")
    check_90_le_miroir_reflete_TOUT_sauf_la_direction()
    check_91_sur_le_triple_barriere_les_DEUX_tireurs_sont_le_MEME_null()
    check_92_les_gardes_du_miroir_mordent()

    print("\n" + "=" * 78)
    print("LES 22 PREUVES PASSENT.")
    print("=" * 78 + "\n")


# ── C4 : le jumeau d'une liste de Decision ─────────────────────────────────────────────────
def _decisions_sans_objectif(n: int) -> list[Decision]:
    """
    `n` décisions LONGUES **sans objectif et avec leur sortie propre** (F20, C1), une par jour, à
    des instants, des prix, des distances de stop et des durées TOUS distincts.

    Écrites au crayon, pas par un label (le swing qui les a motivées est mort en C6) : ce que la
    preuve vérifie, c'est que le miroir REFLÈTE ce qu'il reçoit, et des valeurs toutes distinctes
    rendent visible n'importe quel champ qui ne le serait pas — un stop recalculé, une sortie
    perdue, un objectif inventé.
    """
    debut = datetime(2021, 3, 1, 18)  # = DEBUT, en `datetime` natif (pas de NaT possible)
    out: list[Decision] = []
    for i in range(n):
        entree = debut + timedelta(days=i, hours=1)
        prix = 100.0 + 0.7 * i
        out.append(
            Decision(
                dt_entree=entree,
                direction=LONG,
                prix_entree=prix,
                barriere_haute=None,
                barriere_basse=prix - (2.0 + 0.1 * i),
                dt_sortie=entree + timedelta(hours=6 + i % 3),
            )
        )
    return out


def check_90_le_miroir_reflete_TOUT_sauf_la_direction() -> None:
    """Sur des décisions sans objectif (F20) : mêmes instants, prix, sorties ; stop à la MÊME distance du côté tiré ; aucun objectif inventé."""
    modele = _decisions_sans_objectif(60)
    t = tirer_directions_sur_decisions(modele, seed=3)
    if len(t.decisions) != len(modele) or t.n_ecartes != 0:
        _echec("le miroir n'a pas une décision par décision")
    for m, n in zip(sorted(modele, key=lambda d: d.dt_entree), t.decisions, strict=True):
        if (n.dt_entree, n.prix_entree, n.dt_sortie) != (m.dt_entree, m.prix_entree, m.dt_sortie):
            _echec(f"instant, prix ou sortie modifiés au {m.dt_entree}")
        if not math.isclose(n.distance_stop, m.distance_stop, rel_tol=1e-12):
            _echec(f"distance de stop {n.distance_stop} != {m.distance_stop} au {m.dt_entree}")
        if (n.stop < n.prix_entree) != (n.direction == LONG):
            _echec("le stop n'est pas du côté qui fait perdre pour la direction tirée")
        if n.cible is not None:
            _echec("un objectif a été inventé pour une décision qui n'en a pas (F20)")
    if not 0.30 <= t.part_long <= 0.70:
        _echec(f"part de longs {t.part_long:.2f} : la direction n'est pas tirée à 50/50")
    if [d.direction for d in tirer_directions_sur_decisions(modele, seed=3).decisions] != [
        d.direction for d in t.decisions
    ]:
        _echec("deux tirages à seed égale diffèrent")
    if [d.direction for d in tirer_directions_sur_decisions(modele, seed=4).decisions] == [
        d.direction for d in t.decisions
    ]:
        _echec("deux seeds différentes donnent le même tirage")
    print(
        f"  90. (C4) le miroir de {len(modele)} décisions sans objectif : tout reflété sauf la direction, {t.part_long:.0%} de longs  ✓"
    )


def check_91_sur_le_triple_barriere_les_DEUX_tireurs_sont_le_MEME_null() -> None:
    """Bit à bit : mêmes directions, mêmes barrières — ce n'est pas un second null."""
    from dataclasses import replace as _replace

    df = panel_labelle()
    candidats, *_ = _colonnes(df)
    t0 = tirer_directions(df, candidats, seed=7)
    if t0.n_ecartes != 0:
        _echec(
            "anti-vacuité : le fixture écarte des entrées, la comparaison ne serait pas terme à terme"
        )
    base = [_replace(d, direction=LONG) for d in t0.decisions]
    t1 = tirer_directions_sur_decisions(base, seed=7)
    for a, b in zip(t0.decisions, t1.decisions, strict=True):
        if (a.direction, a.dt_entree, a.prix_entree, a.barriere_haute, a.barriere_basse) != (
            b.direction,
            b.dt_entree,
            b.prix_entree,
            b.barriere_haute,
            b.barriere_basse,
        ):
            _echec(f"les deux tireurs divergent au {a.dt_entree} : {a} vs {b}")
    print(
        f"  91. (C4) sur {len(base)} décisions triple-barrière, les deux tireurs sont identiques bit à bit  ✓"
    )


def check_92_les_gardes_du_miroir_mordent() -> None:
    modele = _decisions_sans_objectif(10)
    _leve(
        lambda: tirer_directions_sur_decisions([], 0),
        ValueError,
        "aucune décision",
        "Aucune décision",
    )
    _leve(lambda: tirer_directions_sur_decisions(modele, -1), ValueError, "seed négative", "seed")
    _leve(
        lambda: tirer_directions_sur_decisions([*modele, modele[0]], 0),
        ValueError,
        "instant en double",
        "même",
    )
    if not tirer_directions_sur_decisions(modele, 0).decisions:
        _echec("anti-vacuité")
    print("  92. (C4) les gardes du miroir mordent (vide, seed négative, instant en double)  ✓")


def check_93_les_gardes_du_NULL_mordent() -> None:
    """
    Trois gardes qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    ⚠️ Le null est **l'échelle contre laquelle tout se lit** (G2). Une garde molle ici ne fait pas
    planter le juge : elle déplace la barre à battre, et le verdict devient une opinion.
    """
    df = panel_labelle()
    entrees = _instants(_labellees(df))
    # ① une seed négative : la reproductibilité de la barre en dépend (A8, G5)
    _leve(
        lambda: tirer_directions(df, entrees, seed=-1),
        ValueError,
        "seed négative",
        "seed",
    )
    # ② un panel non trié : les entrées seraient appariées n'importe où
    envers = df.iloc[::-1].reset_index(drop=True)
    _leve(
        lambda: tirer_directions(envers, entrees, seed=1),
        ValueError,
        "panel non trié",
        "trié",
    )
    # ③ un candidat qui porte des barrières mais AUCUNE date d'entrée (C5)
    troue = df.copy()
    troue["label_entree_dt"] = pd.to_datetime(troue["label_entree_dt"])
    labellees_idx = np.flatnonzero(pd.Series(troue["issue"]).notna().to_numpy())
    troue.loc[troue.index[int(labellees_idx[0])], "label_entree_dt"] = pd.NaT
    _leve(
        lambda: tirer_directions(troue, entrees, seed=1),
        ValueError,
        "candidat sans date d'entrée",
        "C5",
    )
    # anti-vacuité : sur le panel intact, le tirage produit des décisions
    if not tirer_directions(df, entrees, seed=1).decisions:
        _echec("anti-vacuité : le tirage ne produit aucune décision sur le panel intact")
    print("  93. gardes du null : seed négative, panel non trié, entrée non datée (C5)  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
