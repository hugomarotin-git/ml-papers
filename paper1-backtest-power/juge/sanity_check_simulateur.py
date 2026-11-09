"""
Preuve du simulateur FTMO — tâche 4.5, et elle valide aussi 4.2, 4.3 et 4.4.

Vingt-trois vérifications, toutes sur des séries **fabriquées** où le résultat se calcule au crayon.
Sur les vraies données, un P(funded) faux ne se signale pas : c'est un nombre entre 0 et 1,
d'allure raisonnable, qu'aucune autre mesure ne viendra contredire. C'est la zone du projet où
le faux silencieux coûte le plus cher.

Les quatre scénarios exigés par le plan sont les checks 1 à 4 : marche normale, bust journalier,
**bust total redevenu le plus contraignant**, et challenge financé de bout en bout. Le cinquième
cas exigé — une barrière de compte plus proche qu'un stop numériquement plus loin — est le
check 6, avec son symétrique au check 5.

Lancer :  .venv/bin/python juge/sanity_check_simulateur.py
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta

import pandas as pd
from aimd import AIMDConfig
from chemins import horizon_du_label
from regles_ftmo import Issue, ReglesFTMO
from simulateur import LONG, SHORT, Decision, horizon_global, simuler

R = ReglesFTMO()
DEBUT = datetime(2024, 6, 3, 18, 0)
PLAT = (100.0, 100.0, 100.0, 100.0)


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """
    Vérifie qu'un appel lève — et, avec `contient`, que c'est **pour la bonne raison**.

    ⚠️ Le contrôle du message n'est pas du zèle. `_valider_barres` lève une `ValueError` pour
    trois défauts distincts : colonne absente, panel vide, barres non triées. Sans vérifier le
    message, un bug qui ferait remonter le premier pour les trois cas laisserait les trois
    preuves passer. Le cas a été rencontré pour de vrai le 2026-08-09 dans la preuve de 4.6.
    """
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(
                f"{quoi} : {attendu.__name__} levée mais pour une AUTRE raison — attendu un "
                f"message contenant « {contient} », reçu « {str(e)[:90]} »"
            )
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def panel(journees: list[list[tuple[float, float, float, float]]]) -> pd.DataFrame:
    """
    Un panel de barres horaires découpé en **journées FTMO explicites**.

    Chaque sous-liste est une journée : c'est le simulateur qui doit LIRE cette colonne, jamais
    la recalculer. Le panel est donc l'endroit où l'on peut lui mentir — et le check 8 s'en sert
    pour vérifier qu'il ne triche pas avec les dates calendaires.
    """
    lignes: list[tuple[datetime, float, float, float, float, date]] = []
    t = DEBUT
    for n, jour in enumerate(journees):
        jour_ftmo = (DEBUT + timedelta(days=n + 1)).date()
        for o, h, b, c in jour:
            lignes.append((t, o, h, b, c, jour_ftmo))
            t += timedelta(hours=1)
    return pd.DataFrame(lignes, columns=["DateTime", "Open", "High", "Low", "Close", "jour_ftmo"])


def dt(i: int) -> datetime:
    """L'horodatage de la i-ème barre du panel."""
    return DEBUT + timedelta(hours=i)


def dec(i: int, direction: int = LONG, stop: float = 99.0, cible: float = 101.0) -> Decision:
    """Une décision entrant à la barre `i`, à 100 — l'Open de toutes les barres fabriquées."""
    haute, basse = (cible, stop) if direction == LONG else (stop, cible)
    return Decision(dt(i), direction, 100.0, haute, basse)


#: Une journée type : 2 barres plates, puis une barre gagnante ou perdante.
GAGNE = [PLAT, PLAT, (100.0, 101.0, 100.0, 101.0)]
PERD = [PLAT, PLAT, (100.0, 100.0, 99.0, 99.0)]


# ── Les quatre scénarios exigés par le plan ──────────────────────────────────────────────
def check_1_marche_normale() -> None:
    """Un trade gagnant, rien ne casse, la balance monte du risque exact."""
    r = simuler([dec(1)], panel([GAGNE]), risque_par_trade_pct=0.01)
    if r.issue is not Issue.EN_COURS:
        _echec(f"un simple trade gagnant ne doit rien terminer, reçu {r.issue.value}")
    if abs(r.balance_finale - 50_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale}, attendu 50 500 (RR 1:1, risque 1 %)")
    if r.n_trades != 1 or r.jours_tradees != 1:
        _echec(f"{r.n_trades} trade(s), {r.jours_tradees} journée(s) — attendu 1 et 1")
    print("  1. marche normale : un trade gagnant rapporte exactement le risque  ✓")


def check_2_bust_journalier() -> None:
    """
    Trois pertes à 2 % dans la MÊME journée : -3 000, sous le plancher journalier de -2 500.

    Le troisième trade ne va même pas jusqu'à son stop : à 48 000 de balance il ne reste que
    500 $ avant le plancher, soit moins que le risque du trade — F5 fait mordre le compte
    d'abord. C'est pour ça que la balance finale vaut le plancher PILE.
    """
    b = panel([PERD + PERD + PERD])
    r = simuler([dec(1), dec(4), dec(7)], b, risque_par_trade_pct=0.02)
    if r.issue is not Issue.BUST or r.motif_bust != "plancher_journalier":
        _echec(f"attendu bust journalier, reçu {r.issue.value} / {r.motif_bust}")
    if abs(r.balance_finale - 47_500.0) > 1e-9:
        _echec(f"un bust liquide AU plancher : balance {r.balance_finale}, attendu 47 500")
    print("  2. bust journalier : 2 500 $ sous la balance d'ouverture, liquidation au plancher  ✓")


def check_3_bust_total_redevenu_le_plus_contraignant() -> None:
    """
    Le scénario que le plan nomme explicitement, et le seul qui distingue les deux planchers.

    À 4 % de risque : jour 1 ouvre à 50 000 et perd 2 000 (plancher journalier 47 500, tenu).
    Jour 2 ouvre à 48 000, perd 2 000 (plancher 45 500, tenu). Jour 3 ouvre à **46 000** : son
    plancher journalier tombe à 43 500, **sous** le plancher total de 45 000. C'est donc le
    TOTAL qui mord désormais — un juge qui ne teste que le journalier laisserait passer.
    """
    b = panel([PERD, PERD, PERD])
    r = simuler([dec(1), dec(4), dec(7)], b, risque_par_trade_pct=0.04)
    if r.issue is not Issue.BUST:
        _echec(f"attendu un bust, reçu {r.issue.value}")
    if r.motif_bust != "plancher_total":
        _echec(
            f"motif {r.motif_bust} : au 3e jour le plancher journalier (43 500) est SOUS le "
            "total (45 000), c'est donc le TOTAL qui doit mordre"
        )
    if abs(r.balance_finale - 45_000.0) > 1e-9:
        _echec(f"balance {r.balance_finale}, attendu 45 000 (le plancher total, pile)")
    print("  3. bust TOTAL redevenu le plus contraignant, après deux journées perdantes  ✓")


def check_4_funded_de_bout_en_bout() -> None:
    """
    Phase 1 (+10 %) puis phase 2 (+5 %), 4 journées minimum chacune.

    À 4 % par trade, trois journées gagnantes suffisent à dépasser +10 % — mais la phase n'est
    validée qu'à la 4ᵉ (F16). Puis le compte repart de 50 000 (F17).

    ⚠️ **Une phase se valide dès l'OUVERTURE du 4ᵉ trade, pas à sa clôture.** Une journée est
    comptée quand un trade y est placé (c'est la règle FTMO), donc le couple « cible atteinte +
    4ᵉ journée » est réuni à l'instant où le 4ᵉ trade s'ouvre, la balance valant déjà 56 000
    grâce aux trois journées précédentes. Le 4ᵉ trade ne se dénoue jamais : la phase est finie.
    C'est pour ça que la balance attendue est **56 000 et non 58 000** — je m'étais trompé en
    écrivant cette preuve, et c'est le code qui avait raison.
    """
    b = panel([GAGNE] * 8)
    r = simuler([dec(1 + 3 * k) for k in range(8)], b, risque_par_trade_pct=0.04)
    if r.issue is not Issue.FUNDED:
        _echec(f"attendu FUNDED, reçu {r.issue.value} (balance {r.balance_finale})")
    if r.jours_tradees != R.jours_minimum:
        _echec(f"{r.jours_tradees} journées comptées en phase 2, attendu {R.jours_minimum}")
    # La balance finale est la seule chose qui distingue un compte NEUF d'un compte qui aurait
    # gardé le gain de la phase 1 : 50 000 + 4 gains, pas 58 000 + 4 gains. Sans cette ligne,
    # supprimer la remise à zéro de F17 passerait inaperçu — l'issue resterait FUNDED.
    if abs(r.balance_finale - 56_000.0) > 1e-9:
        _echec(
            f"balance finale {r.balance_finale}, attendu 56 000 : la phase 2 repart de 50 000 "
            "(F17) puis gagne 3 journées avant que la 4ᵉ ne valide à l'ouverture. Un compte qui "
            "aurait gardé le gain de la phase 1 finirait bien plus haut"
        )
    print("  4. challenge financé de bout en bout : phase 1, compte neuf, phase 2  ✓")


# ── F5 — le départage stop / barrière de compte, dans les deux sens ──────────────────────
def check_5_stop_plus_proche_que_le_plancher() -> None:
    """
    **Le bug que la version 4.2 fabriquait.** Barre qui plonge à 90, stop à 99, plancher atteint
    seulement à 95. Le stop est cinq fois plus proche de l'entrée : il ferme le trade en chemin,
    et l'équity ne descend JAMAIS jusqu'au bas de la barre.

    La 4.2 présentait au compte l'équity au plus bas (45 000) et déclarait un bust que le stop
    avait empêché. Si cette preuve échoue, c'est que ce bust fabriqué est revenu.
    """
    b = panel([[PLAT, (100.0, 100.0, 90.0, 90.0)]])
    r = simuler([dec(1)], b, risque_par_trade_pct=0.01)
    if r.issue is not Issue.EN_COURS:
        _echec(
            f"reçu {r.issue.value} : le stop (distance 1) est bien plus proche que le plancher "
            "(distance 5), il doit fermer le trade AVANT que le compte ne meure"
        )
    if abs(r.balance_finale - 49_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale}, attendu 49 500 — sortie AU stop, pas au bas de barre")
    print("  5. stop plus proche : le trade sort au stop, aucun bust fabriqué  ✓")


def check_6_plancher_plus_proche_que_le_stop() -> None:
    """
    L'autre sens, exigé par le plan : « une barrière plus proche touchée avant un SL
    numériquement plus loin ».

    Il faut avoir déjà perdu dans la journée pour y arriver — le plancher n'est plus proche que
    le stop que si `balance - plancher < risque du trade`. Après deux pertes à 2 %, la balance
    est à 48 000 : il reste 500 $ avant le plancher, soit un demi-point de prix, contre un point
    entier pour le stop.
    """
    b = panel([PERD + PERD + [PLAT, (100.0, 100.0, 99.0, 99.0)]])
    r = simuler([dec(1), dec(4), dec(7)], b, risque_par_trade_pct=0.02)
    if r.issue is not Issue.BUST:
        _echec(f"reçu {r.issue.value} : le compte meurt à 99,5, avant que le stop à 99 ne serve")
    if abs(r.balance_finale - 47_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale} : la liquidation se fait AU plancher (47 500)")
    print("  6. plancher plus proche : le compte meurt avant que le stop ne soit atteint  ✓")


# ── F3, F4 — équity et convention intra-barre ────────────────────────────────────────────
def check_7_bust_sur_flottant_sans_trade_cloture() -> None:
    """
    F3 — un compte peut sauter sans qu'aucun trade soit clôturé.

    Propriété à connaître : la perte flottante d'un trade est bornée par son risque, donc il
    faut un risque **supérieur** au plancher journalier (5 %) pour y parvenir avec un seul
    trade. Ici 8 %, stop à 10 de distance, prix descendant de 8 : le stop n'est jamais touché.
    """
    b = panel([[PLAT, (100.0, 100.0, 92.0, 92.0)]])
    d = Decision(dt(1), LONG, 100.0, 110.0, 90.0)
    r = simuler([d], b, risque_par_trade_pct=0.08)
    if r.issue is not Issue.BUST:
        _echec("le compte devait sauter sur du FLOTTANT, stop jamais touché")
    print("  7. bust sur du flottant, sans qu'aucun trade soit clôturé  ✓")


def check_8_journee_ftmo_lue_jamais_recalculee() -> None:
    """
    **B11 — le piège à 7 heures.** Une journée FTMO traverse minuit. Ici, une seule journée court
    de 22:00 à 03:00 : elle contient un changement de date calendaire.

    Le compte ouvre à 50 000 (plancher 47 500), perd jusqu'à 48 000 AVANT minuit, puis descend à
    47 400 APRÈS. Sous la règle correcte, 47 400 < 47 500 : bust. Un simulateur qui recalculerait
    le plancher à minuit le poserait à 48 000 x 0,95 = 45 600, et **ne verrait rien**.

    C'est exactement l'erreur du build précédent, et elle ne levait aucune exception.
    """
    # Six barres de 22:00 à 03:00 : elles traversent minuit et forment UNE SEULE journée FTMO.
    depart = datetime(2024, 6, 3, 22, 0)
    ohlc = [PLAT, (100.0, 100.0, 99.0, 99.0), PLAT, PLAT, (100.0, 100.0, 99.0, 99.0), PLAT]
    b = pd.DataFrame(
        {
            "DateTime": [depart + timedelta(hours=k) for k in range(len(ohlc))],
            "Open": [o for o, _, _, _ in ohlc],
            "High": [h for _, h, _, _ in ohlc],
            "Low": [x for _, _, x, _ in ohlc],
            "Close": [c for _, _, _, c in ohlc],
            "jour_ftmo": [date(2024, 6, 4)] * len(ohlc),
        }
    )
    if len(set(pd.Series(b["DateTime"]).dt.date)) < 2:
        _echec("le panel doit traverser minuit pour que la preuve ait un sens")

    # Risque 3 % = 1 500 $. Trade 1 (avant minuit) : 50 000 -> 48 500. Trade 2 (après minuit)
    # viserait 47 000, sous le plancher de 47 500 : BUST.
    # Sous la règle fausse — plancher recalculé à MINUIT — il vaudrait 48 500 x 0,95 = 46 075,
    # et 47 000 passerait tranquillement. Aucune exception, juste un P(funded) trop haut.
    d1 = Decision(depart, LONG, 100.0, 101.0, 99.0)
    d2 = Decision(depart + timedelta(hours=3), LONG, 100.0, 101.0, 99.0)
    r = simuler([d1, d2], b, risque_par_trade_pct=0.03)
    if r.issue is not Issue.BUST:
        _echec(
            f"reçu {r.issue.value} (balance {r.balance_finale}) : le plancher journalier a été "
            "recalculé à MINUIT au lieu de suivre la journée FTMO (B11) — 7 heures d'écart tous "
            "les jours, aucune erreur levée, P(funded) trop haut"
        )
    if abs(r.balance_finale - 47_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale}, attendu 47 500 (le plancher de la journée FTMO)")
    print("  8. la journée FTMO est LUE, pas recalculée : le plancher ne bouge pas à minuit  ✓")


def check_9_barre_touchant_les_deux_barrieres() -> None:
    """F4 / C7 — sur une barre qui touche stop ET cible, c'est le stop qui compte."""
    b = panel([[PLAT, (100.0, 101.0, 99.0, 100.5)]])
    r = simuler([dec(1)], b, risque_par_trade_pct=0.01)
    if abs(r.balance_finale - 49_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale}, attendu 49 500 — le stop l'emporte sur la cible")
    print("  9. barre touchant les deux barrières : convention pessimiste, le stop gagne  ✓")


def check_10_long_et_short_symetriques() -> None:
    """Un short doit gagner exactement ce qu'un long perd sur la même barre."""
    b = panel([PERD])
    rl = simuler([dec(1, LONG)], b, risque_par_trade_pct=0.01)
    rs = simuler([dec(1, SHORT, stop=101.0, cible=99.0)], b, risque_par_trade_pct=0.01)
    if abs((rl.balance_finale - 50_000.0) + (rs.balance_finale - 50_000.0)) > 1e-9:
        _echec(f"asymétrie long/short : {rl.balance_finale} et {rs.balance_finale}")
    print(" 10. long et short sont symétriques au centime près  ✓")


# ── F6 / C3 — la barrière temporelle ─────────────────────────────────────────────────────
def check_11_horizon_importe_du_label() -> None:
    """Le juge ne recopie pas l'horizon : il le lit à sa source unique."""
    if horizon_du_label() != horizon_global():
        _echec(f"horizon_global()={horizon_global()} != horizon du label={horizon_du_label()}")
    print(f" 11. la barrière temporelle EST l'horizon du label ({horizon_global()} barres)  ✓")


def check_12_cloture_forcee_a_lhorizon() -> None:
    """
    L'attendu se calcule **depuis** la constante, jamais depuis un nombre écrit ici : si
    l'horizon du label change un jour, cette preuve change avec lui.

    Vérification par l'effet observable : après la clôture forcée, le compte est libre, donc une
    seconde décision peut s'ouvrir. S'il ne fermait rien, elle serait ignorée pour cause de
    compte occupé.
    """
    n = horizon_global()
    b = panel([[PLAT] * (n + 4)])
    r = simuler([dec(1)], b, risque_par_trade_pct=0.01)
    if abs(r.balance_finale - 50_000.0) > 1e-9:
        _echec(f"clôture au Close d'une série plate : balance {r.balance_finale}, attendu 50 000")

    r2 = simuler([dec(1), dec(1 + n)], b, risque_par_trade_pct=0.01)
    if (r2.n_trades, r2.n_decisions_ignorees) != (2, 0):
        _echec(
            f"{r2.n_trades} trade(s) / {r2.n_decisions_ignorees} ignorée(s) : la barrière "
            f"temporelle n'a pas libéré le compte après {n} barres"
        )
    print(f" 12. clôture forcée après exactement {n} barres, compte libéré ensuite  ✓")


# ── Les garde-fous ───────────────────────────────────────────────────────────────────────
def check_13_risque_obligatoire_sans_defaut() -> None:
    """
    H9 — un risque hérité d'un périmètre abandonné survit sans que rien ne le signale.

    Deux refus DISTINCTS, et chacun est vérifié sur son propre message (piège n°4) :
    **ne rien passer** doit lever, et passer une valeur **hors ]0, 1[** doit lever. Le premier
    cas était couvert par la garde de source unique de l'AIMD v1 ; celle-ci est partie avec la
    table rase de 5.2, donc sans cette ligne « aucun défaut fourni » redeviendrait déclaratif.
    """
    b = panel([GAGNE])
    _leve(
        lambda: simuler([dec(1)], b),
        ValueError,
        "aucune source de sizing",
        contient="sizing est obligatoire",
    )
    # Critère de recette ① de 5.6 : la garde est à SOURCE UNIQUE. Les deux ensemble laisseraient
    # le code choisir en silence lequel l'emporte — et le P(funded) resterait plausible.
    _leve(
        lambda: simuler([dec(1)], b, risque_par_trade_pct=0.01, aimd=_cfg_aimd()),
        ValueError,
        "DEUX sources de sizing",
        contient="exactement une",
    )
    for mauvais in (0.0, 1.0, -0.01, 1.5):
        _leve(
            lambda m=mauvais: simuler([dec(1)], b, risque_par_trade_pct=m),
            ValueError,
            f"risque {mauvais}",
            contient="FRACTION",
        )
    print(" 13. sizing absent ET risque hors ]0,1[ refusés, aucun défaut fourni  ✓")


def check_14_prix_dentree_doit_etre_lopen_de_sa_barre() -> None:
    """
    C5 — l'entrée se fait à l'Open de t+1. Une décision qui annonce un autre prix a été
    construite sur d'autres données que celles qu'on simule : le juge évaluerait des entrées à
    un prix qui n'a jamais existé.
    """
    b = panel([GAGNE])
    fausse = Decision(dt(1), LONG, 100.5, 101.5, 99.5)
    _leve(
        lambda: simuler([fausse], b, risque_par_trade_pct=0.01),
        ValueError,
        "prix d'entrée != Open de la barre",
        contient="Open de la barre",
    )
    print(" 14. un prix d'entrée qui n'est pas l'Open de sa barre est REFUSÉ  ✓")


def check_15_barre_dentree_inexistante() -> None:
    b = panel([GAGNE])
    _leve(
        lambda: simuler([dec(99)], b, risque_par_trade_pct=0.01),
        ValueError,
        "barre d'entrée hors panel",
    )
    print(" 15. une décision pointant une barre inexistante est REFUSÉE  ✓")


def check_16_panel_invalide() -> None:
    b = panel([GAGNE])
    # Les trois défauts remontent de la MÊME fonction avec le même type d'exception : sans
    # contrôle du message, l'un d'eux pourrait masquer les deux autres.
    _leve(
        lambda: simuler([], b.drop(columns=["jour_ftmo"]), risque_par_trade_pct=0.01),
        ValueError,
        "colonne jour_ftmo absente",
        contient="jour_ftmo",
    )
    _leve(
        lambda: simuler([], b.iloc[0:0], risque_par_trade_pct=0.01),
        ValueError,
        "panel vide",
        contient="vide",
    )
    _leve(
        lambda: simuler([], b.iloc[::-1], risque_par_trade_pct=0.01),
        ValueError,
        "barres non triées",
        contient="triées",
    )
    print(" 16. panel sans `jour_ftmo`, vide ou non trié : REFUSÉ  ✓")


def check_17_decision_incoherente() -> None:
    _leve(lambda: Decision(dt(1), 0, 100.0, 101.0, 99.0), ValueError, "direction 0")
    _leve(lambda: Decision(dt(1), LONG, 100.0, 99.0, 101.0), ValueError, "barrières inversées")
    _leve(lambda: Decision(dt(1), LONG, 102.0, 101.0, 99.0), ValueError, "entrée hors barrières")
    print(" 17. une décision aux barrières incohérentes est REFUSÉE à la construction  ✓")


def check_18_compte_occupe() -> None:
    """H11 — un seul trade à la fois, et les décisions écartées sont COMPTÉES, pas perdues."""
    b = panel([[PLAT] * 5])
    r = simuler([dec(1), dec(2)], b, risque_par_trade_pct=0.01)
    if (r.n_trades, r.n_decisions_ignorees) != (1, 1):
        _echec(f"attendu 1 trade et 1 ignorée, reçu {r.n_trades} et {r.n_decisions_ignorees}")
    print(" 18. compte occupé : la 2e décision est ignorée ET comptée (4.9 la choisira)  ✓")


def check_19_pas_de_lot_non_implemente() -> None:
    """Une omission déclarée vaut mieux qu'un arrondi silencieux."""
    b = panel([GAGNE])
    _leve(
        lambda: simuler([dec(1)], b, risque_par_trade_pct=0.01, pas_de_lot=0.01),
        NotImplementedError,
        "pas de lot",
    )
    print(" 19. l'arrondi au pas de lot LÈVE au lieu d'être silencieusement ignoré  ✓")


def check_20_deterministe() -> None:
    b = panel([GAGNE, PERD, GAGNE])
    args = ([dec(1), dec(4), dec(7)], b)
    a = simuler(*args, risque_par_trade_pct=0.02)
    c = simuler(*args, risque_par_trade_pct=0.02)
    if (a.issue, a.balance_finale, a.equity) != (c.issue, c.balance_finale, c.equity):
        _echec("deux simulations identiques donnent des résultats différents")
    print(" 20. deux simulations identiques donnent exactement le même résultat  ✓")


def check_21_sizing_sur_le_capital_initial_pas_sur_la_balance() -> None:
    """
    H3 — le risque se mesure sur le capital **INITIAL**, jamais sur la balance courante.

    Deux pertes à 2 % doivent coûter exactement 2 x 1 000. Un sizing assis sur la balance
    coûterait 1 000 puis 980 : l'écart est de 20 $ sur deux trades, invisible à l'œil, et il
    compose sur des centaines de trades jusqu'à changer le P(funded).

    Sans cette preuve, la mutation « sizing sur la balance » passe : les scénarios de bust
    finissent tous à la liquidation au plancher, qui masque l'écart.
    """
    b = panel([PERD + PERD])
    r = simuler([dec(1), dec(4)], b, risque_par_trade_pct=0.02)
    if abs(r.balance_finale - 48_000.0) > 1e-9:
        _echec(
            f"balance {r.balance_finale}, attendu 48 000 exactement. Un sizing assis sur la "
            "balance courante donnerait 48 020 — le risque se mesure sur le capital INITIAL (H3)"
        )
    print(" 21. le risque se mesure sur le capital INITIAL, pas sur la balance courante  ✓")


def check_22_lancre_du_plancher_est_la_BALANCE_pas_lequity() -> None:
    """
    **Règle FTMO réelle (alignée le 2026-08-10)** : la ligne du jour se pose sur la **balance**
    de début de journée moins 2 500 $ — un flottant porté au passage de journée ne déplace PAS
    l'ancre. Il compte en revanche dans la violation, qui se teste sur l'équity (F3).

    Les chiffres sont choisis pour que les deux lectures **divergent** — c'est le scénario
    inverse de l'ancienne version de cette preuve, qui gravait la lecture équity. Risque 8 %,
    stop à 10 points : 400 unités.

    - Jour 1 : entrée à 100, le prix descend à 94. Flottant -2 400, équity 47 600 — au-dessus
      du plancher de 47 500. Le compte survit.
    - Jour 2 : il ouvre à 94, **position toujours ouverte**, aucun trade clôturé. L'ancre est
      la BALANCE : 50 000, plancher du jour **47 500**. Le prix descend à 93,5 : l'équity
      traverse 47 500 (à 93,75) — le compte est mort, liquidé au plancher.

    L'ancienne lecture posait la ligne sur l'équity d'ouverture (47 600 → plancher 45 220) et
    laissait ce compte survivre à 47 400 : 2 280 $ de marge fictive. Note : ce scénario ne peut
    pas naître de trades issus du label (aucun ne franchit la frontière de 17:00 NY, C9) — mais
    la règle doit être juste même sur les cas fabriqués à la main.
    """
    jour1 = [PLAT, PLAT, (100.0, 100.0, 94.0, 94.0)]
    jour2 = [(94.0, 94.0, 93.5, 93.5)]
    b = panel([jour1, jour2])
    d = Decision(dt(1), LONG, 100.0, 110.0, 90.0)  # stop à 10 points : il n'est jamais atteint
    r = simuler([d], b, risque_par_trade_pct=0.08)
    if r.issue is not Issue.BUST or r.motif_bust != "plancher_journalier":
        _echec(
            f"reçu {r.issue.value} / {r.motif_bust} (balance {r.balance_finale}) : au jour 2 "
            "l'ancre est la BALANCE (50 000, aucun trade clôturé) donc le plancher vaut 47 500, "
            "et l'équity le traverse à 93,75. Le poser sur l'équity d'ouverture (47 600 -> "
            "45 220) laisserait survivre un compte que FTMO ferme"
        )
    if abs(r.balance_finale - 47_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale!r} : la liquidation se fait AU plancher (47 500)")
    print(" 22. l'ancre du plancher journalier est la BALANCE de début de journée (FTMO réel)  ✓")


def check_23_bust_au_plancher_sur_des_prix_NON_RONDS() -> None:
    """
    **Le piège flottant de la liquidation au plancher — classe F18, côté simulateur.**

    Même scénario que le check 6 (deux pertes réalisées, puis le flottant du 3ᵉ trade touche le
    plancher journalier), mais sur des prix NON RONDS, choisis pour que l'aller-retour
    `prix_pour_equity` → `pnl` ressorte à **+2,2e-11 au-dessus** du plancher.

    Reconstruire l'équity présentée à `observer` via `pnl(prix_plancher)` au lieu d'affecter la
    valeur du plancher elle-même faisait alors échouer le test `equity <= plancher` : le compte
    était liquidé AU plancher... et le challenge continuait, EN_COURS, à 2e-11 du plancher. Un
    bust raté environ une fois sur deux sur des flottants quelconques — les checks 2, 3 et 6
    passaient parce que leurs nombres ronds (entrée 100, distance 1) tombent juste.

    Vérifié sur ce cas exact avant correction : issue = en_cours, balance = 47 500,000000000022.
    """
    e1, d1 = 19_273.4, 121.3
    e2, d2 = 19_180.7, 113.9
    e3, d3 = 19_071.2, 87.3
    bas3 = e3 - 60.0  # sous le prix-plancher (43,65 pts), au-dessus du stop (87,3 pts)
    jour = [
        (e1, e1, e1, e1),
        (e1, e1, e1 - d1, e1 - d1),
        (e2, e2, e2 - d2, e2 - d2),
        (e3, e3, bas3, bas3),
        (bas3, bas3, bas3, bas3),
    ]
    b = panel([jour])
    decisions = [
        Decision(dt(1), LONG, e1, e1 + d1, e1 - d1),
        Decision(dt(2), LONG, e2, e2 + d2, e2 - d2),
        Decision(dt(3), LONG, e3, e3 + d3, e3 - d3),
    ]
    r = simuler(decisions, b, risque_par_trade_pct=0.02)
    if r.issue is not Issue.BUST or r.motif_bust != "plancher_journalier":
        _echec(
            f"reçu {r.issue.value} / {r.motif_bust} (balance {r.balance_finale!r}) : le compte a "
            "touché son plancher journalier sur des prix non ronds et doit buster. S'il est "
            "EN_COURS à ~1e-11 du plancher, l'équity présentée à `observer` a été reconstruite "
            "par un aller-retour flottant au lieu d'être le plancher lui-même (classe F18)"
        )
    if abs(r.balance_finale - 47_500.0) > 1e-9:
        _echec(f"balance {r.balance_finale!r} : la liquidation se fait AU plancher (47 500)")
    print(" 23. bust au plancher sur des prix non ronds : pas d'aller-retour flottant  ✓")


def check_24_cout_par_trade_non_fini_refuse() -> None:
    """
    **Piège n°8, audit du 2026-08-18.** `cout_par_trade` n'était validé nulle part.

    Il vaut zéro jusqu'à la tâche 10.1 — mais c'est précisément 10.1 qui commencera à y passer de
    vraies valeurs. Un `NaN` mettrait la **balance** à `NaN` dès le premier trade, et à partir de
    là `observer` ne déclencherait **plus jamais** : toute comparaison avec `NaN` est fausse, y
    compris le `<=` du plancher. Aucun bust, aucune cible, tous les chemins EN_COURS.

    Un coût **négatif** serait un gain encaissé à chaque ouverture — le compte gonflerait d'autant
    plus qu'il trade, ce qui est exactement l'inverse d'un coût.
    """
    b = panel([[PLAT] * 10])
    d = [Decision(dt(1), LONG, 100.0, 101.0, 99.0)]
    for valeur, quoi in ((float("nan"), "NaN"), (float("inf"), "inf"), (-1.0, "négatif")):
        _leve(
            lambda v=valeur: simuler(d, b, risque_par_trade_pct=0.01, cout_par_trade=v),
            ValueError,
            f"un cout_par_trade {quoi}",
            contient="cout_par_trade",
        )
    print(" 24. un cout_par_trade NaN, infini ou négatif est REFUSÉ (piège n°8)  ✓")


# ── Le branchement du sizing adaptatif (tâche 5.6) ───────────────────────────────────────
def _cfg_aimd(**remplacements) -> AIMDConfig:
    """Une config d'essai. `AIMDConfig` n'a **aucun** défaut (H9) : ils sont tous ici."""
    base = dict(
        risque_base=0.01,
        hausse_additive=0.25,
        seuil_journees_perdantes=2,
        facteur_coupe=0.5,
        m_initial=1.0,
        m_min=0.25,
        m_max=1.25,
        reset_au_passage_de_phase=True,
        airbag_n=None,
    )
    base.update(remplacements)
    return AIMDConfig(**base)  # pyright: ignore[reportArgumentType]


def check_25_config_DEGENEREE_identique_au_sizing_constant() -> None:
    """
    **Critère de recette ④ de 5.6, et le seul qui prouve que brancher le sizing n'a rien cassé
    au passage.**

    Une config où `m` ne peut pas bouger (`m_min = m_max = m_initial`, airbag désactivé) doit
    reproduire le sizing constant **au centime près**. Si le branchement avait décalé un appel,
    inversé une balance ou compté une journée de travers, l'écart apparaîtrait ici — et nulle
    part ailleurs, parce que tout le reste continuerait de produire des chiffres plausibles.
    """
    # ⚠️ **La séquence est volontairement ASYMÉTRIQUE.** Une alternance gagnante/perdante
    # ramène le compte à 50 000 pile : les deux simulations tomberaient d'accord sur un chiffre
    # qui ne dépend NI du risque NI du branchement, et le check passerait sans rien prouver.
    # C'est le piège du « point neutre » rencontré en 5.3, transposé au branchement.
    b = panel([GAGNE, PERD, GAGNE, GAGNE, PERD, GAGNE])  # 4 gagnantes, 2 perdantes
    d = [dec(i) for i in range(0, len(b), 3)]
    constant = simuler(d, b, risque_par_trade_pct=0.01)
    if constant.balance_finale == R.capital_initial:
        _echec("l'essai ne discrimine pas : le compte finit exactement au capital initial")
    degenere = simuler(d, b, aimd=_cfg_aimd(m_initial=1.0, m_min=1.0, m_max=1.0))

    if constant.balance_finale != degenere.balance_finale:
        _echec(
            f"balance {degenere.balance_finale!r} (dégénéré) contre "
            f"{constant.balance_finale!r} (constant) : le branchement a changé le résultat"
        )
    if constant.risques != degenere.risques:
        _echec("les risques appliqués diffèrent entre config dégénérée et sizing constant")
    if (constant.issue, constant.n_trades, constant.n_journees) != (
        degenere.issue,
        degenere.n_trades,
        degenere.n_journees,
    ):
        _echec("issue, nombre de trades ou de journées diffèrent")
    if degenere.n_montees or degenere.n_coupes or degenere.n_declenchements_airbag:
        _echec("une config dégénérée a bougé : le diagnostic devrait être vide")
    print(
        f" 25. config dégénérée ≡ sizing constant, au centime près ({constant.balance_finale:,.2f})  ✓"
    )


def check_26_le_sizing_adaptatif_fait_reellement_quelque_chose() -> None:
    """
    Le pendant du check 25 : une config **non** dégénérée doit produire des risques qui
    **bougent**. Sans ce contrôle, le check 25 serait satisfait par un branchement qui ignore
    complètement l'AIMD — le pire des faux silencieux, puisque tout resterait vert.
    """
    b = panel([GAGNE] * 6)  # six journées gagnantes : `m` doit monter, donc le risque aussi
    d = [dec(i) for i in range(0, len(b), 3)]
    r = simuler(d, b, aimd=_cfg_aimd())
    if len(set(r.risques)) < 2:
        _echec(f"les risques ne bougent jamais : {sorted(set(r.risques))} — l'AIMD est inerte")
    if r.n_montees == 0:
        _echec("aucune montée comptée alors que les risques ont bougé : le diagnostic ment")
    if sum(r.bilans_journee.values()) != max(r.n_journees - 1, 0):
        _echec(
            f"{sum(r.bilans_journee.values())} journées classées pour {r.n_journees} vécues : "
            "il doit y en avoir exactement une de moins (la dernière n'a pas de frontière après)"
        )
    print(f" 26. le sizing adaptatif bouge réellement ({len(set(r.risques))} risques distincts)  ✓")


def check_27_aucune_memoire_entre_deux_simulations() -> None:
    """
    **Critère de recette ② de 5.6 — H4 ② rendue STRUCTURELLE.** L'état est créé par `simuler`,
    donc deux appels ne peuvent rien partager. Le jour où ils partageraient une mémoire,
    `P(funded)` cesserait d'être une probabilité pour devenir un artefact d'ordonnancement — et
    le chiffre resterait parfaitement plausible.

    On le prouve par le seul moyen honnête : jouer une séquence **perdante** qui coupe `m`, puis
    rejouer la même simulation et exiger un résultat **identique**.
    """
    b = panel([PERD] * 6)  # six journées perdantes d'affilée : `m` coupe (k = 2)
    d = [dec(i) for i in range(0, len(b), 3)]
    config = _cfg_aimd()
    premier = simuler(d, b, aimd=config)
    second = simuler(d, b, aimd=config)
    if premier.risques != second.risques or premier.balance_finale != second.balance_finale:
        _echec("deux simulations sur la MÊME config diffèrent : un état a survécu entre elles")
    if premier.n_coupes == 0:
        _echec("l'essai ne discrimine pas : aucune coupe n'a eu lieu, donc rien à hériter")
    print(
        f" 27. aucune mémoire entre deux simulations ({premier.n_coupes} coupes, à l'identique)  ✓"
    )


def check_28_le_bilan_porte_sur_la_BALANCE_pas_sur_lequity() -> None:
    """
    **Site d'appel, H2 ① — et le seul cas où l'erreur est visible.**

    Passer l'équity au lieu de la balance à la frontière ferait entrer un résultat **non réalisé**
    dans une décision de sizing, et le même résultat serait compté **deux fois** : une fois
    flottant, une fois réalisé.

    ⚠️ **Ce n'est détectable que si une position est OUVERTE à la frontière** — sinon les deux
    valeurs sont égales et la mutation est invisible. Sur les vraies données ce cas n'existe pas
    (C9 interdit qu'un trade enjambe la frontière), donc il faut le **fabriquer** : une journée
    qui perd 500 $ en RÉALISÉ tout en portant +750 $ de FLOTTANT au moment de sa clôture.

    Lue sur la balance, cette journée est **perdante**. Lue sur l'équity, elle serait **gagnante**
    — et ferait monter `m` au lieu de le laisser. Le bilan bascule, donc la mutation tombe.
    """
    perdant = (100.0, 100.0, 99.0, 99.0)  # touche le stop à 99
    flottant = (100.0, 101.5, 100.0, 101.5)  # +1,5 pt non réalisé, cible lointaine non touchée
    # ⚠️ La barre flottante est la PREMIÈRE de la journée suivante, pas la dernière de la
    # précédente : c'est la barre de la frontière que le simulateur regarde pour connaître
    # l'équity. Placée un cran trop tôt, le flottant vaut zéro à l'instant qui compte et la
    # mutation devient invisible — c'est ce qui est arrivé au premier jet de ce check.
    b = panel([[PLAT, perdant, PLAT], [flottant, PLAT, PLAT], [PLAT] * 3])
    decisions = [
        dec(0),  # se ferme au stop dans la journée : -500 $ RÉALISÉS
        Decision(dt(2), LONG, 100.0, 105.0, 99.0),  # reste ouvert à la frontière : +750 $ flottants
    ]
    r = simuler(decisions, b, aimd=_cfg_aimd())

    if r.bilans_journee.get("perdante", 0) < 1:
        _echec(
            f"la 1ʳᵉ journée n'est pas classée perdante : {r.bilans_journee} — le flottant de "
            "+750 $ a été compté, donc c'est l'ÉQUITY qui est passée et non la balance (H2 ①)"
        )
    if r.bilans_journee.get("gagnante", 0) != 0:
        _echec(
            f"une journée gagnante est apparue : {r.bilans_journee} — le flottant a basculé le bilan"
        )
    print("  28. le bilan de journée porte sur la BALANCE, position ouverte à la frontière  ✓")


def check_29_lairbag_est_reellement_branche_et_lit_le_bon_plancher() -> None:
    """
    **Site d'appel, H16.** Trois choses d'un coup, qu'aucun autre check n'exerce parce qu'ils
    désactivent tous l'airbag :

    1. l'airbag **mord** à travers le branchement (sinon il serait décoratif dans la simulation
       tout en étant prouvé dans son module) ;
    2. il est appelé **exactement une fois par trade** — le décompte est exact, donc un second
       appel gonflerait le compteur et ferait lire une config comme plus protectrice qu'elle
       n'est, ce qui est précisément la mesure de la phase 6 ;
    3. il vise le plancher **le plus proche des deux** : au matin, c'est le **journalier**
       (47 500) et non le total (45 000). Sur le total, la marge vaudrait 5 000 $ et l'airbag ne
       mordrait pas du tout — chez FTMO les deux tuent, en ignorer un revient à mesurer une marge
       qui n'existe pas.
    """
    b = panel([GAGNE] * 3)
    d = [dec(i) for i in range(0, len(b), 3)]
    config = _cfg_aimd(risque_base=0.03, m_initial=1.0, m_min=0.5, m_max=1.0, airbag_n=2)
    r = simuler(d, b, aimd=config)

    # marge du matin = 2 500 $ (plancher journalier), N = 2 -> plafond 2,5 % < nominal 3 %.
    attendu = 2_500.0 / (2 * R.capital_initial)
    if any(not math.isclose(x, attendu, rel_tol=1e-12) for x in r.risques):
        _echec(
            f"risques {r.risques}, attendu {attendu} partout — l'airbag ne mord pas au bon niveau"
        )
    if r.n_declenchements_airbag != len(r.risques):
        _echec(
            f"{r.n_declenchements_airbag} déclenchements pour {len(r.risques)} trades : "
            "`risque_effectif` n'est pas appelée exactement une fois par trade"
        )
    print(
        f"  29. airbag branché : mord à {attendu:.4f}, {r.n_declenchements_airbag} fois pour {len(r.risques)} trades  ✓"
    )


def check_30_reset_du_sizing_au_passage_en_phase_2() -> None:
    """
    **Critère de recette ⑤ de 5.6** — F17, H4 ③. La phase 2 est un compte **neuf** : le capital
    repart du capital initial, et la confiance aussi.

    Sans ce reset, la phase 2 hériterait du multiplicateur accumulé en phase 1 — un compte neuf
    qui démarre avec la confiance d'un compte qui a déjà gagné. Rien ne planterait, et le
    `P(funded)` serait optimiste d'autant.
    """
    b = panel([GAGNE] * 8)
    d = [dec(i) for i in range(0, len(b), 3)]
    config = _cfg_aimd(risque_base=0.03, m_initial=0.5, m_min=0.25, m_max=1.5)
    r = simuler(d, b, aimd=config)

    depart = 0.03 * 0.5
    if not math.isclose(r.risques[0], depart, rel_tol=1e-12):
        _echec(f"le 1er trade sort à {r.risques[0]}, attendu risque_base x m₀ = {depart}")
    if len(r.risques) < 5:
        _echec(
            f"seulement {len(r.risques)} trades : la phase 2 n'est pas atteinte, essai non concluant"
        )
    if r.risques[3] <= r.risques[0]:
        _echec("`m` n'a pas monté en phase 1 : l'essai ne discriminerait pas le reset")
    if not math.isclose(r.risques[4], depart, rel_tol=1e-12):
        _echec(
            f"le 1er trade de la phase 2 sort à {r.risques[4]}, attendu {depart} : le sizing n'a "
            "PAS été remis à zéro au passage de phase — la phase 2 hérite de la confiance de la 1"
        )

    # Et la lecture inverse doit être observable, sinon le champ serait décoratif (H4 ③).
    sans = simuler(
        d,
        b,
        aimd=_cfg_aimd(
            risque_base=0.03, m_initial=0.5, m_min=0.25, m_max=1.5, reset_au_passage_de_phase=False
        ),
    )
    if math.isclose(sans.risques[4], depart, rel_tol=1e-12):
        _echec("reset=False donne le même résultat que reset=True : le champ ne sert à rien")
    print(
        f"  30. reset au passage en phase 2 : {r.risques[3]:.4f} -> {r.risques[4]:.4f} (retour à m₀)  ✓"
    )


def check_31_les_clotures_sont_comptees_a_la_FERMETURE_pas_a_louverture() -> None:
    """
    **Site d'appel, H2 ① / H15.** Compter les trades à l'**ouverture** au lieu de la **fermeture**
    est le genre d'erreur qu'on croit indétectable : sur les vraies données, un trade ouvre et
    ferme dans la même journée FTMO (C9 interdit qu'il enjambe la frontière), donc les deux
    décomptes sont **identiques**. Le plan de 5.6 l'annonçait comme telle.

    Elle devient détectable dès qu'on **fabrique** une journée dont le trade ne conclut pas :

    - journée 1 : un trade perd → **perdante**, série à 1 ;
    - journée 2 : un trade **s'ouvre sans se fermer**, balance inchangée. Comptée à la fermeture,
      elle est **sans trade** — transparente, la série reste à 1 (H15). Comptée à l'ouverture,
      elle devient **nulle** et **casse la série** ;
    - journée 3 : ce trade se ferme au stop → **perdante**, série à 2 → **la coupe tombe**.

    Avec le décompte à l'ouverture, la coupe **n'a jamais lieu** : `m` reste à 1 et le trade
    suivant sort deux fois trop gros. C'est exactement ce que H15 protège — *effacer une série de
    pertes sur une journée vide, c'est remettre le compteur de danger à zéro sans que rien n'ait
    montré que le mauvais régime est fini.*
    """
    stop = (100.0, 100.0, 99.0, 99.0)
    b = panel([[PLAT, stop, PLAT], [PLAT] * 3, [stop, PLAT, PLAT], [PLAT] * 3])
    decisions = [dec(0), dec(3), dec(9)]
    r = simuler(decisions, b, aimd=_cfg_aimd())

    attendus = {"perdante": 2, "sans_trade": 1}
    for nom, n in attendus.items():
        if r.bilans_journee.get(nom, 0) != n:
            _echec(
                f"bilans {r.bilans_journee}, attendu {attendus} — une journée dont le trade "
                "n'a pas CONCLU a été comptée comme tradée (décompte à l'ouverture)"
            )
    if r.n_coupes != 1:
        _echec(
            f"{r.n_coupes} coupe(s) : la série de deux journées perdantes n'a pas été vue. "
            "Une journée sans conclusion a cassé la série, ce que H15 interdit"
        )
    if len(r.risques) != 3 or r.risques[2] >= r.risques[0]:
        _echec(f"risques {r.risques} : le trade qui suit la coupe devrait être plus petit")
    print(
        f"  31. clôtures comptées à la FERMETURE : la coupe tombe ({r.risques[0]:.4f} -> {r.risques[2]:.4f})  ✓"
    )


def check_32_H11_tient_TANT_QU_AUCUN_SEUIL_DE_COMPTE_N_EST_FRANCHI() -> None:
    """
    **L'amendement de H11 — tâche 5.7.** H11 promet que la séquence de trades (quoi, quand) est
    indépendante du sizing et peut être précalculée. **La promesse est trop forte, et on mesure
    ici exactement où elle casse.**

    **Elle tient tant qu'aucun seuil de compte n'est franchi.** Sur un challenge qui reste en
    cours, sizing constant et sizing adaptatif prennent **les mêmes trades et en ignorent les
    mêmes**, alors que le risque appliqué, lui, a bien bougé. C'est ce que H11 demandait de
    re-prouver contre le code v2, et c'est ce qui rend H8 possible : comparer deux configs sur
    exactement la même séquence.

    **Elle casse dès qu'un seuil est franchi** — passage en phase 2, ou conclusion du challenge.
    Deux mécanismes, tous deux dans `simuler` :

    1. au **passage de phase**, la position ouverte est fermée (`position = None`) : le compte se
       libère plus tôt qu'il ne l'aurait fait, donc une décision qui aurait été ignorée est prise ;
    2. à la **conclusion**, la boucle s'arrête : le reste de la séquence n'est jamais joué.

    Dans les deux cas, la **date** du franchissement dépend du rythme d'accumulation, donc du
    sizing. Le code a raison ; c'est la ligne de H11 qui ment, et elle est amendée.

    ⚠️ **Pas de mutation dédiée, et c'est délibéré.** La partie « tient » est **structurelle** :
    la table décision → barre est construite **avant** la boucle et la garde d'occupation ne
    consulte jamais le sizing. Aucune ligne existante ne peut être mutée pour la casser — il
    faudrait en **ajouter** une, c'est-à-dire écrire du code uniquement pour le casser. Le projet
    a déjà refusé ce marché une fois : du code qu'aucun test ne peut exercer donne une fausse
    confiance. C'est donc une **assertion de non-régression**, au même titre que le check 1 de
    `sanity_check_methodes`.
    """
    # ── Partie 1 : aucun seuil franchi -> H11 tient au trade près ────────────────────────
    b = panel([GAGNE, PERD] * 4)  # net ~ nul : ni funded ni bust
    d = [dec(i) for i in range(len(b) - 1)]  # une décision à CHAQUE barre : l'occupation mord
    cst = simuler(d, b, risque_par_trade_pct=0.01)
    ad = simuler(d, b, aimd=_cfg_aimd())

    if cst.issue is not Issue.EN_COURS or ad.issue is not Issue.EN_COURS:
        _echec("l'essai ne discrimine pas : un seuil a été franchi, la partie 1 n'a plus d'objet")
    if len(set(ad.risques)) < 2:
        _echec("le sizing adaptatif n'a pas bougé : l'égalité des séquences serait triviale")
    if (cst.n_trades, cst.n_decisions_ignorees) != (ad.n_trades, ad.n_decisions_ignorees):
        _echec(
            f"séquences différentes sans franchissement de seuil — constant "
            f"({cst.n_trades} pris, {cst.n_decisions_ignorees} ignorés) contre AIMD "
            f"({ad.n_trades}, {ad.n_decisions_ignorees}). H11 devrait tenir ici, et H8 avec."
        )

    # ── Partie 2 : un seuil est franchi -> la séquence DIVERGE ───────────────────────────
    b2 = panel([GAGNE] * 30)
    d2 = [dec(i) for i in range(len(b2) - 1)]
    # Appels écrits en clair plutôt que par dépaquetage de dictionnaire : le dépaquetage prive
    # pyright de tout contrôle de type, et c'est lui qui attrape la classe d'erreur du 2026-08-03.
    joues = {
        "constant": simuler(d2, b2, risque_par_trade_pct=0.01).n_trades,
        "aimd_m0_1": simuler(d2, b2, aimd=_cfg_aimd()).n_trades,
        "aimd_m0_025": simuler(d2, b2, aimd=_cfg_aimd(m_initial=0.25)).n_trades,
    }
    if len(set(joues.values())) < 2:
        _echec(
            f"séquences identiques malgré des seuils franchis : {joues}. L'exception de H11 "
            "n'est plus observable, donc l'amendement ne décrit plus le code"
        )
    print(f"  32. H11 amendée : identique sans seuil franchi, divergente avec ({joues})  ✓")


def check_33_H10_le_sizing_najuste_QUE_le_pourcentage_de_risque() -> None:
    """
    **H10 — l'AIMD n'ajuste QUE le pourcentage de risque**, et rien d'autre : ni la distance du
    stop, ni le choix des trades, ni la direction, ni la cadence. L'adaptation à la volatilité
    reste à l'ATR, via la distance du stop portée par le label. **Deux mécanismes distincts, et
    les mélanger est l'erreur que H10 nomme.**

    On le prouve par le seul chiffre qui les sépare : **la perte encaissée au stop doit valoir
    exactement `capital_initial x risque_effectif`**, quel que soit `m`. Si le multiplicateur
    fuyait dans le nombre d'unités — par exemple en divisant aussi la distance du stop — le
    risque affiché resterait juste et la perte réelle serait fausse. Le diagnostic mentirait sans
    qu'aucun autre check ne s'en aperçoive : `risques` enregistre le risque **voulu**, pas celui
    qui a été encaissé.

    ⚠️ Le multiplicateur vaut **0,5** ici, pas 1 : au point neutre, une fuite de `m` dans les
    unités serait une division par 1, donc invisible. Même piège que les checks 8 et 13 du sizing.
    """
    b = panel([[PLAT, (100.0, 100.0, 99.0, 99.0), PLAT]])
    d = [dec(0)]
    config = _cfg_aimd(risque_base=0.02, m_initial=0.5, m_min=0.25, m_max=1.25)
    r = simuler(d, b, aimd=config)

    if len(r.risques) != 1:
        _echec(f"{len(r.risques)} trade(s), attendu 1 — l'essai ne mesure rien")
    if not math.isclose(r.risques[0], 0.01, rel_tol=1e-12):
        _echec(f"risque affiché {r.risques[0]}, attendu risque_base x m = 0.02 x 0.5 = 0.01")

    perte = R.capital_initial - r.balance_finale
    attendue = R.capital_initial * r.risques[0]
    if not math.isclose(perte, attendue, rel_tol=1e-12):
        _echec(
            f"perte encaissée au stop {perte:,.2f} $ contre {attendue:,.2f} $ annoncés par le "
            "risque : le multiplicateur a fui ailleurs que dans le pourcentage (H10) — la "
            "distance du stop ou le nombre d'unités, très probablement"
        )
    print(
        f"  33. H10 : la perte au stop vaut exactement capital x risque ({perte:,.0f} $, m = 0,5)  ✓"
    )


# ── C1 : le contrat `Decision` généralisé — sortie propre, objectif optionnel ─────────────
def check_34_dt_sortie_ferme_a_l_OPEN_de_sa_barre_et_libere_le_compte() -> None:
    """
    ⚠️⚠️ Fixture hostile : la barre de sortie **krache** (Low loin sous le stop) et **flambe**
    (High loin au-dessus de l'objectif), avec un Open à 102 et un Close à 100. Si le simulateur
    lisait cette barre, le stop ou la cible fermerait le trade ; s'il sortait au Close, le gain
    serait nul. La seule réponse juste est +2 points par unité, à l'Open — et le compte libre.
    """
    b = panel([[PLAT] * 3 + [(102.0, 150.0, 50.0, 100.0)] + [PLAT] * 4])
    d = Decision(dt(1), LONG, 100.0, 101.0, 99.0, dt_sortie=dt(3))
    r = simuler([d], b, risque_par_trade_pct=0.01)
    unites = 50_000.0 * 0.01 / 1.0
    if abs(r.balance_finale - (50_000.0 + 2.0 * unites)) > 1e-9:
        _echec(f"balance {r.balance_finale} : la sortie n'est pas à l'Open de la barre de sortie")
    # Le compte est LIBRE à la barre de sortie : une décision qui y entre — à SON Open, 102 (C5) —
    # s'ouvre. Stop et objectif larges : elle ne doit se fermer que par l'horizon.
    suivante = Decision(dt(3), LONG, 102.0, 130.0, 80.0)
    r2 = simuler([d, suivante], b, risque_par_trade_pct=0.01)
    if (r2.n_trades, r2.n_decisions_ignorees) != (2, 0):
        _echec(
            f"{r2.n_trades} trade(s) / {r2.n_decisions_ignorees} ignorée(s) : le compte n'est "
            "pas libre à la barre de sortie"
        )
    print(
        " 34. (C1) dt_sortie ferme à l'OPEN de sa barre — krach et flambée ignorés — et libère "
        "le compte  ✓"
    )


def check_35_dt_sortie_n_est_PAS_coupee_par_l_horizon_global() -> None:
    """Une décision qui porte sa sortie vit jusqu'à sa barre, même bien au-delà de l'horizon du label."""
    n = horizon_global()
    b = panel([[PLAT] * (2 * n + 6)])
    longue = Decision(dt(1), LONG, 100.0, 101.0, 99.0, dt_sortie=dt(2 * n + 2))
    r = simuler([longue, dec(n + 2)], b, risque_par_trade_pct=0.01)
    if (r.n_trades, r.n_decisions_ignorees) != (1, 1):
        _echec(
            f"{r.n_trades} trade(s) / {r.n_decisions_ignorees} ignorée(s) : l'horizon global a coupé une décision qui portait sa sortie"
        )
    # anti-vacuité : la même décision SANS dt_sortie est coupée par l'horizon, et la seconde s'ouvre.
    r2 = simuler([dec(1), dec(n + 2)], b, risque_par_trade_pct=0.01)
    if (r2.n_trades, r2.n_decisions_ignorees) != (2, 0):
        _echec("anti-vacuité : sans dt_sortie, l'horizon global devrait libérer le compte")
    print(
        f" 35. (C1) une décision avec dt_sortie vit {2 * n + 1} barres — l'horizon de {n} ne la coupe pas  ✓"
    )


def check_36_sans_objectif_rien_ne_fait_gagner_avant_la_sortie() -> None:
    """
    `barriere_haute=None` (long) / `barriere_basse=None` (short) : une flambée ne clôt rien.

    L'anti-vacuité passe par l'OCCUPATION, pas par la balance : avec un objectif à 101, la
    flambée de la barre 1 ferme le trade et libère le compte pour la barre 2 ; sans objectif, la
    position est encore ouverte à la barre 2 et la décision qui s'y présente est ignorée.
    """
    b = panel([[PLAT, (100.0, 150.0, 100.0, 100.0), PLAT, (101.0, 101.0, 101.0, 101.0), PLAT]])
    sans_tp = Decision(dt(1), LONG, 100.0, None, 99.0, dt_sortie=dt(3))
    if sans_tp.cible is not None:
        _echec("une décision sans objectif rend une cible")
    unites = 50_000.0 * 0.01 / 1.0
    r = simuler([sans_tp, dec(2)], b, risque_par_trade_pct=0.01)
    if abs(r.balance_finale - (50_000.0 + 1.0 * unites)) > 1e-9:
        _echec(f"balance {r.balance_finale} : la flambée à 150 a été prise pour un objectif touché")
    if (r.n_trades, r.n_decisions_ignorees) != (1, 1):
        _echec("sans objectif, la position devrait encore occuper le compte à la barre 2")
    # anti-vacuité : AVEC un objectif à 101, la flambée ferme le trade et la barre 2 est libre.
    avec_tp = Decision(dt(1), LONG, 100.0, 101.0, 99.0, dt_sortie=dt(3))
    r2 = simuler([avec_tp, dec(2)], b, risque_par_trade_pct=0.01)
    if (r2.n_trades, r2.n_decisions_ignorees) != (2, 0):
        _echec("anti-vacuité : avec un objectif, la flambée devrait fermer le trade à la barre 1")
    # Symétrie : un short sans objectif bas, face à un krach.
    bs = panel([[PLAT, (100.0, 100.0, 50.0, 100.0), PLAT, (99.0, 99.0, 99.0, 99.0), PLAT]])
    court = Decision(dt(1), SHORT, 100.0, 101.0, None, dt_sortie=dt(3))
    rs = simuler([court], bs, risque_par_trade_pct=0.01)
    if abs(rs.balance_finale - (50_000.0 + 1.0 * unites)) > 1e-9:
        _echec(
            f"short sans objectif : balance {rs.balance_finale}, le krach à 50 pris pour un objectif"
        )
    print(
        " 36. (C1) sans objectif, une flambée (long) ou un krach (short) ne ferme rien — la "
        "sortie seule le fait  ✓"
    )


def check_37_les_gardes_du_contrat_MORDENT() -> None:
    """Le stop ne peut pas manquer ni être du mauvais côté ; la sortie doit suivre l'entrée et exister."""
    _leve(
        lambda: Decision(dt(1), LONG, 100.0, 101.0, None),
        ValueError,
        "long sans stop",
        "côté du stop",
    )
    _leve(
        lambda: Decision(dt(1), SHORT, 100.0, None, 99.0),
        ValueError,
        "short sans stop",
        "côté du stop",
    )
    _leve(
        lambda: Decision(dt(1), LONG, 100.0, None, 101.0),
        ValueError,
        "stop long au-dessus de l'entrée",
        "mauvais côté",
    )
    _leve(
        lambda: Decision(dt(1), LONG, 100.0, 101.0, 99.0, dt_sortie=dt(1)),
        ValueError,
        "sortie à la barre d'entrée",
        "STRICTEMENT",
    )
    b = panel([[PLAT] * 6])
    _leve(
        lambda: simuler(
            [Decision(dt(1), LONG, 100.0, 101.0, 99.0, dt_sortie=dt(40))],
            b,
            risque_par_trade_pct=0.01,
        ),
        ValueError,
        "barre de sortie absente du panel",
        "n'existe pas",
    )
    # anti-vacuité : une décision sans objectif mais AVEC stop passe, et se simule.
    r = simuler(
        [Decision(dt(1), LONG, 100.0, None, 99.0, dt_sortie=dt(4))], b, risque_par_trade_pct=0.01
    )
    if r.n_trades != 1:
        _echec("anti-vacuité : une décision sans objectif ne se simule plus")
    print(
        " 37. (C1) les gardes du contrat mordent : stop obligatoire et du bon côté, sortie après l'entrée et réelle  ✓"
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU SIMULATEUR FTMO — tâche 4.5 (valide aussi 4.2, 4.3 et 4.4)")
    print("=" * 78 + "\n")

    print("  ── les quatre scénarios exigés par le plan ──")
    check_1_marche_normale()
    check_2_bust_journalier()
    check_3_bust_total_redevenu_le_plus_contraignant()
    check_4_funded_de_bout_en_bout()

    print("\n  ── F5 : le départage stop / barrière de compte, dans les deux sens ──")
    check_5_stop_plus_proche_que_le_plancher()
    check_6_plancher_plus_proche_que_le_stop()
    check_23_bust_au_plancher_sur_des_prix_NON_RONDS()

    print("\n  ── F3, F4, B11 : équity, convention intra-barre, journée FTMO ──")
    check_7_bust_sur_flottant_sans_trade_cloture()
    check_8_journee_ftmo_lue_jamais_recalculee()
    check_9_barre_touchant_les_deux_barrieres()
    check_10_long_et_short_symetriques()

    print("\n  ── F6 / C3 : la barrière temporelle ──")
    check_11_horizon_importe_du_label()
    check_12_cloture_forcee_a_lhorizon()

    print("\n  ── C1 : le contrat Decision — sortie propre, objectif optionnel ──")
    check_34_dt_sortie_ferme_a_l_OPEN_de_sa_barre_et_libere_le_compte()
    check_35_dt_sortie_n_est_PAS_coupee_par_l_horizon_global()
    check_36_sans_objectif_rien_ne_fait_gagner_avant_la_sortie()
    check_37_les_gardes_du_contrat_MORDENT()

    print("\n  ── les garde-fous ──")
    check_13_risque_obligatoire_sans_defaut()
    check_14_prix_dentree_doit_etre_lopen_de_sa_barre()
    check_15_barre_dentree_inexistante()
    check_16_panel_invalide()
    check_17_decision_incoherente()
    check_18_compte_occupe()
    check_19_pas_de_lot_non_implemente()
    check_20_deterministe()

    print("\n  ── sizing et référentiels (H3) ──")
    check_21_sizing_sur_le_capital_initial_pas_sur_la_balance()
    check_22_lancre_du_plancher_est_la_BALANCE_pas_lequity()
    check_24_cout_par_trade_non_fini_refuse()
    check_25_config_DEGENEREE_identique_au_sizing_constant()
    check_26_le_sizing_adaptatif_fait_reellement_quelque_chose()
    check_27_aucune_memoire_entre_deux_simulations()
    check_28_le_bilan_porte_sur_la_BALANCE_pas_sur_lequity()
    check_29_lairbag_est_reellement_branche_et_lit_le_bon_plancher()
    check_30_reset_du_sizing_au_passage_en_phase_2()
    check_31_les_clotures_sont_comptees_a_la_FERMETURE_pas_a_louverture()
    check_32_H11_tient_TANT_QU_AUCUN_SEUIL_DE_COMPTE_N_EST_FRANCHI()
    check_33_H10_le_sizing_najuste_QUE_le_pourcentage_de_risque()
    check_93_les_gardes_D_INSTALLATION_mordent()

    print("\n" + "=" * 78)
    print("LES 38 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_les_gardes_D_INSTALLATION_mordent() -> None:
    """
    Les deux gardes qui s'exécutent **à l'import** — audit du 2026-09-21.

    ⚠️ Elles ne sont atteignables qu'en cassant l'installation, donc aucune preuve ne les voyait :
    elles étaient **déclaratives**. Or ce sont elles qui garantissent que le juge IMPORTE son
    horizon de `src/` au lieu de le recopier (C3, F6) — une constante recopiée marche le jour où
    on l'écrit et diverge en silence le jour où l'original change.

    Chacune se vérifie dans un **sous-processus** : une garde d'import ne peut pas se tester dans
    le processus qui a déjà réussi l'import.
    """
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path

    racine = Path(__file__).resolve().parent.parent

    # ① `chemins` sans dossier `src/` voisin : on copie le module seul, ailleurs.
    with tempfile.TemporaryDirectory() as d:
        faux = Path(d) / "juge"
        faux.mkdir()
        (faux / "chemins.py").write_bytes((racine / "juge" / "chemins.py").read_bytes())
        r = subprocess.run(
            [_sys.executable, "-c", "import chemins"],
            cwd=faux,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            _echec("`chemins` s'importe SANS dossier src/ : la garde d'installation est décorative")
        if "introuvable" not in r.stderr.lower():
            _echec(f"`chemins` lève pour une autre raison — « {r.stderr.strip()[-120:]} »")

    # ② `simulateur` avec un horizon de label incohérent : on injecte un faux `chemins`.
    #    ⚠️ Depuis C1 (2026-09-23), l'horizon est lu à l'APPEL, plus à l'import : l'import doit
    #    RÉUSSIR, et c'est `horizon_global()` qui doit lever. Une décision qui porte sa propre
    #    sortie ne touche jamais cette garde — c'est voulu.
    code = (
        "import sys, types\n"
        "faux = types.ModuleType('chemins')\n"
        "faux.horizon_du_label = lambda: 0\n"
        "faux.SRC = ''\n"
        "sys.modules['chemins'] = faux\n"
        "import simulateur\n"
        "simulateur.horizon_global()\n"
    )
    r = subprocess.run(
        [_sys.executable, "-c", code], cwd=racine / "juge", capture_output=True, text=True
    )
    if r.returncode == 0:
        _echec("`horizon_global()` accepte un horizon de 0 : le juge jugerait hors du label (C3)")
    if "horizon du label incohérent" not in r.stderr.lower():
        _echec(f"`horizon_global` lève pour une autre raison — « {r.stderr.strip()[-120:]} »")
    if "import simulateur" in r.stderr and "Traceback" in r.stderr.split("horizon_global")[0]:
        _echec("l'IMPORT du simulateur a levé : l'horizon est encore évalué à l'import (C1)")

    # ③ LA COUPURE (C1) : importer le juge ne charge PLUS le label triple-barrière.
    code = (
        "import sys, chemins, simulateur, methodes_pfunded\n"
        "assert 'label_triple_barriere' not in sys.modules, 'ancien label charge a l import'\n"
    )
    r = subprocess.run(
        [_sys.executable, "-c", code], cwd=racine / "juge", capture_output=True, text=True
    )
    if r.returncode != 0:
        _echec(f"le juge charge encore l'ancien label à l'import — « {r.stderr.strip()[-160:]} »")

    # anti-vacuité : l'import normal, lui, fonctionne, et l'horizon se lit bien à l'appel.
    r = subprocess.run(
        [_sys.executable, "-c", "import chemins, simulateur; simulateur.horizon_global()"],
        cwd=racine / "juge",
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        _echec(f"anti-vacuité : l'import normal échoue — « {r.stderr.strip()[-120:]} »")
    print(
        "  93. gardes d'installation : src/ introuvable, horizon incohérent à l'APPEL, et l'ancien label n'est plus chargé à l'import (C1)  ✓"
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
