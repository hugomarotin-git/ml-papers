"""
Preuve des deux méthodes de P(funded) — tâche 4.7.

Dix-neuf vérifications. La plus importante est la **numéro 1** : `resoudre_trades` duplique la
résolution intra-barre du simulateur, ce que D3 met en garde de faire. La contrepartie exigée
par D3 est une assertion de non-régression entre les deux implémentations — c'est le check 1, et
s'il tombe, c'est que les deux ont divergé en silence.

Le reste verrouille ce que les décisions imposent et que rien ne signalerait autrement : le
plancher de taille de bloc (F9), le budget borné de B (F13), l'interdiction de moyenner (F8),
l'ordre de lecture de l'écart (F10), et le chevauchement des chemins de A (D4 transposé).

Lancer :  .venv/bin/python juge/sanity_check_methodes.py
"""

from __future__ import annotations

import sys

from aimd import AIMDConfig
from methodes_pfunded import (
    BUDGET_B,
    ComparaisonAB,
    TradeRealise,
    _jouer_sequence,
    chemins_effectifs,
    methode_a,
    methode_b,
    plus_longue_serie_perdante,
    resoudre_trades,
    resultats_methode_a,
)
from regles_ftmo import Issue, ReglesFTMO
from sanity_check_simulateur import GAGNE, PERD, PLAT, dec, panel
from simulateur import LONG, SHORT, simuler
from trajectoires import agreger

R = ReglesFTMO()


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(f"{quoi} : levée pour une AUTRE raison — « {str(e)[:90]} »")
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def tr(rendements: list[float], nouvelles: list[bool] | None = None) -> list[TradeRealise]:
    """Une séquence de trades fabriquée, où l'on choisit tout."""
    n = nouvelles if nouvelles is not None else [True] * len(rendements)
    # `strict=True` : deux listes de longueurs différentes signaleraient un test mal construit,
    # et la troncature silencieuse de zip le cacherait.
    return [
        TradeRealise(rendement=r, nouvelle_journee=j) for r, j in zip(rendements, n, strict=True)
    ]


# ── LA preuve : les deux implémentations de la résolution ne divergent pas ────────────────
def check_1_non_regression_contre_le_simulateur() -> None:
    """
    **D3.** `resoudre_trades` réimplémente la résolution intra-barre pour produire la séquence
    indépendante du sizing (H11). Deux implémentations d'une même règle divergent tôt ou tard —
    la seule protection est de les confronter.

    On joue chaque décision **seule** dans le simulateur (compte intact, aucune barrière de
    compte atteignable) et on compare son P&L au rendement annoncé, en multiples de risque.
    """
    cas = [
        ("long gagnant", GAGNE, LONG, 1.0),
        ("long perdant", PERD, LONG, -1.0),
        ("short gagnant", PERD, SHORT, 1.0),
        ("short perdant", GAGNE, SHORT, -1.0),
        ("timeout à plat", [PLAT] * 8, LONG, 0.0),
        ("barre touchant les deux", [PLAT, (100.0, 101.0, 99.0, 100.5)], LONG, -1.0),
    ]
    risque = 0.01
    for nom, jour, sens, attendu in cas:
        b = panel([jour])
        d = dec(1, sens) if sens == LONG else dec(1, SHORT, stop=101.0, cible=99.0)
        rendus = resoudre_trades([d], b)
        if abs(rendus[0].rendement - attendu) > 1e-9:
            _echec(f"{nom} : rendement {rendus[0].rendement}, attendu {attendu}")

        # Le simulateur, sur le même trade : son P&L doit valoir rendement x risque x capital.
        r = simuler([d], b, risque_par_trade_pct=risque)
        pnl_simule = r.balance_finale - R.capital_initial
        pnl_attendu = rendus[0].rendement * risque * R.capital_initial
        if abs(pnl_simule - pnl_attendu) > 1e-6:
            _echec(
                f"{nom} : DIVERGENCE entre les deux implémentations — simulateur {pnl_simule}, "
                f"resoudre_trades {pnl_attendu}. C'est exactement ce que D3 redoute"
            )
    print("  1. NON-RÉGRESSION : resoudre_trades et le simulateur donnent le même P&L (D3)  ✓")


def check_2_rendement_independant_du_sizing() -> None:
    """H11 — le rendement est en multiples de risque, donc le sizing ne fait que multiplier."""
    b = panel([GAGNE, PERD])
    rendus = resoudre_trades([dec(1), dec(4)], b)
    for risque in (0.005, 0.01, 0.03):
        r = simuler([dec(1), dec(4)], b, risque_par_trade_pct=risque)
        attendu = sum(t.rendement for t in rendus) * risque * R.capital_initial
        if abs((r.balance_finale - R.capital_initial) - attendu) > 1e-6:
            _echec(f"risque {risque} : le P&L ne suit pas proportionnellement le sizing")
    print("  2. le rendement est en multiples de risque : le sizing ne fait que multiplier  ✓")


def check_3_drapeau_de_nouvelle_journee() -> None:
    """
    Ce drapeau est ce que le bootstrap doit préserver : deux pertes le même jour partagent le
    plancher journalier, deux pertes sur deux jours non. C'est la différence entre un compte
    mort et un compte vivant, à trades identiques.
    """
    b = panel([PERD + PERD, PERD])  # 2 trades le 1er jour, 1 le second
    rendus = resoudre_trades([dec(1), dec(4), dec(7)], b)
    if [t.nouvelle_journee for t in rendus] != [True, False, True]:
        _echec(f"drapeaux {[t.nouvelle_journee for t in rendus]}, attendu [True, False, True]")
    print("  3. le drapeau « nouvelle journée FTMO » suit le vrai découpage  ✓")


def check_4_un_bust_ninterrompt_pas_la_sequence() -> None:
    """H11 — la séquence est une propriété du marché, pas du capital. Aucun compte ici."""
    b = panel([PERD + PERD + PERD + PERD])
    decisions = [dec(1), dec(4), dec(7), dec(10)]
    rendus = resoudre_trades(decisions, b)
    if len(rendus) != 4:
        _echec(f"{len(rendus)} trades résolus, attendu 4 : un bust ne doit rien interrompre ici")
    r = simuler(decisions, b, risque_par_trade_pct=0.02)
    if r.issue is not Issue.BUST:
        _echec("le scénario devait buster côté simulateur, pour que la preuve ait un sens")
    print("  4. la séquence de trades est complète, même là où un compte aurait sauté  ✓")


# ── F9 : la taille de bloc ───────────────────────────────────────────────────────────────
def check_5_plus_longue_serie_perdante() -> None:
    if plus_longue_serie_perdante(tr([1, -1, -1, -1, 1, -1])) != 3:
        _echec("la plus longue série perdante devrait valoir 3")
    if plus_longue_serie_perdante(tr([1, 1, 1])) != 0:
        _echec("aucune perte : la série vaut 0")
    if plus_longue_serie_perdante(tr([0.0, -0.0])) != 0:
        _echec("un rendement nul n'est PAS une perte")
    print("  5. la plus longue série perdante réelle est mesurée exactement  ✓")


def check_6_bloc_trop_court_refuse() -> None:
    """
    **F9.** Un bloc plus court que la pire série vécue ne peut plus jamais la reproduire : le
    bootstrap sous-estime alors le risque de bust, sans que rien ne le signale. C'est le genre
    de réglage qui donne un résultat flatteur et introuvable.
    """
    trades = tr([1, -1, -1, -1, -1, 1, 1])  # série perdante de 4
    for court in (1, 2, 3):
        _leve(
            lambda c=court: methode_b(trades, 0.01, taille_bloc=c, n_chemins=5, seed=1),
            ValueError,
            f"bloc de {court}",
            contient="série perdante",
        )
    methode_b(trades, 0.01, taille_bloc=4, n_chemins=5, seed=1)  # doit passer
    print("  6. une taille de bloc sous la pire série perdante est REFUSÉE (F9)  ✓")


def check_7_bloc_plus_grand_que_la_serie_refuse() -> None:
    trades = tr([1, -1, 1, -1])
    _leve(
        lambda: methode_b(trades, 0.01, taille_bloc=99, n_chemins=3, seed=1),
        ValueError,
        "bloc plus grand que la série",
        contient="dépasse le nombre de trades",
    )
    print("  7. un bloc plus grand que l'historique est REFUSÉ (plus rien d'aléatoire)  ✓")


# ── F13 : le budget borné ────────────────────────────────────────────────────────────────
def check_8_budget_borne_et_censure() -> None:
    """
    **F13.** Une séquence qui ne conclut jamais — que des rendements nuls — doit s'arrêter au
    budget et ressortir censurée, jamais boucler.
    """
    trades = tr([0.0] * 10)
    d = methode_b(trades, 0.01, taille_bloc=2, n_chemins=5, seed=7)
    if d.n_censures != 5:
        _echec(f"{d.n_censures} censurés sur 5 : une séquence à rendement nul ne conclut jamais")
    if d.n_chemins != 5:
        _echec(f"{d.n_chemins} chemins, attendu 5")
    print(f"  8. budget borné à len(trades) x {BUDGET_B} : aucune boucle infinie (F13)  ✓")


def check_9_deterministe_a_seed_fixe() -> None:
    """A8 : rien n'est stocké, tout se recalcule. Donc à seed égale, résultat identique."""
    trades = tr([1, -1, 1, 1, -1, -1, 1, 1, 1, -1])
    a = methode_b(trades, 0.02, taille_bloc=3, n_chemins=40, seed=42)
    b = methode_b(trades, 0.02, taille_bloc=3, n_chemins=40, seed=42)
    c = methode_b(trades, 0.02, taille_bloc=3, n_chemins=40, seed=43)
    if (a.n_funded, a.n_bust, a.n_censures) != (b.n_funded, b.n_bust, b.n_censures):
        _echec("même seed, résultats différents : le bootstrap n'est pas déterministe")
    if (a.n_funded, a.n_bust) == (c.n_funded, c.n_bust) and a.n_chemins > 10:
        _echec("deux seeds différentes donnent exactement le même tirage : suspect")
    print("  9. bootstrap déterministe à seed fixe, différent à seed différente (A8, H8)  ✓")


def check_10_seed_et_n_chemins_obligatoires() -> None:
    trades = tr([1, -1, 1, -1])
    _leve(
        lambda: methode_b(trades, 0.01, taille_bloc=2, n_chemins=0, seed=1),
        ValueError,
        "n_chemins nul",
        contient="au moins 1",
    )
    _leve(
        lambda: methode_b([], 0.01, taille_bloc=2, n_chemins=5, seed=1),
        ValueError,
        "aucun trade",
        contient="rien à tirer",
    )
    print(" 10. un bootstrap sans trades ou sans chemins est REFUSÉ  ✓")


def check_11_le_clustering_journalier_est_preserve() -> None:
    """
    **Le cœur de la méthode B.** Mêmes rendements, mêmes proportions — seul le regroupement
    journalier change. Le compte doit survivre dans un cas et mourir dans l'autre.

    Si cette preuve tombe, le bootstrap a perdu le drapeau de journée et mesure autre chose.

    ⚠️ On appelle ici `_jouer_sequence` directement, **sans passer par le bootstrap**, et c'est
    délibéré : le tirage répète les blocs jusqu'au budget, donc douze pertes finissent par
    toucher le plancher TOTAL même étalées, ce qui noierait la démonstration. Ma première
    version faisait cette erreur et échouait pour une raison qui n'avait rien à voir avec le
    clustering.
    """
    pertes = [-1.0, -1.0, -1.0]
    groupees = _jouer_sequence(tr(pertes, [True, False, False]), 0.02, R)
    etalees = _jouer_sequence(tr(pertes, [True, True, True]), 0.02, R)

    if groupees.issue is not Issue.BUST:
        _echec("3 pertes de 2 % le MÊME jour = -6 %, sous le plancher journalier : bust attendu")
    if groupees.motif_bust != "plancher_journalier":
        _echec(f"motif {groupees.motif_bust}, attendu plancher_journalier")
    if etalees.issue is Issue.BUST:
        _echec(
            "3 pertes de 2 % sur 3 journées différentes ne doivent PAS buster : le plancher se "
            "remet à zéro chaque jour, et 47 000 reste au-dessus du plancher total de 45 000"
        )
    if abs(etalees.balance_finale - 47_000.0) > 1e-9:
        _echec(f"balance étalée {etalees.balance_finale}, attendu 47 000")
    print(" 11. le clustering journalier est préservé : mêmes trades, une issue opposée  ✓")


# ── Méthode A ────────────────────────────────────────────────────────────────────────────
def check_12_un_chemin_par_depart() -> None:
    b = panel([GAGNE] * 6)
    decisions = [dec(1 + 3 * k) for k in range(6)]
    d = methode_a(decisions, b, risque_par_trade_pct=0.01)
    if d.n_chemins != 6:
        _echec(f"{d.n_chemins} chemins pour 6 décisions, attendu 6")
    d2 = methode_a(decisions, b, risque_par_trade_pct=0.01, pas=2)
    if d2.n_chemins != 3:
        _echec(f"pas=2 : {d2.n_chemins} chemins, attendu 3")
    print(" 12. méthode A : un chemin par départ, `pas` les espace  ✓")


def check_13_chemins_effectifs_inferieurs_aux_chemins() -> None:
    """
    **D4 transposé.** Les fenêtres se chevauchent : compter les départs comme indépendants
    rétrécit artificiellement les intervalles de confiance et fabrique de la certitude.
    """
    d = agreger([])
    if chemins_effectifs(d, 100) != 0.0:
        _echec("sans chemin résolu, les effectifs valent 0, pas une division par zéro")

    b = panel([GAGNE] * 8)
    dist = methode_a([dec(1 + 3 * k) for k in range(8)], b, risque_par_trade_pct=0.04)
    eff = chemins_effectifs(dist, n_journees_totales=8)
    if eff > dist.n_chemins:
        _echec(f"effectifs {eff} > chemins {dist.n_chemins} : impossible")
    if eff <= 0:
        _echec("des chemins résolus existent : les effectifs doivent être > 0")
    print(f" 13. chemins effectifs ({eff:.1f}) <= chemins joués ({dist.n_chemins})  ✓")


def check_14_pas_invalide_refuse() -> None:
    b = panel([GAGNE])
    _leve(lambda: methode_a([dec(1)], b, 0.01, pas=0), ValueError, "pas nul", contient="au moins 1")
    _leve(lambda: methode_a([], b, 0.01), ValueError, "aucune décision", contient="aucun départ")
    print(" 14. un `pas` nul ou une liste vide sont REFUSÉS  ✓")


def check_20_l_agregat_des_resultats_bruts_EST_la_methode_A() -> None:
    """
    9.2 (2026-09-26) : `methode_a` = `agreger(resultats_methode_a(...))`, bit à bit, y compris avec
    `pas` et un budget. C'est ce qui autorise à FUSIONNER plusieurs séquences (les chemins CPCV)
    avant d'agréger, au lieu de moyenner des distributions — faux dès que les séquences n'ont pas
    le même nombre de départs.
    """
    b = panel([GAGNE, PERD, GAGNE, GAGNE, PERD, GAGNE, GAGNE, GAGNE])
    decisions = [dec(1 + 3 * k) for k in range(8)]
    for pas, budget in ((1, None), (2, None), (1, 3)):
        d = methode_a(decisions, b, risque_par_trade_pct=0.02, pas=pas, budget_journees=budget)
        r = resultats_methode_a(
            decisions, b, risque_par_trade_pct=0.02, pas=pas, budget_journees=budget
        )
        if agreger(r) != d:
            _echec(f"pas {pas}, budget {budget} : agreger(resultats) ≠ methode_a")
        if len(r) != d.n_chemins:
            _echec(
                f"pas {pas}, budget {budget} : {len(r)} résultats bruts pour {d.n_chemins} chemins"
            )
    # la fusion de deux séquences = l'agrégat de l'union, et PAS la moyenne des deux agrégats
    a, c = decisions[:5], decisions[5:]
    ra = resultats_methode_a(a, b, risque_par_trade_pct=0.02)
    rc = resultats_methode_a(c, b, risque_par_trade_pct=0.02)
    fusion = agreger(ra + rc)
    if (
        fusion.n_chemins != len(ra) + len(rc)
        or fusion.n_funded != agreger(ra).n_funded + agreger(rc).n_funded
    ):
        _echec("la fusion ne somme pas les chemins joués")
    _leve(lambda: resultats_methode_a([], b, 0.01), ValueError, "aucune décision", "aucun départ")
    print(
        " 20. (9.2) agreger(resultats_methode_a) == methode_a bit à bit ; la fusion somme les chemins  ✓"
    )


# ── F8, F10 : la comparaison ─────────────────────────────────────────────────────────────
def check_15_conservative_est_le_minimum_jamais_la_moyenne() -> None:
    """**F8.** Une moyenne des deux n'aurait aucun sens : elles ne mesurent pas la même chose."""
    from sanity_check_trajectoires import lot

    a, bb = agreger(lot(3, 7, 0)), agreger(lot(6, 4, 0))  # 30 % et 60 %
    c = ComparaisonAB(a=a, b=bb)
    if abs(c.conservative - 0.30) > 1e-12:
        _echec(f"conservative = {c.conservative}, attendu 0,30 (le MINIMUM des deux)")
    if abs(c.conservative - 0.45) < 1e-9:
        _echec("conservative vaut la MOYENNE des deux — F8 l'interdit explicitement")
    if abs(c.ecart - 0.30) > 1e-12:
        _echec(f"écart = {c.ecart}, attendu 0,30")
    print(" 15. le chiffre retenu est le MINIMUM des deux, jamais la moyenne (F8)  ✓")


def check_16_conservative_symetrique() -> None:
    """Le minimum ne dépend pas de l'ordre : si B est sous A, c'est B qui est retenue."""
    from sanity_check_trajectoires import lot

    a, bb = agreger(lot(8, 2, 0)), agreger(lot(2, 8, 0))  # 80 % et 20 %
    if abs(ComparaisonAB(a=a, b=bb).conservative - 0.20) > 1e-12:
        _echec("B sous A : c'est B qui doit être retenue")
    if abs(ComparaisonAB(a=bb, b=a).conservative - 0.20) > 1e-12:
        _echec("le minimum doit être symétrique")
    print(" 16. le minimum est retenu quel que soit le sens de l'écart  ✓")


def check_17_ordre_de_lecture_de_lecart() -> None:
    """
    **F10.** Un grand écart se lit AVANT de s'interpréter, et la non-conclusion de A se lit en
    premier — sinon on attribue à une fragilité ce qui n'est qu'un manque de recul.
    """
    from sanity_check_trajectoires import lot

    # A très censurée + gros écart : le message sur A doit venir EN PREMIER.
    c = ComparaisonAB(a=agreger(lot(2, 3, 45)), b=agreger(lot(60, 40, 0)))
    messages = c.lecture()
    if not messages or "LIRE D'ABORD" not in messages[0]:
        _echec(f"la non-conclusion de A doit être annoncée en premier (F10) : {messages}")
    if not any("ÉCART A/B" in m for m in messages):
        _echec("un écart de plus de 10 points doit être signalé")
    if not any("gonflé" in m for m in messages):
        _echec("le biais de budget de B (F13) doit être rappelé avec l'écart")
    print(" 17. l'écart A/B se lit dans l'ordre imposé : la censure de A d'abord (F10)  ✓")


def check_18_b_sous_a_est_signale_comme_serieux() -> None:
    """
    B est structurellement avantagée par son budget (F13). Si elle passe quand même SOUS A,
    c'est que le clustering des pertes coûte réellement quelque chose.
    """
    from sanity_check_trajectoires import lot

    c = ComparaisonAB(a=agreger(lot(60, 40, 0)), b=agreger(lot(30, 70, 0)))
    if not any("SOUS A" in m for m in c.lecture()):
        _echec("B sous A malgré son biais de budget doit être signalé comme sérieux")
    print(" 18. B passant sous A malgré son avantage de budget est signalé  ✓")


def check_19_F11_la_methode_B_REFUSE_un_sizing_adaptatif() -> None:
    """
    **Critère de recette ③ de 5.6 — F11 redevient EXÉCUTABLE.**

    B rebat les trades par blocs, alors qu'un sizing adaptatif dépend entièrement de l'ordre réel
    des résultats et de la frontière des journées FTMO. Lui passer une config produirait un
    `P(funded)` parfaitement lisible et **dénué de sens** — le pire des résultats, puisque rien
    ne signalerait qu'il ne veut rien dire.

    La v1 portait déjà ce paramètre, et il n'existait **que pour lever**. Il est parti avec la
    table rase de 5.2 ; entre-temps, F11 n'était plus qu'une phrase dans une docstring. **Une
    convention écrite finit par être contournée ; une garde qui lève, non.**
    """
    config = AIMDConfig(
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
    barres = panel([GAGNE, PERD, GAGNE, GAGNE, PERD, GAGNE])
    candidats = [dec(i) for i in range(0, len(barres), 3)]
    trades = resoudre_trades(candidats, barres)
    _leve(
        lambda: methode_b(trades, 0.01, taille_bloc=8, n_chemins=4, seed=1, aimd=config),
        NotImplementedError,
        "un sizing adaptatif passé à la méthode B",
        contient="F11",
    )
    # Et la méthode A, elle, l'accepte : la garde vise B, pas le sizing adaptatif en général.
    d = methode_a(candidats, barres, aimd=config, pas=2)
    if d.n_chemins < 1:
        _echec("la méthode A doit accepter une config de sizing adaptatif (F11)")
    if d.n_montees == 0:
        _echec("le diagnostic AIMD n'est pas remonté jusqu'à la Distribution de la méthode A")
    print(" 19. F11 EXÉCUTABLE : B refuse le sizing adaptatif, A l'accepte  ✓")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DES DEUX MÉTHODES DE P(funded) — tâche 4.7 (F8-F13, H11)")
    print("=" * 78 + "\n")

    print("  ── la séquence de trades, et la non-régression exigée par D3 ──")
    check_1_non_regression_contre_le_simulateur()
    check_2_rendement_independant_du_sizing()
    check_3_drapeau_de_nouvelle_journee()
    check_4_un_bust_ninterrompt_pas_la_sequence()

    print("\n  ── F9 : la taille de bloc ──")
    check_5_plus_longue_serie_perdante()
    check_6_bloc_trop_court_refuse()
    check_7_bloc_plus_grand_que_la_serie_refuse()

    print("\n  ── F13 : le budget borné, et le clustering ──")
    check_8_budget_borne_et_censure()
    check_9_deterministe_a_seed_fixe()
    check_10_seed_et_n_chemins_obligatoires()
    check_11_le_clustering_journalier_est_preserve()

    print("\n  ── méthode A ──")
    check_12_un_chemin_par_depart()
    check_13_chemins_effectifs_inferieurs_aux_chemins()
    check_14_pas_invalide_refuse()

    print("\n  ── F8, F10 : la comparaison ──")
    check_15_conservative_est_le_minimum_jamais_la_moyenne()
    check_16_conservative_symetrique()
    check_17_ordre_de_lecture_de_lecart()
    check_18_b_sous_a_est_signale_comme_serieux()
    check_19_F11_la_methode_B_REFUSE_un_sizing_adaptatif()
    check_20_l_agregat_des_resultats_bruts_EST_la_methode_A()
    check_93_les_gardes_des_METHODES_mordent()

    print("\n" + "=" * 78)
    print("LES 20 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_les_gardes_des_METHODES_mordent() -> None:
    """
    Trois gardes qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    ⚠️ Ce sont les deux voies qui produisent le `P(funded)` lui-même. Une garde molle ici et le
    chiffre sort quand même — sur des trades qui ne correspondent à aucune barre, ou sur un
    bootstrap dont les blocs n'ont pas de sens.
    """
    barres = panel([GAGNE, PERD])
    # ① une décision dont la barre d'entrée n'existe pas dans le panel
    _leve(
        lambda: resoudre_trades([dec(9_999, LONG)], barres),
        ValueError,
        "barre d'entrée absente du panel",
        "n'existe pas dans le panel",
    )
    # ② un budget de journées que la série ne peut pas offrir
    bonnes = [dec(0, LONG)]
    _leve(
        lambda: methode_a(bonnes, barres, budget_journees=10_000),
        ValueError,
        "budget de journées impossible",
        "budget",
    )
    # ③ une taille de bloc nulle : un bootstrap par blocs sans bloc.
    # ⚠️⚠️ **Deux pièges ici, trouvés en écrivant la preuve le 2026-09-21.**
    # ① `contient="taille_bloc"` ne prouvait RIEN : la garde suivante (F9, bloc plus court que la
    #    plus longue série perdante) porte le même mot. On exige le message exact de CETTE garde.
    # ② Le décor doit contenir une **perte** : sans elle `plus_longue_serie_perdante` vaut 0, la
    #    garde F9 ne rattrape pas, et le bootstrap part en **boucle infinie** — `while
    #    len(sequence) < budget` avec des blocs de taille 0 n'ajoute jamais rien. La mutation qui
    #    désarme cette garde se serait alors « détectée » par un timeout de 15 minutes.
    avec_perte = [dec(0, LONG), dec(3, LONG)]
    trades = resoudre_trades(avec_perte, barres)
    if not any(t.rendement < 0 for t in trades):
        _echec("anti-vacuité : le décor ne contient aucune perte, la garde F9 ne rattraperait pas")
    _leve(
        lambda: methode_b(trades, 0.01, taille_bloc=0, n_chemins=10, seed=0),
        ValueError,
        "taille_bloc nulle",
        "doit valoir au moins 1",
    )
    # anti-vacuité : les mêmes appels, bien formés, produisent un résultat
    if not resoudre_trades(bonnes, barres):
        _echec("anti-vacuité : aucun trade résolu sur des décisions valides")
    print("  93. gardes des méthodes : barre absente, budget impossible, taille_bloc nulle  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
