"""
Preuve de la métrique de payoff — tâche 4.1.

Vingt-deux vérifications. Ce module ne touche aucune donnée de marché : tout se prouve sur des
trajectoires d'équity **fabriquées**, où l'on sait à l'avance ce qui doit arriver et quand. C'est
la raison d'être de la séparation 4.1 / 4.2 — sur une vraie série on ne pourrait que constater un
`P(funded)` plausible.

Ce qui est réellement en jeu ici, c'est l'**ordre** des événements. Un `P(funded)` faux ne se
signale jamais : il sort un nombre entre 0 et 1, d'allure raisonnable. Les preuves les plus
importantes de ce fichier sont donc celles qui fabriquent une collision — cible ET plancher sur
la même observation — et vérifient que le compte meurt.

Lancer :  .venv/bin/python juge/sanity_check_regles_ftmo.py
"""

from __future__ import annotations

import sys

from regles_ftmo import (
    Issue,
    MotifBust,
    Phase,
    ReglesFTMO,
    SuiviChallenge,
)

R = ReglesFTMO()


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """
    Vérifie qu'un appel lève — et, avec `contient`, que c'est **pour la bonne raison**.

    ⚠️ Sans le contrôle du message, ces preuves peuvent passer pour la mauvaise raison :
    `SuiviChallenge` a plusieurs gardes qui lèvent toutes `RuntimeError` (« aucune journée »,
    « déjà terminé ») et `ReglesFTMO.__post_init__` plusieurs `ValueError` distinctes — l'une
    peut masquer l'autre. Le piège a été rencontré pour de vrai le 2026-08-09 (preuve de 4.6).
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
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__}")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def _demarre(regles: ReglesFTMO | None = None) -> SuiviChallenge:
    """Un challenge prêt à observer, journée 1 ouverte au capital initial."""
    s = SuiviChallenge(regles)
    s.nouvelle_journee(None, s.equity_depart_phase)
    return s


def _quatre_journees(s: SuiviChallenge, equity: float) -> None:
    """Consomme les 4 journées minimum sans bouger l'équity."""
    for _ in range(R.jours_minimum):
        s.nouvelle_journee(None, equity)
        s.marquer_journee_tradee()
        s.observer(equity=equity, balance=equity)


# ── Le barème ────────────────────────────────────────────────────────────────────────────
def check_1_bareme_ftmo_50k() -> None:
    if R.capital_de_depart(Phase.UN) != 50_000.0:
        _echec("le capital de départ de la phase 1 n'est pas 50 000")
    if R.cible(Phase.UN) != 55_000.0:
        _echec(f"cible phase 1 = {R.cible(Phase.UN)}, attendu 55 000 (+10 %)")
    if R.cible(Phase.DEUX) != 52_500.0:
        _echec(f"cible phase 2 = {R.cible(Phase.DEUX)}, attendu 52 500 (+5 %)")
    if R.plancher_total(Phase.UN) != 45_000.0:
        _echec(f"plancher total = {R.plancher_total(Phase.UN)}, attendu 45 000 (-10 %)")
    print("  1. barème FTMO 2-step 50k : cibles 55 000 / 52 500, plancher total 45 000  ✓")


def check_2_phase2_repart_du_capital_initial() -> None:
    """Le gain de la phase 1 n'est PAS un coussin : la phase 2 est un compte neuf."""
    if R.cible(Phase.DEUX) != R.capital_initial * 1.05:
        _echec("la cible de la phase 2 ne se calcule pas sur le capital INITIAL")
    if R.plancher_total(Phase.DEUX) != R.plancher_total(Phase.UN):
        _echec("le plancher total de la phase 2 devrait repartir du même capital")
    print("  2. la phase 2 repart du capital initial, pas de l'équity de fin de phase 1  ✓")


def check_3_plancher_total_statique() -> None:
    """Le piège classique : faire suivre le plancher total à l'équity."""
    niveaux = {R.plancher_total(Phase.UN) for _ in range(3)}
    if len(niveaux) != 1 or niveaux.pop() != 45_000.0:
        _echec("le plancher total dépend de quelque chose : il doit être STATIQUE")
    s = _demarre()
    s.observer(equity=54_000.0, balance=54_000.0)  # grosse bonne passe
    if R.plancher_total(s.phase) != 45_000.0:
        _echec("le plancher total a bougé après un gain — il ne bouge JAMAIS")
    print("  3. plancher total statique : inchangé après une hausse de +8 %  ✓")


def check_4_plancher_journalier_montant_fixe() -> None:
    """
    Règle FTMO réelle (alignée le 2026-08-10) : le montant autorisé par jour est **fixe** —
    5 % du capital INITIAL, soit 2 500 $ — et c'est l'ancre (la balance du jour) qui bouge.

    Les deux valeurs testées hors de 50 000 discriminent l'ancienne lecture (-5 % de la
    balance du jour) : elle donnait 49 400 à 52k (2 600 $ de marge) et 45 600 à 48k (2 400 $).
    """
    if R.plancher_journalier(50_000.0) != 47_500.0:
        _echec("plancher journalier != balance d'ouverture - 2 500")
    if R.plancher_journalier(52_000.0) != 49_500.0:
        _echec("au-dessus de 50k le montant doit rester 2 500 $ (l'ancienne lecture donnait 2 600)")
    if R.plancher_journalier(48_000.0) != 45_500.0:
        _echec(
            "en dessous de 50k le montant doit rester 2 500 $ (l'ancienne lecture donnait 2 400)"
        )
    print("  4. plancher journalier = balance d'ouverture - 5 % du capital INITIAL (fixe)  ✓")


def check_5_les_deux_planchers_se_croisent() -> None:
    """
    Le scénario que le plan (4.5) appelle « bust total redevenu plus contraignant ».

    Tant que la balance d'ouverture est haute, le journalier mord en premier. Sous 47 500
    pile (balance - 2 500 < 45 000), c'est le total. Un juge qui ne teste qu'un seul des deux
    se trompe d'un côté ou de l'autre.
    """
    s = _demarre()
    if s.plancher_effectif() != 47_500.0:
        _echec("au départ, c'est le plancher JOURNALIER qui doit mordre")
    s.nouvelle_journee(None, 46_000.0)  # après une mauvaise passe
    if s.plancher_effectif() != 45_000.0:
        _echec("après une baisse, c'est le plancher TOTAL qui doit redevenir contraignant")
    print("  5. les deux planchers se croisent : journalier puis total, sans hypothèse  ✓")


# ── L'ordre, c'est-à-dire le first-passage ───────────────────────────────────────────────
def check_6_le_plancher_bat_la_cible() -> None:
    """
    **La preuve centrale.** Une observation qui franchit la cible ET le plancher est un BUST.

    Impossible en pratique sur une équity seule, mais parfaitement possible avec la convention
    par défaut : balance au-dessus de la cible (trades clôturés gagnants) et équity sous le
    plancher (une position flottante catastrophique encore ouverte). Le compte est fermé.
    """
    s = _demarre()
    _quatre_journees(s, 50_000.0)
    issue = s.observer(equity=44_000.0, balance=56_000.0)
    if issue is not Issue.BUST:
        _echec(
            f"cible ET plancher franchis : attendu BUST, reçu {issue.value} — le juge est optimiste"
        )
    print("  6. cible et plancher sur la MÊME observation -> BUST (l'ordre est la métrique)  ✓")


def check_7_motif_du_bust_le_plus_haut_mord() -> None:
    """Mouvement monotone : une équity qui descend croise d'abord le plancher le plus haut."""
    s = _demarre()
    s.observer(equity=47_400.0, balance=47_400.0)  # sous 47 500, au-dessus de 45 000
    if s.motif_bust is not MotifBust.PLANCHER_JOURNALIER:
        _echec(f"motif attendu journalier, reçu {s.motif_bust}")

    s2 = SuiviChallenge()
    s2.nouvelle_journee(None, 46_000.0)  # journalier à 43 500, sous le total
    s2.observer(equity=44_900.0, balance=44_900.0)
    if s2.motif_bust is not MotifBust.PLANCHER_TOTAL:
        _echec(f"motif attendu total, reçu {s2.motif_bust}")
    print("  7. le motif du bust est le plancher le plus HAUT — celui croisé en premier  ✓")


def check_8_cible_sur_la_balance_pas_sur_le_flottant() -> None:
    """Un profit flottant qu'on rend n'a jamais existé."""
    s = _demarre()
    _quatre_journees(s, 50_000.0)
    s.observer(equity=56_000.0, balance=50_000.0)  # +6k flottant, rien de clôturé
    if s.phase is not Phase.UN:
        _echec(
            "la phase a été validée sur un profit FLOTTANT — elle ne doit compter que le réalisé"
        )

    s.observer(equity=55_000.0, balance=55_000.0)
    if s.phase is not Phase.DEUX:
        _echec("la phase n'a pas été validée alors que la BALANCE atteint la cible")
    print("  8. la cible se juge sur la balance, les planchers sur l'équity (asymétrie voulue)  ✓")


def check_9_quatre_journees_minimum() -> None:
    s = _demarre()
    s.marquer_journee_tradee()
    s.observer(equity=56_000.0, balance=56_000.0)  # cible atteinte dès le 1er jour
    if s.phase is not Phase.UN:
        _echec("phase validée avec 1 seule journée tradée — le minimum de 4 est ignoré")
    if s.jours_tradees != 1:
        _echec(f"jours_tradees = {s.jours_tradees}, attendu 1")
    print("  9. cible atteinte au 1er jour : la phase N'EST PAS validée (4 journées minimum)  ✓")


def check_10_cible_doit_etre_tenue_au_moment_des_4_jours() -> None:
    """Lecture pessimiste : toucher la cible puis la rendre ne vaut rien."""
    s = _demarre()
    s.marquer_journee_tradee()
    s.observer(equity=56_000.0, balance=56_000.0)  # cible touchée, jour 1
    for _ in range(3):
        s.nouvelle_journee(None, 51_000.0)
        s.marquer_journee_tradee()
        s.observer(equity=51_000.0, balance=51_000.0)  # rendue
    if s.phase is not Phase.UN:
        _echec("phase validée alors que la cible avait été RENDUE avant le 4e jour")
    if s.jours_tradees != 4:
        _echec(f"jours_tradees = {s.jours_tradees}, attendu 4")
    print(" 10. cible touchée puis rendue : rien n'est acquis, il faut la tenir au 4e jour  ✓")


def check_11_lecture_alternative_cible_acquise() -> None:
    """L'autre lecture existe et fait bien quelque chose de DIFFÉRENT — sinon le champ ment."""
    regles = ReglesFTMO(cible_et_jours_simultanes=False)
    s = _demarre(regles)
    s.marquer_journee_tradee()
    s.observer(equity=56_000.0, balance=56_000.0)
    for _ in range(3):
        s.nouvelle_journee(None, 51_000.0)
        s.marquer_journee_tradee()
        s.observer(equity=51_000.0, balance=51_000.0)
    if s.phase is not Phase.DEUX:
        _echec("en mode « cible acquise », la phase devait être validée au 4e jour")
    print(
        " 11. le mode « cible acquise » change réellement le verdict (le champ n'est pas décoratif)  ✓"
    )


# ── Les garde-fous ───────────────────────────────────────────────────────────────────────
def check_12_observer_sans_journee_leve() -> None:
    """Le garde-fou qui rend impossible le bug de la journée calendaire."""
    s = SuiviChallenge()
    _leve(
        lambda: s.observer(equity=50_000.0, balance=50_000.0),
        RuntimeError,
        "observer sans journée",
        contient="aucune journée",
    )
    _leve(
        lambda: s.plancher_effectif(),
        RuntimeError,
        "plancher_effectif sans journée",
        contient="aucune journée",
    )
    _leve(
        lambda: s.marquer_journee_tradee(),
        RuntimeError,
        "marquer_journee_tradee sans journée",
        contient="aucune journée",
    )
    print(" 12. observer sans avoir annoncé de journée FTMO lève une erreur  ✓")


def check_13_observer_apres_la_fin_leve() -> None:
    s = _demarre()
    s.observer(equity=40_000.0, balance=40_000.0)
    if not s.termine:
        _echec("le challenge devrait être terminé")
    _leve(
        lambda: s.observer(equity=50_000.0, balance=50_000.0),
        RuntimeError,
        "observer après la fin",
        contient="déjà terminé",
    )
    _leve(
        lambda: s.nouvelle_journee(None, 50_000.0),
        RuntimeError,
        "nouvelle journée après la fin",
        contient="déjà terminé",
    )
    print(" 13. observer ou rouvrir une journée sur un compte fermé lève une erreur  ✓")


def check_14_passage_en_phase_2_invalide_la_journee() -> None:
    """Compte neuf : impossible d'hériter du plancher journalier de la phase 1."""
    s = _demarre()
    _quatre_journees(s, 50_000.0)
    s.observer(equity=55_000.0, balance=55_000.0)
    if s.phase is not Phase.DEUX:
        _echec("la phase 1 aurait dû être validée")
    if s.jours_tradees != 0:
        _echec("le compteur de journées n'a pas été remis à zéro pour la phase 2")
    # Le fragment discrimine : la journée doit être INVALIDÉE (« aucune journée »), pas le
    # challenge terminé (« déjà terminé ») — un FUNDED prématuré lèverait la même RuntimeError.
    _leve(
        lambda: s.observer(equity=50_000.0, balance=50_000.0),
        RuntimeError,
        "observer après passage en phase 2",
        contient="aucune journée",
    )
    print(" 14. le passage en phase 2 invalide la journée : le compte est NEUF  ✓")


def check_15_config_incoherente_refusee() -> None:
    _leve(
        lambda: ReglesFTMO(capital_initial=0.0),
        ValueError,
        "capital nul",
        contient="capital_initial",
    )
    _leve(
        lambda: ReglesFTMO(cible_phase1_pct=10.0),
        ValueError,
        "pourcentage donné en 10 au lieu de 0,10",
        contient="fraction",
    )
    _leve(
        lambda: ReglesFTMO(jours_minimum=-1),
        ValueError,
        "jours_minimum négatif",
        contient="jours_minimum",
    )
    _leve(
        lambda: ReglesFTMO(plancher_journalier_pct=0.10, plancher_total_pct=0.05),
        ValueError,
        "plancher journalier >= plancher total (rendrait le total inatteignable)",
        contient="inatteignable",
    )
    print(" 15. une configuration incohérente est REFUSÉE, pas silencieusement acceptée  ✓")


def check_21_capital_initial_non_fini_refuse() -> None:
    """
    **Piège n°8, audit du 2026-08-18.** La garde s'écrivait `if capital_initial <= 0`, et
    `nan <= 0` vaut **False** : un `NaN` passait.

    Ce que ça produisait, et c'est total : `cible`, `plancher_total` et `plancher_journalier`
    devenaient tous `NaN`. Or **toute comparaison avec `NaN` est fausse**, y compris le `<=` du
    plancher dans `observer` — donc **plus aucun bust n'aurait jamais été détecté**, ni aucune
    cible atteinte. Tous les chemins seraient restés `EN_COURS`, et `P(funded)` se serait calculé
    sur zéro chemin conclu (F19).

    La correction tient dans la FORME : `not x > 0` refuse le `NaN`, `x <= 0` l'accepte.
    """
    _leve(
        lambda: ReglesFTMO(capital_initial=float("nan")),
        ValueError,
        "un capital initial NaN",
        contient="fini",
    )
    _leve(
        lambda: ReglesFTMO(capital_initial=float("-inf")),
        ValueError,
        "un capital initial -inf",
        contient="fini",
    )
    # La démonstration du mécanisme, pour que la propriété ne soit pas seulement affirmée.
    nan = float("nan")
    if nan <= 0:
        _echec("`nan <= 0` est vrai ici : la démonstration du piège n'a plus d'objet")
    if not nan > 0:
        pass
    else:
        _echec("`not nan > 0` est faux : la forme corrigée ne refuserait pas le NaN")
    print(" 21. un capital_initial NaN ou infini est REFUSÉ (piège n°8, forme de la borne)  ✓")


def check_16_challenge_funded_de_bout_en_bout() -> None:
    """Le chemin complet — c'est lui qui produira le numérateur de P(funded)."""
    s = _demarre()
    _quatre_journees(s, 50_000.0)
    s.observer(equity=55_000.0, balance=55_000.0)
    if s.phase is not Phase.DEUX:
        _echec("phase 1 non validée")

    s.nouvelle_journee(None, 50_000.0)  # compte neuf, capital initial
    for _ in range(R.jours_minimum):
        s.nouvelle_journee(None, 50_000.0)
        s.marquer_journee_tradee()
        s.observer(equity=50_000.0, balance=50_000.0)
    issue = s.observer(equity=52_500.0, balance=52_500.0)

    if issue is not Issue.FUNDED:
        _echec(f"attendu FUNDED, reçu {issue.value}")
    if not s.termine or s.phase is not Phase.TERMINE:
        _echec("un compte financé doit être terminé")
    print(" 16. challenge complet : phase 1 puis phase 2 -> FUNDED  ✓")


def check_17_plancher_touche_exactement_fait_buster() -> None:
    """
    Le cas limite : équity **égale** au plancher, au centime près.

    FTMO ferme un compte qui touche la limite, il ne demande pas de la dépasser. Un `<` au lieu
    d'un `<=` ne se verrait sur aucun jeu de données réel — les prix ne tombent jamais pile — et
    resterait donc invisible jusqu'au jour où il compterait.
    """
    s = _demarre()
    issue = s.observer(equity=47_500.0, balance=47_500.0)  # le plancher journalier PILE
    if issue is not Issue.BUST:
        _echec("équity exactement AU plancher : attendu BUST, le compte est touché")

    s2 = SuiviChallenge()
    s2.nouvelle_journee(None, 45_000.0)
    if s2.observer(equity=45_000.0, balance=45_000.0) is not Issue.BUST:
        _echec("équity exactement au plancher TOTAL : attendu BUST")
    print(" 17. un plancher touché EXACTEMENT fait buster (limite incluse)  ✓")


def check_18_cible_atteinte_exactement_valide() -> None:
    """
    L'autre bord, et il est piégeux : `50_000 * 1.10` vaut `55000.00000000001` en flottant.

    Avec cette formulation, une balance de 55 000,00 pile ne validerait PAS la phase. Décalage
    de 1e-11, aucun symptôme, et un `P(funded)` légèrement sous-estimé pour toujours.
    """
    if R.cible(Phase.UN) != 55_000.0:
        _echec(f"cible non exacte en flottant : {R.cible(Phase.UN)!r} — utiliser base + base * pct")
    s = _demarre()
    _quatre_journees(s, 50_000.0)
    s.observer(equity=55_000.0, balance=55_000.0)  # la cible PILE
    if s.phase is not Phase.DEUX:
        _echec("balance exactement À la cible : la phase devait être validée")
    print(" 18. la cible atteinte EXACTEMENT valide la phase (pas de dérive flottante)  ✓")


def check_19_journee_tradee_idempotente() -> None:
    """Dix trades dans la même journée FTMO ne font qu'UNE journée de trading."""
    s = _demarre()
    for _ in range(10):
        s.marquer_journee_tradee()
    if s.jours_tradees != 1:
        _echec(f"10 trades le même jour comptés {s.jours_tradees} journées, attendu 1")
    s.nouvelle_journee(None, 50_000.0)
    s.marquer_journee_tradee()
    if s.jours_tradees != 2:
        _echec(f"après une nouvelle journée tradée, attendu 2, reçu {s.jours_tradees}")
    print(" 19. le compteur de journées est idempotent : 10 trades le même jour = 1 journée  ✓")


def check_20_egalite_exacte_des_deux_planchers() -> None:
    """
    Le point de croisement, là où les **deux** planchers valent exactement la même chose.

    Trouvé en cherchant à muter le départage `plancher_j >= plancher_t` : la mutation `>` passait
    inaperçue, donc aucune preuve n'exerçait ce cas. Sous la règle réelle (montant fixe,
    2026-08-10), le point d'égalité est même exact en décimal : à 47 500 de balance d'ouverture,
    le journalier tombe pile sur 45 000.

    Convention retenue : **à égalité, c'est le JOURNALIER**. Il se referme le plus vite, et c'est
    donc lui que l'**airbag** doit viser en priorité (H16). *(Citait H2 jusqu'au 2026-08-15,
    quand le déclencheur du sizing était encore la distance à la barrière.)*
    """
    e = R.plancher_total(Phase.UN) + R.capital_initial * R.plancher_journalier_pct
    if R.plancher_journalier(e) != R.plancher_total(Phase.UN):
        _echec(f"le cas d'égalité n'est pas atteint avec e={e!r} — preuve à recalibrer")

    s = SuiviChallenge()
    s.nouvelle_journee(None, e)
    if s.plancher_effectif() != R.plancher_total(Phase.UN):
        _echec("à égalité, le plancher effectif devrait valoir la valeur commune")
    s.observer(equity=44_000.0, balance=44_000.0)
    if s.motif_bust is not MotifBust.PLANCHER_JOURNALIER:
        _echec(f"à égalité stricte, motif attendu JOURNALIER, reçu {s.motif_bust}")
    print(" 20. à égalité exacte des deux planchers, c'est le journalier qui est rapporté  ✓")


def check_22_la_garde_de_progression_de_journee() -> None:
    """
    **Tâche 4.12 — la dette ouverte le 2026-08-16, réglée avant la phase 6.**

    `nouvelle_journee` croyait l'appelant sur parole. Appelée à chaque **barre** au lieu de chaque
    **frontière de journée**, le plancher journalier se reposerait à chaque barre et **suivrait la
    balance** : il ne mordrait pratiquement plus jamais, presque tous les busts journaliers
    disparaîtraient, et **`P(funded)` serait gonflé sans le moindre symptôme**.

    C'est exactement le trou que 5.3 avait fermé côté `SuiviAIMD`. L'asymétrie était connue et
    inscrite comme dette : le fait que le sizing soit protégé rendait l'oubli **plus** probable,
    puisqu'on aurait lu « la garde existe » et conclu que le juge l'était aussi.

    On mesure ici les trois comportements, plus le cas où la garde doit **se taire**.
    """
    from datetime import date, timedelta

    lundi = date(2021, 1, 25)
    s = SuiviChallenge(R)
    s.nouvelle_journee(lundi, 50_000.0)

    _leve(
        lambda: s.nouvelle_journee(lundi, 50_000.0),
        RuntimeError,
        "la même journée FTMO annoncée deux fois",
        contient="strictement avancer",
    )
    _leve(
        lambda: s.nouvelle_journee(lundi - timedelta(days=1), 50_000.0),
        RuntimeError,
        "une journée FTMO qui recule",
        contient="strictement avancer",
    )
    s.nouvelle_journee(lundi + timedelta(days=1), 50_000.0)  # le cas légitime passe toujours

    # ⚠️ `jour=None` désactive la garde, et c'est un choix ÉCRIT (methode_b n'a pas de dates).
    libre = SuiviChallenge(R)
    for _ in range(3):
        libre.nouvelle_journee(None, 50_000.0)

    # ⚠️ Et elle doit se TAIRE au passage de phase : le compte est neuf, le simulateur redéclare
    # la journée EN COURS avec le capital de départ de la phase 2 (F17). Une garde qui refuserait
    # ce cas rendrait le passage de phase impossible.
    phase = SuiviChallenge(R)
    phase.nouvelle_journee(lundi, R.capital_initial)
    phase.marquer_journee_tradee()
    for _ in range(R.jours_minimum - 1):
        lundi += timedelta(days=1)
        phase.nouvelle_journee(lundi, R.capital_initial)
        phase.marquer_journee_tradee()
    phase.observer(equity=R.cible(Phase.UN), balance=R.cible(Phase.UN))
    if phase.phase is not Phase.DEUX:
        _echec(f"l'essai ne discrimine pas : phase {phase.phase}, attendu la phase 2")
    phase.nouvelle_journee(lundi, R.capital_initial)  # MÊME journée, et ce doit être accepté
    print(" 22. la journée doit STRICTEMENT avancer, sauf jour=None et passage de phase  ✓")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DE LA MÉTRIQUE DE PAYOFF — tâche 4.1 (F0, F1, F2, F3)")
    print("=" * 78 + "\n")

    check_1_bareme_ftmo_50k()
    check_2_phase2_repart_du_capital_initial()
    check_3_plancher_total_statique()
    check_4_plancher_journalier_montant_fixe()
    check_5_les_deux_planchers_se_croisent()
    print()
    check_6_le_plancher_bat_la_cible()
    check_7_motif_du_bust_le_plus_haut_mord()
    check_8_cible_sur_la_balance_pas_sur_le_flottant()
    check_9_quatre_journees_minimum()
    check_10_cible_doit_etre_tenue_au_moment_des_4_jours()
    check_11_lecture_alternative_cible_acquise()
    print()
    check_12_observer_sans_journee_leve()
    check_13_observer_apres_la_fin_leve()
    check_14_passage_en_phase_2_invalide_la_journee()
    check_15_config_incoherente_refusee()
    check_16_challenge_funded_de_bout_en_bout()
    print()
    check_17_plancher_touche_exactement_fait_buster()
    check_18_cible_atteinte_exactement_valide()
    check_19_journee_tradee_idempotente()
    check_20_egalite_exacte_des_deux_planchers()
    check_21_capital_initial_non_fini_refuse()
    check_22_la_garde_de_progression_de_journee()
    check_93_une_phase_TERMINEE_na_pas_de_capital()

    print("\n" + "=" * 78)
    print("LES 23 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_une_phase_TERMINEE_na_pas_de_capital() -> None:
    """
    La garde qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    Demander le capital de départ d'une phase terminée n'a pas de sens : rendre un nombre
    « par défaut » ferait calculer une cible et des planchers sur un compte qui n'existe plus.
    """
    _leve(
        lambda: ReglesFTMO().capital_de_depart(Phase.TERMINE),
        ValueError,
        "capital de départ d'une phase terminée",
        "terminée",
    )
    # anti-vacuité : les deux phases vivantes en ont un
    if ReglesFTMO().capital_de_depart(Phase.UN) <= 0:
        _echec("anti-vacuité : la phase 1 n'a pas de capital de départ")
    print("  93. une phase TERMINÉE n'a pas de capital de départ, et le dit  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
