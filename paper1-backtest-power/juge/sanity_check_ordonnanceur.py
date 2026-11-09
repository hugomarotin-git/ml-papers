"""
Preuve de l'ordonnanceur d'occupation — tâche 4.10, décisions H11 et F14.

Quatorze vérifications. Ce module décide **quels trades existent** : une erreur ici ne produit
pas un plantage, elle produit une cadence différente — donc une autre colonne du twin-null à
laquelle se comparer (G3), donc un verdict faussé sans le moindre symptôme.

Deux propriétés y sont plus importantes que les autres :

- **le départage par le score** (F14, checks 5 à 7). Retomber sur l'ordre d'arrivée jetterait
  l'information de confiance du modèle et rendrait le résultat dépendant d'un tri en amont ;
- **l'indépendance au sizing** (H11, check 12). Si l'ordonnancement changeait avec la taille des
  positions, comparer deux configurations d'AIMD reviendrait à comparer deux systèmes
  différents, et toute la phase 5 mesurerait du vide.

Lancer :  .venv/bin/python juge/sanity_check_ordonnanceur.py
"""

from __future__ import annotations

import sys

from methodes_pfunded import resoudre_trades
from ordonnanceur import ordonnancer
from sanity_check_simulateur import GAGNE, PERD, PLAT, dt, panel
from simulateur import LONG, Decision, horizon_global


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


def cand(barre: int, score: float = float("nan")) -> Decision:
    """Un candidat entrant à la barre donnée, avec un score optionnel."""
    return Decision(dt(barre), LONG, 100.0, 101.0, 99.0, score=score)


def barres_plates(n: int):
    """Un panel où rien ne touche jamais de barrière : chaque trade vit l'horizon entier."""
    return panel([[PLAT] * n])


# ── L'occupation ─────────────────────────────────────────────────────────────────────────
def check_1_un_seul_trade_a_la_fois() -> None:
    """H11 — le compte ne porte qu'une position."""
    b = barres_plates(30)
    o = ordonnancer([cand(i) for i in range(0, 12)], b)
    for a, suivant in zip(o.retenus, o.retenus[1:], strict=False):
        if suivant.dt_entree <= a.dt_entree:
            _echec("deux trades retenus se chevauchent ou partagent un instant")
    if len(o.retenus) < 2:
        _echec("le panel est assez long pour plusieurs trades successifs")
    print("  1. un seul trade à la fois : aucun chevauchement dans les retenus  ✓")


def check_2_occupation_dure_exactement_la_vie_du_trade() -> None:
    """
    Sur une série plate, chaque trade vit l'horizon entier. Le candidat suivant ne peut donc
    entrer qu'à `entrée + horizon_global()`. Un décalage d'une barre ici change la cadence de tout
    le système — et la cadence décide de la colonne du null (G3).
    """
    b = barres_plates(40)
    o = ordonnancer([cand(i) for i in range(0, 25)], b)
    ecarts = {
        int((s.dt_entree - a.dt_entree).total_seconds() // 3600)
        for a, s in zip(o.retenus, o.retenus[1:], strict=False)
    }
    if ecarts != {horizon_global()}:
        _echec(
            f"écarts entre trades retenus : {sorted(ecarts)} heures, attendu exactement "
            f"{horizon_global()} — le compte se libère à la barre qui suit la sortie"
        )
    print(f"  2. l'occupation dure exactement la vie du trade ({horizon_global()} barres ici)  ✓")


def check_3_un_trade_court_libere_plus_tot() -> None:
    """
    Un trade qui touche son stop à la 2ᵉ barre libère le compte bien avant l'horizon. Si
    l'ordonnanceur bloquait toujours l'horizon complet, il **sous-estimerait la cadence** — donc
    comparerait le système à une colonne du null trop clémente.
    """
    # Jour 1 : le trade entré à la barre 1 est stoppé à la barre 2. Puis tout est plat.
    b = panel([[PLAT, PLAT, (100.0, 100.0, 99.0, 99.0)] + [PLAT] * 10])
    resolus = resoudre_trades([cand(1)], b)
    if resolus[0].duree_barres >= horizon_global():
        _echec(f"le trade devait se résoudre tôt, durée {resolus[0].duree_barres}")

    o = ordonnancer([cand(1), cand(3)], b)
    if len(o.retenus) != 2:
        _echec(
            f"{len(o.retenus)} retenus : le premier trade se résout à la barre 2, le candidat "
            "de la barre 3 doit donc pouvoir entrer"
        )
    print("  3. un trade résolu tôt libère le compte tôt : la cadence ne se perd pas  ✓")


def check_4_candidats_pendant_occupation_comptes() -> None:
    b = barres_plates(20)
    o = ordonnancer([cand(i) for i in range(0, 6)], b)
    if o.n_rejetes_occupation != 5:
        _echec(f"{o.n_rejetes_occupation} rejets pour occupation, attendu 5")
    if len(o.retenus) + o.n_rejetes_occupation + o.n_rejetes_departage != o.n_candidats:
        _echec("les comptages ne couvrent pas l'ensemble des candidats")
    print("  4. les candidats écartés pour occupation sont COMPTÉS, pas perdus  ✓")


# ── F14 : le départage par le score ──────────────────────────────────────────────────────
def check_5_le_mieux_note_gagne() -> None:
    """**F14.** Trois candidats au même instant : c'est le score qui tranche, pas la position."""
    b = barres_plates(20)
    o = ordonnancer([cand(1, score=0.30), cand(1, score=0.90), cand(1, score=0.55)], b)
    if len(o.retenus) != 1:
        _echec(f"{len(o.retenus)} retenus pour un instant unique, attendu 1")
    if o.retenus[0].score != 0.90:
        _echec(f"score retenu {o.retenus[0].score}, attendu 0,90 — le mieux noté doit gagner")
    if o.n_rejetes_departage != 2:
        _echec(f"{o.n_rejetes_departage} rejets par départage, attendu 2")
    print("  5. à instant égal, le MIEUX NOTÉ est retenu (F14)  ✓")


def check_6_lordre_de_la_liste_ne_change_rien() -> None:
    """
    Le test qui distingue vraiment F14 d'un glouton : mêmes candidats, ordre inversé dans la
    liste, même gagnant. Un filtre par ordre d'arrivée échouerait ici.
    """
    b = barres_plates(20)
    dans_lordre = ordonnancer([cand(1, 0.30), cand(1, 0.90), cand(1, 0.55)], b)
    a_lenvers = ordonnancer([cand(1, 0.55), cand(1, 0.90), cand(1, 0.30)], b)
    if dans_lordre.retenus[0].score != a_lenvers.retenus[0].score:
        _echec(
            "le gagnant dépend de l'ordre de la liste : c'est un glouton par ordre d'arrivée, "
            "pas le départage par score qu'impose F14"
        )
    print("  6. le gagnant ne dépend PAS de l'ordre de la liste  ✓")


def check_7_departage_sans_score_leve() -> None:
    """
    **Le corollaire de F14, asserté.** Sans score, l'ordonnanceur retomberait en silence sur
    l'ordre d'arrivée — et personne ne le verrait, puisque le résultat resterait plausible.
    """
    b = barres_plates(20)
    _leve(
        lambda: ordonnancer([cand(1), cand(1)], b),
        ValueError,
        "départage sans score",
        contient="score",
    )
    # Un seul candidat à l'instant : aucun départage nécessaire, donc aucun score requis.
    o = ordonnancer([cand(1)], b)
    if len(o.retenus) != 1:
        _echec("un candidat seul ne nécessite aucun score et doit être retenu")
    print("  7. un départage sans score LÈVE ; un candidat seul n'en a pas besoin  ✓")


def check_8_score_partiel_leve_aussi() -> None:
    """Un seul score manquant suffit à rendre le départage arbitraire."""
    b = barres_plates(20)
    _leve(
        lambda: ordonnancer([cand(1, 0.8), cand(1)], b),
        ValueError,
        "un score manquant sur deux",
        contient="score",
    )
    print("  8. un seul score manquant parmi les ex aequo suffit à LEVER  ✓")


# ── Les garde-fous et l'invariant central ────────────────────────────────────────────────
def check_9_candidats_non_tries_refuses() -> None:
    b = barres_plates(20)
    _leve(
        lambda: ordonnancer([cand(5), cand(1)], b),
        ValueError,
        "candidats non triés",
        contient="pas triés",
    )
    print("  9. des candidats non triés sont REFUSÉS, jamais retriés en silence  ✓")


def check_10_liste_vide() -> None:
    o = ordonnancer([], barres_plates(10))
    if (len(o.retenus), o.n_candidats, o.taux_de_retention) != (0, 0, 0.0):
        _echec("une liste vide doit donner un ordonnancement vide et neutre")
    print(" 10. une liste vide donne un ordonnancement vide, sans division par zéro  ✓")


def check_11_retenus_sont_un_sous_ensemble_ordonne() -> None:
    """Rien n'est inventé : les retenus sortent des candidats, dans le même ordre temporel."""
    b = barres_plates(30)
    candidats = [cand(i, score=i / 100) for i in range(0, 15)]
    o = ordonnancer(candidats, b)
    if any(r not in candidats for r in o.retenus):
        _echec("un trade retenu ne figure pas dans les candidats")
    dts = [r.dt_entree for r in o.retenus]
    if dts != sorted(dts):
        _echec("les retenus ne sont pas dans l'ordre temporel")
    print(" 11. les retenus sont un sous-ensemble des candidats, dans l'ordre  ✓")


def check_12_ordonnancement_independant_du_sizing() -> None:
    """
    **H11, l'invariant central.** L'ordonnanceur ne reçoit aucune notion de taille, et c'est ce
    qui permettra de comparer des configurations d'AIMD sur exactement la même séquence (H8).

    On le vérifie par la seule voie honnête : sa signature ne contient ni capital, ni risque, ni
    règles. S'il fallait les lui passer un jour, ce serait le signe que la sélection dépend du
    sizing — et toute la comparaison de la phase 5 deviendrait bancale.
    """
    import inspect

    params = set(inspect.signature(ordonnancer).parameters)
    interdits = {"risque", "risque_par_trade_pct", "capital", "regles", "balance", "equity"}
    if params & interdits:
        _echec(
            f"`ordonnancer` reçoit {params & interdits} : la sélection dépendrait du sizing, "
            "ce qui viole H11 et rend incomparables deux configurations d'AIMD"
        )
    if params != {"candidats", "barres"}:
        _echec(f"signature inattendue : {params}")
    print(" 12. l'ordonnanceur ignore tout du sizing (H11) — vérifié sur sa signature  ✓")


def check_13_deterministe() -> None:
    b = barres_plates(30)
    candidats = [cand(i, score=(i * 7 % 13) / 13) for i in range(0, 15)]
    a, c = ordonnancer(candidats, b), ordonnancer(candidats, b)
    if [r.dt_entree for r in a.retenus] != [r.dt_entree for r in c.retenus]:
        _echec("deux ordonnancements identiques donnent des séquences différentes")
    print(" 13. l'ordonnancement est déterministe  ✓")


def check_14_cadence_reduite_par_loccupation() -> None:
    """
    Le fait le plus important à retenir de ce module : **la cadence réelle est bien inférieure
    au nombre de candidats.** C'est elle qui décide de la colonne du twin-null (G3), et la
    supposer égale au nombre de candidats comparerait le modèle à une barre calculée pour une
    fréquence qui n'est pas la sienne.
    """
    b = panel([GAGNE, PERD, GAGNE, PERD, GAGNE])
    candidats = [cand(i, score=i / 100) for i in range(0, 14)]
    o = ordonnancer(candidats, b)
    if o.taux_de_retention >= 1.0:
        _echec("tous les candidats retenus : l'occupation ne filtre rien, ce qui est impossible")
    if not o.retenus:
        _echec("aucun candidat retenu : l'ordonnanceur bloque tout")
    print(
        f" 14. la cadence réelle est réduite par l'occupation : {len(o.retenus)}/{o.n_candidats} "
        f"({100 * o.taux_de_retention:.0f} %) — c'est elle qui fixe la colonne du null (G3)  ✓"
    )


def check_90_C1_resoudre_trades_suit_la_sortie_propre_de_la_decision() -> None:
    """
    Le SECOND résolveur (celui de l'occupation) doit dire la même chose que le simulateur :
    une décision qui porte `dt_sortie` est détenue jusqu'à cette barre EXCLUE, son rendement est
    celui de l'Open de la barre de sortie, et le compte est libre à cette barre.
    ⚠️ Fixture hostile : la barre de sortie a un Open à 102 et un Close à 100 ; sortir au Close
    donnerait 0, sortir à l'Open donne +2 — soit +2R avec un stop à 1 point.
    """
    b = panel([[PLAT] * 9 + [(102.0, 102.0, 102.0, 100.0)] + [PLAT] * 6])
    d = Decision(dt(1), LONG, 100.0, None, 99.0, dt_sortie=dt(9))
    t = resoudre_trades([d], b)[0]
    if t.duree_barres != 8 or t.barre_sortie != 8:
        _echec(
            f"durée {t.duree_barres}, barre de sortie {t.barre_sortie} : attendu 8 et 8 (barres 1 à 8 détenues)"
        )
    if abs(t.rendement - 2.0) > 1e-9:
        _echec(
            f"rendement {t.rendement}R : attendu +2R, le rendement de l'OPEN de la barre de sortie"
        )
    o = ordonnancer([d, cand(8), cand(9)], b)
    entrees = [x.dt_entree for x in o.retenus]
    if dt(8) in entrees or dt(9) not in entrees:
        _echec(
            f"occupation fausse : retenus aux barres {[(e - dt(0)).total_seconds() // 3600 for e in entrees]} — attendu 1 et 9, pas 8"
        )
    print(
        "  90. (C1) resoudre_trades suit dt_sortie : 8 barres détenues, +2R à l'Open, compte libre à la barre 9  ✓"
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DE L'ORDONNANCEUR D'OCCUPATION — tâche 4.10 (H11, F14)")
    print("=" * 78 + "\n")

    print("  ── l'occupation du compte ──")
    check_1_un_seul_trade_a_la_fois()
    check_2_occupation_dure_exactement_la_vie_du_trade()
    check_3_un_trade_court_libere_plus_tot()
    check_4_candidats_pendant_occupation_comptes()

    print("\n  ── F14 : le départage par le score ──")
    check_5_le_mieux_note_gagne()
    check_6_lordre_de_la_liste_ne_change_rien()
    check_7_departage_sans_score_leve()
    check_8_score_partiel_leve_aussi()

    print("\n  ── garde-fous et invariant central ──")
    check_9_candidats_non_tries_refuses()
    check_10_liste_vide()
    check_11_retenus_sont_un_sous_ensemble_ordonne()
    check_12_ordonnancement_independant_du_sizing()
    check_13_deterministe()
    check_14_cadence_reduite_par_loccupation()

    check_90_C1_resoudre_trades_suit_la_sortie_propre_de_la_decision()
    print("\n" + "=" * 78)
    print("LES 15 PREUVES PASSENT.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
