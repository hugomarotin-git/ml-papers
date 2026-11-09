"""
Preuve de la calibration, du budget de temps et du plancher de taille — tâche 6.4.
Décisions H5 à H8, H16, F13, F19, G4.

**Ce module produit la barre à laquelle le modèle sera comparé.** Une erreur ici ne plante pas :
elle produit un **classement**, donc un verdict. Trois propriétés valent plus que les autres, et
chacune a déjà failli nous tromper pour de vrai :

1. **le budget de temps est le même pour tous** (checks 4 à 6). Sans lui, un chemin parti tôt a
   quatre ans et un chemin parti tard quelques mois : `EN_COURS` ne dit alors rien de la
   configuration, seulement de la date de départ ;
2. **le plancher de taille rend la mort possible** (checks 7 à 9). Sans lui, l'airbag crée une
   **asymptote** — son plafond vaut `marge / N`, donc chaque perte retire une fraction de ce qui
   reste et le plancher FTMO n'est jamais atteint. **Mesuré le 2026-08-21 : zéro bust**, et un
   `P(funded)` de 1,000 faute de bust au dénominateur ;
3. **un point non mesurable n'est pas un mauvais point** (checks 10 et 11). `P(funded)` vaut 0
   par convention quand rien n'a conclu ; le classer reviendrait à le condamner pour une question
   qu'on n'a jamais posée.

Lancer :  .venv/bin/python juge/sanity_check_calibration.py
"""

from __future__ import annotations

import sys

import numpy as np
from aimd import AIMDConfig
from calibration import (
    AXES,
    UNITES_MINIMUM_PROVISOIRE,
    PointDeGrille,
    calibrer,
    classement,
    grille_complete,
)
from methodes_pfunded import departs_utilisables, methode_a
from regles_ftmo import ReglesFTMO
from sanity_check_simulateur import GAGNE, PERD, PLAT, dec, panel
from simulateur import simuler
from trajectoires import SEUIL_CENSURE_ALARMANT, jouer_chemin

R = ReglesFTMO()


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


def cfg(**remplacements) -> AIMDConfig:
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


def _point(**remplacements) -> PointDeGrille:
    base = dict(
        config=cfg(),
        p_funded=0.3,
        p_bust=0.7,
        p_censure=0.02,
        n_chemins=100,
        n_resolus=98,
        n_funded=29,
        p_funded_borne_basse=0.29,
        duree_mediane_funded=40.0,
        duree_mediane_bust=45.0,
        n_montees=10,
        n_coupes=5,
        n_declenchements_coupe=6,
        n_declenchements_airbag=0,
        avertissements=(),
    )
    base.update(remplacements)
    return PointDeGrille(**base)  # pyright: ignore[reportArgumentType]


# ── La grille (H5, H6, A8) ───────────────────────────────────────────────────────────────
def check_1_la_grille_a_la_bonne_forme_et_aucun_doublon() -> None:
    """
    Le produit cartésien des cinq axes, sans répétition. Un doublon ferait jouer deux fois la
    même configuration et pèserait double dans toute lecture agrégée, sans qu'on le voie.
    """
    configs = grille_complete()
    attendu = 1
    for valeurs in AXES.values():
        attendu *= len(valeurs)
    if len(configs) != attendu:
        _echec(f"{len(configs)} configurations pour un produit cartésien de {attendu}")
    if len({repr(c) for c in configs}) != len(configs):
        _echec("la grille contient des doublons : une configuration pèserait double")
    for axe, valeurs in AXES.items():
        vues = {getattr(c, axe) for c in configs}
        if vues != set(valeurs):
            _echec(f"axe {axe} : valeurs vues {vues}, attendu {set(valeurs)}")
    print(
        f"  1. grille = produit cartésien exact des {len(AXES)} axes ({len(configs)} configs, sans doublon)  ✓"
    )


def check_2_toute_la_grille_passe_la_validation_de_5_5() -> None:
    """
    Chaque configuration est validée à la construction, **y compris contre le barème** :
    `risque_base x m_max` doit rester strictement sous le plancher journalier (5.5).

    Une grille dont un coin serait refusé planterait au milieu d'une campagne de vingt minutes.
    Pire : si la garde disparaissait, la grille contiendrait des configurations **mortelles par
    construction** — un seul trade perdant tuant la journée — et elles seraient calibrées comme
    les autres.
    """
    pire = max(c.risque_base * c.m_max for c in grille_complete())
    if not pire < R.plancher_journalier_pct:
        _echec(f"le pire produit risque x m_max vaut {pire}, au-delà du plancher journalier")
    print(
        f"  2. les {len(grille_complete())} configs passent la garde de 5.5 (pire produit {pire:.4f})  ✓"
    )


def check_3_la_meme_sequence_pour_TOUTES_les_configs() -> None:
    """
    **H8 — la comparaison est appariée, et c'est structurel.** `calibrer` ne reçoit qu'une seule
    liste de décisions : il n'existe aucun moyen de comparer deux configurations sur deux
    séquences différentes.

    On le vérifie par l'observable : à budget et pas identiques, toutes les configurations jouent
    **exactement le même nombre de chemins**. Une différence signalerait que la sélection des
    départs dépend du sizing — ce que H11 interdit.
    """
    # Un panel gagnant : à 2 % de risque le compte atteint la cible, à 0,5 % non. Sur une série
    # neutre, toutes les configs resteraient EN_COURS et l'essai ne prouverait rien.
    b = panel([GAGNE] * 14)
    d = [dec(i) for i in range(0, len(b), 3)]
    points = calibrer(
        d,
        b,
        [cfg(risque_base=0.005), cfg(risque_base=0.02), cfg(risque_base=0.01)],
        pas=1,
        budget_journees=None,
    )
    if len({p.n_chemins for p in points}) != 1:
        _echec(f"nombres de chemins différents entre configs : {[p.n_chemins for p in points]}")
    if len({p.p_funded for p in points}) == 1 and len({p.p_bust for p in points}) == 1:
        _echec("les trois configs donnent exactement le même résultat : l'essai ne discrimine rien")
    print(f"  3. H8 : {points[0].n_chemins} chemins identiques pour toutes les configs  ✓")


# ── Le budget de temps (décision de Hugo, 2026-08-21) ────────────────────────────────────
def check_4_le_budget_donne_le_MEME_temps_a_tous() -> None:
    """
    Sans budget, la fenêtre d'un chemin va de son départ à la **fin des données** : deux chemins
    n'ont donc pas le même temps pour conclure, et `EN_COURS` mesure la date de départ plutôt que
    la configuration.

    On le vérifie sur ce qui compte : deux départs éloignés doivent produire des simulations qui
    couvrent **le même nombre de journées FTMO**.
    """
    b = panel([[PLAT] * 3] * 40)
    d = [dec(i) for i in range(0, len(b) - 3, 3)]
    budget = 10
    tot = departs_utilisables(d, b, budget)
    if tot < 3:
        _echec("le fixture ne discrimine pas : trop peu de départs disposent du budget")

    tot_sans = [jouer_chemin(d, b, k, 0.01).n_journees for k in (0, tot - 1)]
    if tot_sans[0] == tot_sans[1]:
        _echec(
            f"sans budget, deux départs éloignés couvrent déjà {tot_sans[0]} journées : rien à corriger ?"
        )
    avec = [jouer_chemin(d, b, k, 0.01, None, None, budget).n_journees for k in (0, tot - 1)]
    if avec != [budget, budget]:
        _echec(f"avec budget, les chemins couvrent {avec} journées, attendu {[budget, budget]}")
    print(f"  4. budget : {tot_sans} journées sans budget -> {avec} avec  ✓")


def check_5_les_departs_sans_budget_entier_sont_ECARTES() -> None:
    """
    Tronquer ne suffit pas : il faut aussi **ne pas partir** de points qui n'ont pas le budget
    entier, sinon on rétablit le défaut qu'on vient de corriger.

    `departs_utilisables` rend un **seuil**, pas un ensemble : les départs sont chronologiques,
    donc la propriété est monotone. Ça rend le filtre impossible à appliquer de travers.
    """
    b = panel([[PLAT] * 3] * 30)
    d = [dec(i) for i in range(0, len(b) - 3, 3)]
    seuil = departs_utilisables(d, b, 10)
    if not 0 < seuil < len(d):
        _echec(f"seuil {seuil} sur {len(d)} départs : l'essai ne discrimine pas")
    jouer_chemin(d, b, seuil - 1, 0.01, None, None, 10)  # le dernier utilisable passe
    _leve(
        lambda: jouer_chemin(d, b, seuil, 0.01, None, None, 10),
        ValueError,
        "un départ sans le budget entier",
        contient="moins de temps qu'aux autres",
    )
    # Et `methode_a` doit les écarter d'elle-même, pas lever.
    dist = methode_a(d, b, risque_par_trade_pct=0.01, budget_journees=10)
    if dist.n_chemins != seuil:
        _echec(f"methode_a a joué {dist.n_chemins} chemins, attendu les {seuil} utilisables")
    print(f"  5. seuil de départs utilisables = {seuil}/{len(d)}, et methode_a s'y tient  ✓")


def check_6_le_seuil_ne_depend_QUE_des_donnees() -> None:
    """
    ⚠️ Le point qui garantit H8 : si le filtre des départs dépendait du sizing, deux
    configurations ne joueraient plus les mêmes chemins et l'appariement serait perdu — sans que
    rien ne plante, et avec des `n_chemins` qui se ressembleraient assez pour ne pas alerter.
    """
    b = panel([[PLAT] * 3] * 30)
    d = [dec(i) for i in range(0, len(b) - 3, 3)]
    seuil = departs_utilisables(d, b, 10)
    points = calibrer(d, b, [cfg(), cfg(risque_base=0.02, m_max=1.5)], pas=1, budget_journees=10)
    if {p.n_chemins for p in points} != {seuil}:
        _echec(f"chemins joués {[p.n_chemins for p in points]}, attendu {seuil} pour les deux")
    _leve(
        lambda: departs_utilisables(d, b, 0),
        ValueError,
        "un budget nul",
        contient="au moins 1",
    )
    print("  6. le seuil ne dépend que des données : deux configs jouent les mêmes départs  ✓")


# ── Le plancher de taille (décision de Hugo, 2026-08-22, lecture A) ──────────────────────
def check_7_sous_le_plancher_on_prend_le_MINIMUM_donc_on_SUR_RISQUE() -> None:
    """
    **Lecture A.** En dessous du lot minimum, un compte réel ne se met pas en pause : il passe le
    minimum, et **dépasse donc le plafond que le sizing demandait**.

    C'est ce dépassement qui rend la mort de nouveau possible — sans lui, l'airbag ne fait
    qu'approcher le plancher FTMO sans jamais l'atteindre.

    ⚠️ Le risque **enregistré** doit être celui qui a réellement été pris. Rapporter le risque
    *voulu* ferait mentir le diagnostic sur ce qui était en jeu.
    """
    b = panel([GAGNE])
    d = [dec(0)]
    petit = 0.0001  # 0,01 % du capital
    sans = simuler(d, b, risque_par_trade_pct=petit)
    if sans.risques != [petit]:
        _echec(f"sans plancher, risque enregistré {sans.risques}, attendu {[petit]}")

    unites_voulues = (R.capital_initial * petit) / d[0].distance_stop
    plancher = unites_voulues * 10  # dix fois trop gros pour le sizing demandé
    avec = simuler(d, b, risque_par_trade_pct=petit, unites_minimum=plancher)
    attendu = (plancher * d[0].distance_stop) / R.capital_initial
    if not np.isclose(avec.risques[0], attendu, rtol=1e-12):
        _echec(f"risque enregistré {avec.risques[0]}, attendu {attendu} (celui RÉELLEMENT pris)")
    if avec.risques[0] <= petit:
        _echec("le plancher n'a pas fait SUR-risquer : la lecture A n'est pas appliquée")
    print(
        f"  7. sous le plancher on prend le minimum : {petit} demandé -> {avec.risques[0]:.5f} pris  ✓"
    )


def check_8_le_plancher_ne_touche_PAS_les_trades_normaux() -> None:
    """Un plancher qui mordrait tout le temps serait un sizing constant déguisé."""
    b = panel([GAGNE])
    d = [dec(0)]
    normal = simuler(d, b, risque_par_trade_pct=0.01)
    avec = simuler(d, b, risque_par_trade_pct=0.01, unites_minimum=UNITES_MINIMUM_PROVISOIRE)
    if normal.risques != avec.risques or normal.balance_finale != avec.balance_finale:
        _echec("le plancher a modifié un trade de taille normale")
    for mauvais in (-1.0, float("nan"), float("inf")):
        _leve(
            lambda m=mauvais: simuler(d, b, risque_par_trade_pct=0.01, unites_minimum=m),
            ValueError,
            f"unites_minimum={mauvais}",
            contient="fini",
        )
    print("  8. le plancher ne touche pas un trade normal, et refuse une valeur absurde  ✓")


def check_9_SANS_plancher_lairbag_rend_la_mort_IMPOSSIBLE() -> None:
    """
    **Le défaut qui a coûté deux jours d'analyse, gravé en preuve.**

    Le plafond de l'airbag vaut `marge / N` : chaque perte retire une **fraction** de ce qui
    reste, jamais un montant fixe. La marge décroît géométriquement et **n'atteint jamais zéro** —
    le compte ne meurt pas, il trade des tailles microscopiques.

    Or H16 lit `N` comme « combien de trades perdants au minimum pour mourir ». Les deux moitiés
    de la décision se contredisent : la formule dit **l'infini**.

    Mesuré le 2026-08-21 sur les vraies données : **zéro bust** avec airbag contre 133 sans, à
    nombre de financés identique, et `P(funded)` affichant **1,000** faute de bust au dénominateur.
    """
    marge, n = 5000.0, 2
    restante = marge
    for _ in range(60):
        restante -= restante / n
    if restante <= 0:
        _echec("la marge atteint zéro : la démonstration de l'asymptote n'a plus d'objet")

    # Et sur le simulateur : une longue série perdante ne doit PAS buster sans plancher…
    b = panel([PERD] * 25)
    d = [dec(i) for i in range(0, len(b), 3)]
    config = cfg(risque_base=0.02, airbag_n=2, m_min=0.25)
    sans = simuler(d, b, aimd=config)
    if sans.n_declenchements_airbag == 0:
        _echec("l'airbag n'a jamais mordu : l'essai ne démontre rien")
    if sans.motif_bust is not None:
        _echec("le compte a busté SANS plancher : l'asymptote n'existe pas, la preuve est vide")
    print(
        f"  9. sans plancher, l'airbag interdit la mort : marge encore à {restante:.2e} $ après "
        "60 pertes, et aucun bust en simulation  ✓"
    )


# ── Lire un point de grille (H7, F13, F19, G4) ───────────────────────────────────────────
def check_10_un_point_NON_MESURABLE_nest_pas_un_mauvais_point() -> None:
    """
    Quand aucun chemin ne conclut, `P(funded)` vaut **0 par convention** — le juge le dit lui-même
    dans ses avertissements : *« ce n'est pas un résultat »*. Un classement naïf trierait ce point
    **en dernier**, comme s'il ne finançait jamais, alors qu'on n'en sait rien.
    """
    rien = _point(p_funded=0.0, p_bust=0.0, p_censure=1.0, n_resolus=0)
    if rien.interpretable:
        _echec("un point sans aucun chemin conclu est déclaré interprétable")
    trop = _point(p_censure=SEUIL_CENSURE_ALARMANT + 0.01, n_resolus=50)
    if trop.interpretable:
        _echec(f"un point à {trop.p_censure:.0%} de censure est déclaré interprétable (seuil F13)")
    juste = _point(p_censure=SEUIL_CENSURE_ALARMANT, n_resolus=50)
    if not juste.interpretable:
        _echec("le seuil F13 doit être inclusif : à 10 % pile, le point reste interprétable")
    print("  10. rien conclu ou censure > seuil F13 -> non interprétable  ✓")


def check_11_le_classement_ECARTE_au_lieu_de_classer_en_dernier() -> None:
    """
    ⚠️ **Le tri n'est pas un verdict.** À ±1,8 point de bruit, deux points voisins sont
    indiscernables tant que 6.6 n'a pas mesuré la dispersion des écarts appariés. `classement`
    ordonne, il ne tranche pas — et il **retire** ce qui n'est pas mesurable au lieu de le
    condamner.
    """
    bons = [
        _point(p_funded=x, config=cfg(risque_base=r))
        for x, r in ((0.2, 0.005), (0.4, 0.01), (0.3, 0.02))
    ]
    muet = _point(p_funded=0.0, p_bust=0.0, p_censure=1.0, n_resolus=0, config=cfg(m_max=1.5))
    classe = classement([*bons, muet])
    if len(classe) != 3:
        _echec(f"{len(classe)} points classés sur 3 interprétables : le muet n'a pas été écarté")
    if [round(p.p_funded, 3) for p in classe] != [0.4, 0.3, 0.2]:
        _echec(f"ordre {[p.p_funded for p in classe]}, attendu décroissant")
    if any(p.n_resolus == 0 for p in classe):
        _echec("un point non mesurable a été classé")
    print("  11. le classement écarte les points muets et trie par P(funded) décroissant  ✓")


def check_12_les_DEUX_colonnes_et_la_duree_voyagent_avec_le_point() -> None:
    """
    **H7 — P(funded) ET P(bust), jamais une seule.** Sur un système sans edge, `P(funded)` monte
    mécaniquement avec le risque : choisir sur cette colonne seule sélectionnerait le risque le
    plus élevé, systématiquement.

    Et la **durée** voyage avec, parce que c'est elle qui trahit une configuration dont les morts
    sont tronquées par le budget — celle dont le bust médian dépasse largement le funded médian.
    """
    # Panel gagnant : il FAUT que des chemins concluent, sinon P(funded) et P(bust) valent tous
    # deux zéro et leur somme ne prouve rien — le check serait vide.
    b = panel([GAGNE] * 14)
    d = [dec(i) for i in range(0, len(b), 3)]
    p = calibrer(d, b, [cfg(risque_base=0.02)], pas=1, budget_journees=None)[0]
    if p.n_resolus == 0:
        _echec("aucun chemin conclu : la somme des deux colonnes ne prouverait rien")
    for nom in ("p_funded", "p_bust", "p_censure", "duree_mediane_funded", "duree_mediane_bust"):
        if not hasattr(p, nom):
            _echec(f"un point de grille ne porte pas `{nom}`")
    if not np.isclose(p.p_funded + p.p_bust, 1.0, rtol=1e-12):
        _echec(f"P(funded) {p.p_funded} + P(bust) {p.p_bust} != 1 sur les chemins conclus")
    texte = p.resume()
    for attendu in ("P(funded)", "P(bust)", "censure", "durée"):
        if attendu not in texte:
            _echec(f"le résumé d'un point ne montre pas « {attendu} » : {texte}")
    print("  12. les deux colonnes, la censure et la durée voyagent avec le point (H7, F13)  ✓")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DE LA CALIBRATION — tâche 6.4 (H5-H8, H16, F13, F19)")
    print("=" * 78 + "\n")

    print("  ── la grille (H5, H6, A8) ──")
    check_1_la_grille_a_la_bonne_forme_et_aucun_doublon()
    check_2_toute_la_grille_passe_la_validation_de_5_5()
    check_3_la_meme_sequence_pour_TOUTES_les_configs()

    print("\n  ── le budget de temps, le même pour tous ──")
    check_4_le_budget_donne_le_MEME_temps_a_tous()
    check_5_les_departs_sans_budget_entier_sont_ECARTES()
    check_6_le_seuil_ne_depend_QUE_des_donnees()

    print("\n  ── le plancher de taille (lecture A) ──")
    check_7_sous_le_plancher_on_prend_le_MINIMUM_donc_on_SUR_RISQUE()
    check_8_le_plancher_ne_touche_PAS_les_trades_normaux()
    check_9_SANS_plancher_lairbag_rend_la_mort_IMPOSSIBLE()

    print("\n  ── lire un point de grille (H7, F13, F19) ──")
    check_10_un_point_NON_MESURABLE_nest_pas_un_mauvais_point()
    check_11_le_classement_ECARTE_au_lieu_de_classer_en_dernier()
    check_12_les_DEUX_colonnes_et_la_duree_voyagent_avec_le_point()
    check_93_calibrer_sans_configuration_est_refuse()

    print("\n" + "=" * 78)
    print("LES 13 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_calibrer_sans_configuration_est_refuse() -> None:
    """
    La garde qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    Une liste vide rendrait une liste vide : « aucune configuration ne bat les autres » se
    lirait comme un résultat, alors que c'est un appel qui n'a rien demandé.
    """
    b = panel([[PLAT] * 3] * 30)
    d = [dec(i) for i in range(0, len(b) - 3, 3)]
    _leve(
        lambda: calibrer(d, b, [], pas=1, budget_journees=10),
        ValueError,
        "calibrer sans configuration",
        "aucune configuration",
    )
    print("  93. calibrer sans aucune configuration est refusé, pas rendu vide  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
