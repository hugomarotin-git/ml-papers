"""
Preuve de la simulation full-path et de l'agrégation — tâche 4.6.

Dix-huit vérifications. L'agrégation est une fonction **pure** : elle ne voit ni marché ni tirage,
seulement une liste d'issues. On peut donc lui donner des issues fabriquées et calculer la bonne
réponse de tête — c'est la seule façon de prouver une statistique, et c'est pour ça que
`agreger` a été séparée de la simulation.

Ce qui est réellement en jeu : **le traitement des chemins qui n'ont pas conclu**. Un challenge
FTMO n'ayant aucune limite de temps, ils ne sont pas des échecs — ils sont inachevés. Le verdict
porte donc sur les chemins **conclus** (décision F19), et leur nombre est rapporté à côté.

Les checks 2 à 6 verrouillent cette convention. Le **check 4bis verrouille son piège** : un
système trop lent peut afficher un `p_funded` magnifique en laissant presque tout en suspens.

Lancer :  .venv/bin/python juge/sanity_check_trajectoires.py
"""

from __future__ import annotations

import sys

from regles_ftmo import Issue
from sanity_check_simulateur import GAGNE, PLAT, dec, dt, panel
from simulateur import LONG, Decision, ResultatSimulation
from trajectoires import SEUIL_CENSURE_ALARMANT, agreger, jouer_chemin


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """
    Vérifie qu'un appel lève — et, si `contient` est donné, que c'est **pour la bonne raison**.

    ⚠️ Sans ce contrôle du message, ce helper valide des tests qui passent par accident. Cas
    réellement rencontré le 2026-08-09 : la preuve du refus des décisions non triées passait
    même avec la garde désactivée, parce qu'une AUTRE `ValueError` — « barre d'entrée
    inexistante » — remontait à sa place. La mutation correspondante n'était donc pas détectée,
    et la preuve était décorative sans en avoir l'air.
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


def faux(issue: Issue, jours: int = 10, motif: str | None = None) -> ResultatSimulation:
    """Un résultat fabriqué : `agreger` ne regarde que l'issue, la durée et le motif."""
    return ResultatSimulation(
        issue=issue,
        balance_finale=50_000.0,
        n_trades=1,
        n_barres=1,
        n_decisions_ignorees=0,
        motif_bust=motif,
        jours_tradees=1,
        n_journees=jours,
    )


def lot(n_funded: int, n_bust: int, n_censures: int) -> list[ResultatSimulation]:
    return (
        [faux(Issue.FUNDED) for _ in range(n_funded)]
        + [faux(Issue.BUST, motif="plancher_journalier") for _ in range(n_bust)]
        + [faux(Issue.EN_COURS) for _ in range(n_censures)]
    )


# ── Le comptage ──────────────────────────────────────────────────────────────────────────
def check_1_comptage_exact() -> None:
    d = agreger(lot(3, 5, 2))
    if (d.n_chemins, d.n_funded, d.n_bust, d.n_censures) != (10, 3, 5, 2):
        _echec(f"comptage faux : {d}")
    print("  1. les chemins sont comptés par issue, sans perte  ✓")


def check_2_funded_et_bust_somment_a_un_sur_les_resolus() -> None:
    """
    Les deux proportions du verdict partagent le même dénominateur — les chemins conclus — donc
    elles somment à 1. Des dénominateurs différents entre `p_funded` et `p_bust` seraient
    inlisibles et produiraient des comparaisons fausses sans rien casser.
    """
    for a, b, c in ((3, 5, 2), (10, 0, 0), (1, 1, 1), (0, 4, 9)):
        d = agreger(lot(a, b, c))
        total = d.p_funded + d.p_bust
        if abs(total - 1.0) > 1e-12:
            _echec(f"({a},{b},{c}) : p_funded + p_bust = {total}, attendu 1 sur les résolus")
    print("  2. P(funded) + P(bust) = 1 sur les chemins CONCLUS  ✓")


def check_3_les_censures_sont_ecartees_du_denominateur() -> None:
    """
    **La preuve centrale, et elle a été RETOURNÉE le 2026-08-09** (décision F19).

    3 financés, 5 bustés, 2 non conclus. Le verdict porte sur les 8 qui ont conclu :
    `3/8 = 37,5 %`. Les compter comme des pertes donnerait `3/10 = 30 %` — une pénalité
    mécanique qui affirme un échec jamais observé, alors qu'un challenge FTMO n'a **aucune
    limite de temps**.

    La borne basse reste calculée, sous un nom qui dit ce qu'elle est.
    """
    d = agreger(lot(3, 5, 2))
    if abs(d.p_funded - 0.375) > 1e-12:
        _echec(
            f"p_funded = {d.p_funded}, attendu 0,375 (3/8, les conclus). Si c'est 0,30, les "
            "non conclus sont revenus au dénominateur — ils y comptent comme des pertes"
        )
    if abs(d.p_funded_borne_basse - 0.30) > 1e-12:
        _echec(f"p_funded_borne_basse = {d.p_funded_borne_basse}, attendu 0,30 (3/10)")
    if d.n_resolus != 8:
        _echec(f"n_resolus = {d.n_resolus}, attendu 8")
    print("  3. le verdict porte sur les chemins CONCLUS : 3/8, pas 3/10  ✓")


def check_3bis_relation_exacte_entre_les_deux_bornes() -> None:
    """
    `borne_basse = p_funded x (1 - c)`. La relation est exacte, donc vérifiable — et elle dit que
    la pénalité de l'ancienne convention était un facteur connu, pas une mesure.
    """
    for a, b, c in ((3, 5, 2), (40, 40, 20), (7, 3, 0), (1, 9, 30)):
        d = agreger(lot(a, b, c))
        attendu = d.p_funded * (1 - d.p_censure)
        if abs(d.p_funded_borne_basse - attendu) > 1e-12:
            _echec(f"({a},{b},{c}) : borne_basse={d.p_funded_borne_basse}, attendu {attendu}")
    print("  3bis. borne basse = P(funded) x (1 - taux de non-conclusion), exactement  ✓")


def check_4_avertissement_de_censure() -> None:
    """F13 — au-delà de 10 % de chemins non conclus, le chiffre s'annonce avec sa réserve."""
    calme = agreger(lot(50, 45, 5))  # 5 % de censure
    if any("CENSURE" in m for m in calme.avertissements()):
        _echec("5 % de censure ne doit PAS déclencher l'avertissement")
    bruyant = agreger(lot(50, 30, 20))  # 20 % de censure
    if not any("CENSURE" in m for m in bruyant.avertissements()):
        _echec(
            f"20 % de censure doit déclencher l'avertissement (seuil "
            f"{100 * SEUIL_CENSURE_ALARMANT:.0f} %, F13)"
        )
    print("  4. au-delà de 10 % de censure, un avertissement est émis  ✓")


def check_4bis_le_piege_du_systeme_trop_lent() -> None:
    """
    **Le risque propre à la convention F19, et la raison pour laquelle `p_censure` se lit AVEC le
    chiffre et jamais après.**

    Un système très lent — sizing minuscule, cadence rare — buste peu et finance rarement : la
    plupart de ses chemins ne concluent pas. S'il en résout 10 et en finance 8, il affiche un
    `p_funded` de 80 % en ayant laissé 90 % des chemins en suspens. Le chiffre est exact, la
    lecture est absurde.

    C'est exactement le piège de la calibration AIMD (H5-H8) : la configuration la plus prudente
    gagnerait sur `p_funded` sans rien prouver. Les deux avertissements doivent partir.
    """
    d = agreger(lot(8, 2, 90))
    if abs(d.p_funded - 0.80) > 1e-12:
        _echec(f"p_funded = {d.p_funded}, attendu 0,80 (8/10 conclus)")
    avert = d.avertissements()
    if not any("CENSURE" in m for m in avert):
        _echec("90 % de non-conclus : l'avertissement de censure DOIT partir")
    if not any("MINCE" in m for m in avert):
        _echec("10 chemins résolus : l'avertissement d'échantillon mince DOIT partir")
    if abs(d.p_funded_borne_basse - 0.08) > 1e-12:
        _echec(
            f"borne basse = {d.p_funded_borne_basse}, attendu 0,08 — l'écart doit sauter aux yeux"
        )
    print("  4bis. un système trop lent affiche 80 % avec 90 % de non-conclus : DEUX alertes  ✓")


def check_5_echantillon_mince_signale() -> None:
    if not any("MINCE" in m for m in agreger(lot(2, 3, 0)).avertissements()):
        _echec("5 chemins : l'échantillon mince doit être signalé")
    if any("MINCE" in m for m in agreger(lot(50, 50, 0)).avertissements()):
        _echec("100 chemins ne sont pas un échantillon mince")
    print("  5. un échantillon trop mince pour conclure est signalé  ✓")


def check_6_zero_funded_nest_pas_un_arrondi() -> None:
    d = agreger(lot(0, 10, 0))
    if d.p_funded != 0.0:
        _echec(f"p_funded = {d.p_funded}, attendu 0")
    if not any("AUCUN" in m for m in d.avertissements()):
        _echec("un P(funded) nul doit être annoncé comme tel, pas laissé à l'arrondi")
    print("  6. P(funded) = 0 est annoncé explicitement  ✓")


def check_7_aucun_chemin() -> None:
    d = agreger([])
    if (d.n_chemins, d.p_funded, d.p_bust, d.p_censure) != (0, 0.0, 0.0, 0.0):
        _echec("l'agrégat vide doit être neutre, pas une division par zéro")
    if not d.avertissements():
        _echec("agréger zéro chemin doit dire qu'il n'y a rien à interpréter")
    print("  7. agréger zéro chemin ne divise pas par zéro et le dit  ✓")


# ── Les durées ───────────────────────────────────────────────────────────────────────────
def check_8_medianes_separees_par_issue() -> None:
    """
    Mélanger les durées est la faute classique : un bust au 3ᵉ jour et un financement au 40ᵉ
    donnent une « durée médiane » qui ne décrit ni l'un ni l'autre.
    """
    res = [faux(Issue.FUNDED, 40), faux(Issue.FUNDED, 42), faux(Issue.BUST, 3, "plancher_total")]
    d = agreger(res)
    if d.duree_mediane_funded != 41:
        _echec(f"médiane funded = {d.duree_mediane_funded}, attendu 41 (40 et 42)")
    if d.duree_mediane_bust != 3:
        _echec(f"médiane bust = {d.duree_mediane_bust}, attendu 3 — les issues ne se mélangent pas")
    print("  8. durée médiane calculée par ISSUE, jamais toutes confondues  ✓")


def check_9_pas_de_mediane_sans_chemin() -> None:
    """Une médiane sur ensemble vide vaut `None`, pas zéro : zéro se lirait comme « instantané »."""
    sans_funded = agreger(lot(0, 3, 0))
    if sans_funded.duree_mediane_funded is not None:
        _echec(f"aucun funded : médiane attendue None, reçu {sans_funded.duree_mediane_funded}")
    if sans_funded.duree_mediane_bust is None:
        _echec("des busts existent : leur médiane ne doit pas être None")

    # Le symétrique, sans lequel la moitié du garde-fou n'est jamais exercée : ma première
    # version ne testait que le côté funded, et la mutation « médiane bust rendue comme 0 »
    # passait tranquillement.
    sans_bust = agreger(lot(3, 0, 0))
    if sans_bust.duree_mediane_bust is not None:
        _echec(f"aucun bust : médiane attendue None, reçu {sans_bust.duree_mediane_bust}")
    if sans_bust.duree_mediane_funded is None:
        _echec("des funded existent : leur médiane ne doit pas être None")
    print("  9. sans chemin d'une issue, sa médiane est None — jamais 0, des DEUX côtés  ✓")


def check_10_censures_sans_duree() -> None:
    """
    Un censuré n'a pas de durée : elle est justement inconnue. Le compter dans une médiane la
    tirerait vers le bas d'autant plus que la série est courte.
    """
    d = agreger([faux(Issue.FUNDED, 40), faux(Issue.EN_COURS, 1), faux(Issue.EN_COURS, 1)])
    if d.duree_mediane_funded != 40:
        _echec(
            f"médiane funded = {d.duree_mediane_funded}, attendu 40 : les censurés ne doivent "
            "entrer dans AUCUNE médiane"
        )
    print(" 10. les chemins censurés n'entrent dans aucune médiane  ✓")


def check_11_motifs_de_bust_ventiles() -> None:
    """H16 — lequel des deux planchers mord alimente l'airbag, qui vise le plus proche.

    *(Cette preuve citait H2 jusqu'au 2026-08-15, avant que le déclencheur ne devienne le
    résultat net de la journée. Ce qu'elle vérifie n'a pas changé.)*
    """
    res = [
        faux(Issue.BUST, 5, "plancher_journalier"),
        faux(Issue.BUST, 6, "plancher_journalier"),
        faux(Issue.BUST, 20, "plancher_total"),
    ]
    d = agreger(res)
    if d.motifs_bust != {"plancher_journalier": 2, "plancher_total": 1}:
        _echec(f"ventilation des motifs fausse : {d.motifs_bust}")
    print(" 11. les motifs de bust sont ventilés, jamais agrégés en un seul nombre  ✓")


# ── La simulation full-path ──────────────────────────────────────────────────────────────
def check_12_depart_ecarte_le_passe() -> None:
    """
    Un challenge qui démarre à la 3ᵉ décision ne doit pas voir les barres d'avant : sa première
    journée FTMO poserait son plancher sur une barre antérieure à l'ouverture du compte.
    """
    b = panel([GAGNE, GAGNE, GAGNE, GAGNE])
    decisions = [dec(1 + 3 * k) for k in range(4)]
    complet = jouer_chemin(decisions, b, 0, risque_par_trade_pct=0.01)
    tardif = jouer_chemin(decisions, b, 2, risque_par_trade_pct=0.01)
    if complet.n_trades != 4 or tardif.n_trades != 2:
        _echec(
            f"départ 0 -> {complet.n_trades} trades, départ 2 -> {tardif.n_trades} (attendu 4 et 2)"
        )
    if tardif.n_journees >= complet.n_journees:
        _echec(
            f"le chemin tardif couvre {tardif.n_journees} journées contre {complet.n_journees} : "
            "les barres antérieures au départ n'ont pas été écartées"
        )
    print(" 12. un départ tardif n'hérite pas des barres qui le précèdent  ✓")


def check_13_duree_en_journees_ftmo_pas_en_phases() -> None:
    """
    `n_journees` compte les journées FTMO **écoulées**, `jours_tradees` celles de la phase en
    cours — ce dernier repart à zéro en phase 2. Les confondre diviserait la durée d'un
    challenge financé par deux, sans qu'aucun test ne bronche.
    """
    b = panel([GAGNE] * 8)
    r = jouer_chemin([dec(1 + 3 * k) for k in range(8)], b, 0, risque_par_trade_pct=0.04)
    if r.issue is not Issue.FUNDED:
        _echec(f"attendu FUNDED pour cette preuve, reçu {r.issue.value}")
    if r.n_journees <= r.jours_tradees:
        _echec(
            f"n_journees={r.n_journees} <= jours_tradees={r.jours_tradees} : la durée du "
            "challenge doit couvrir les DEUX phases, pas seulement la dernière"
        )
    print(f" 13. la durée compte les journées FTMO des deux phases ({r.n_journees})  ✓")


def check_14_departs_hors_bornes_refuses() -> None:
    b = panel([GAGNE, GAGNE])
    decisions = [dec(1), dec(4)]
    for mauvais in (-1, 2, 99):
        _leve(
            lambda m=mauvais: jouer_chemin(decisions, b, m, risque_par_trade_pct=0.01),
            ValueError,
            f"départ {mauvais}",
        )
    _leve(
        lambda: jouer_chemin([], b, 0, risque_par_trade_pct=0.01),
        ValueError,
        "aucune décision",
    )
    print(" 14. un départ hors bornes est REFUSÉ, pas rendu comme chemin vide censuré  ✓")


def check_15_decisions_non_triees_refusees() -> None:
    """
    On ne retrie pas en silence : l'ordre réel est une donnée du problème (H11, F14), et un tri
    automatique masquerait une séquence mal construite en amont.

    ⚠️ **Les trois décisions tiennent toutes dans la fenêtre**, et c'est délibéré. Ma première
    version utilisait `[dec(4), dec(1)]` : la fenêtre démarrait alors à la barre 4 et `dec(1)`
    tombait en dehors, ce qui levait une `ValueError` — mais celle de la barre introuvable, pas
    celle du tri. La preuve passait donc **même avec la garde désactivée**, et la mutation ne
    ressortait pas. D'où le contrôle du message dans `_leve`.
    """
    b = panel([GAGNE, GAGNE, GAGNE])
    _leve(
        lambda: jouer_chemin([dec(1), dec(7), dec(4)], b, 0, risque_par_trade_pct=0.01),
        ValueError,
        "décisions non triées",
        contient="pas triées",
    )
    print(" 15. des décisions non triées sont REFUSÉES, jamais retriées en silence  ✓")


def check_16_chemin_censure_quand_les_donnees_manquent() -> None:
    """
    Un challenge qui n'a pas de quoi conclure ressort `EN_COURS`, et l'agrégat le range en
    censuré — ni gagné, ni perdu. C'est la situation normale d'une série trop courte.
    """
    b = panel([[PLAT] * 3])
    r = jouer_chemin([Decision(dt(1), LONG, 100.0, 101.0, 99.0)], b, 0, risque_par_trade_pct=0.01)
    if r.issue is not Issue.EN_COURS:
        _echec(f"série trop courte : attendu EN_COURS, reçu {r.issue.value}")
    d = agreger([r])
    if (d.n_censures, d.p_funded, d.p_censure) != (1, 0.0, 1.0):
        _echec(f"un unique chemin censuré doit donner p_censure=1 : {d}")
    print(" 16. une série trop courte produit un chemin CENSURÉ, ni gagné ni perdu  ✓")


def main() -> None:
    check_93_les_gardes_de_JOUER_CHEMIN_mordent()
    print("\n" + "=" * 78)
    print("PREUVE DE LA SIMULATION FULL-PATH — tâche 4.6 (F8, F13)")
    print("=" * 78 + "\n")

    print("  ── le comptage, et le sort des chemins censurés ──")
    check_1_comptage_exact()
    check_2_funded_et_bust_somment_a_un_sur_les_resolus()
    check_3_les_censures_sont_ecartees_du_denominateur()
    check_3bis_relation_exacte_entre_les_deux_bornes()
    check_4_avertissement_de_censure()
    check_4bis_le_piege_du_systeme_trop_lent()
    check_5_echantillon_mince_signale()
    check_6_zero_funded_nest_pas_un_arrondi()
    check_7_aucun_chemin()

    print("\n  ── les durées ──")
    check_8_medianes_separees_par_issue()
    check_9_pas_de_mediane_sans_chemin()
    check_10_censures_sans_duree()
    check_11_motifs_de_bust_ventiles()

    print("\n  ── la simulation full-path ──")
    check_12_depart_ecarte_le_passe()
    check_13_duree_en_journees_ftmo_pas_en_phases()
    check_14_departs_hors_bornes_refuses()
    check_15_decisions_non_triees_refusees()
    check_16_chemin_censure_quand_les_donnees_manquent()
    check_90_une_issue_INCONNUE_est_refusee()

    print("\n" + "=" * 78)
    print("LES 20 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_90_une_issue_INCONNUE_est_refusee() -> None:
    """
    ⚠️ Garde jamais vue refuser avant le 2026-09-20 (revue de la phase 8, passe 5).

    `agreger` partitionne les résultats en trois catégories et vérifie qu'elles couvrent
    **exactement** l'ensemble. Si une quatrième issue apparaissait un jour — un `Enum` élargi, une
    valeur lue d'un fichier — les proportions ne sommeraient plus à 1 et `P(funded)` serait faux
    **sans que rien ne plante**. La garde existait ; aucune preuve ne l'avait vue refuser.
    """
    from enum import Enum

    class IssueInconnue(Enum):
        AUTRE = "autre"

    sain = [faux(Issue.FUNDED), faux(Issue.BUST), faux(Issue.EN_COURS)]
    d = agreger(sain)
    if d.n_funded + d.n_bust + d.n_censures != d.n_chemins:
        _echec("la partition ne couvre pas l'ensemble sur des issues pourtant valides")
    # Une quatrième issue : elle n'appartient à aucune des trois catégories.
    intrus = faux(Issue.FUNDED)
    object.__setattr__(intrus, "issue", IssueInconnue.AUTRE)
    _leve(
        lambda: agreger([*sain, intrus]),
        RuntimeError,
        "une issue inconnue dans les résultats",
        "issue inconnue",
    )
    print(
        "  90. une quatrième issue fait LEVER l'agrégation : les proportions somment toujours à 1  ✓"
    )


def check_93_les_gardes_de_JOUER_CHEMIN_mordent() -> None:
    """
    Deux gardes qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    ⚠️ Une fenêtre vide ou un budget nul ne plantent pas plus bas : ils rendent un chemin qui
    n'a simplement rien joué, et `P(funded)` se calcule dessus comme si c'était un échec.
    """
    b = panel([GAGNE, GAGNE])
    decisions = [dec(1), dec(4)]
    # ① un départ au-delà de la dernière barre : la fenêtre de simulation est vide.
    # ⚠️ Le panel doit rester PLEIN : sur un panel vide, c'est la garde « Panel de barres vide »
    # qui lève — et avec un fragment mou (« vide »), la preuve passait sans jamais exercer la
    # garde visée. Trou trouvé par le test de mutation le 2026-09-21.
    _leve(
        lambda: jouer_chemin([dec(9_999)], b, 0, risque_par_trade_pct=0.01),
        ValueError,
        "départ au-delà de la dernière barre",
        "fenêtre de simulation est vide",
    )
    # ② un budget de journées nul
    _leve(
        lambda: jouer_chemin(decisions, b, 0, risque_par_trade_pct=0.01, budget_journees=0),
        ValueError,
        "budget de journées nul",
        "budget_journees",
    )
    # anti-vacuité : le même appel, bien formé, joue le chemin
    if jouer_chemin(decisions, b, 0, risque_par_trade_pct=0.01).n_trades == 0:
        _echec("anti-vacuité : aucun trade joué sur un chemin valide")
    print("  93. gardes de jouer_chemin : fenêtre vide, budget de journées nul  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
