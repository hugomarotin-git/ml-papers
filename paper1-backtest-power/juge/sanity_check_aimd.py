"""
Preuve du sizing AIMD v2 — tâches 5.3, 5.4 et 5.5.
Décisions H1, H2, H4, H9, H10, H14, H15, H16.

Quarante-cinq vérifications. Ce module ne plante jamais : une erreur de sizing produit un compte
qui survit un peu plus ou un peu moins, donc un `P(funded)` **plausible et faux**. Tout se prouve
ici sur des journées fabriquées à la main. Le **site d'appel**, lui, est pinné du côté de
`sanity_check_simulateur` depuis la tâche 5.6 — cinq mutations y veillent.

Quatre propriétés comptent plus que les autres :

- **« nulle » n'est pas « sans trade »** (checks 4 et 5). Les deux valent zéro. Les confondre
  remet le compteur de danger à zéro sur une journée vide, et rien ne le signale ;
- **une journée perdante isolée ne coupe jamais** (check 12). C'est toute l'hystérésis : à 42,8 %
  de journées perdantes, couper sur une seule, c'est réagir au bruit ;
- **additif à la hausse, multiplicatif à la baisse** (checks 8 et 13). ⚠️ Ces deux checks partent
  volontairement d'un `m` **différent de 1** et d'un plafond **non atteint** : depuis `m = 1`,
  `+0,25` et `x1,25` donnent le même résultat, et `x0,5` et `-0,5` aussi. Une preuve écrite au
  point neutre ne distingue **rien** — elle passerait sur un code qui confond les deux formes ;
- **le passage de phase n'a jamais classé de journée** (check 19). La balance y saute au capital
  de départ : la classer fabriquerait une journée gagnante de plusieurs milliers de dollars.

Lancer :  .venv/bin/python juge/sanity_check_aimd.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import MISSING, fields
from datetime import date, timedelta

from aimd import TOLERANCE_BILAN_NUL, AIMDConfig, BilanJournee, SuiviAIMD, classer_journee
from regles_ftmo import ReglesFTMO


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """
    Vérifie qu'un garde-fou lève — **et pour la bonne raison** (piège n°4 du projet).

    Contrôler le seul type d'exception laisse passer une preuve décorative : une AUTRE erreur du
    même type, levée plus haut, la validerait sans que la garde visée existe.
    """
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(f"{quoi} : levée pour une AUTRE raison — « {str(e)[:90]} »")
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def cfg(
    risque_base: float = 0.01,
    hausse_additive: float = 0.25,
    seuil_journees_perdantes: int = 2,
    facteur_coupe: float = 0.5,
    m_initial: float = 1.0,
    m_min: float = 0.25,
    m_max: float = 1.25,
    reset_au_passage_de_phase: bool = True,
    airbag_n: int | None = None,
) -> AIMDConfig:
    """
    Une config d'essai. **Les valeurs par défaut sont ICI, dans la preuve, et nulle part ailleurs**
    — `AIMDConfig` n'en a aucune (H9, check 23).

    Les valeurs choisies sont le **centre de la grille** de la phase 6, pas des constantes du
    projet : elles viennent de la configuration AIMD de référence et n'ont aucune valeur chez nous tant que la
    calibration sur le twin-null n'a pas eu lieu (H5, A8). Un fixture de test n'est pas un défaut.
    """
    return AIMDConfig(
        risque_base=risque_base,
        hausse_additive=hausse_additive,
        seuil_journees_perdantes=seuil_journees_perdantes,
        facteur_coupe=facteur_coupe,
        m_initial=m_initial,
        m_min=m_min,
        m_max=m_max,
        reset_au_passage_de_phase=reset_au_passage_de_phase,
        airbag_n=airbag_n,
    )


def suivi(config: AIMDConfig | None = None, regles: ReglesFTMO | None = None) -> SuiviAIMD:
    """
    Un `SuiviAIMD` d'essai. `regles` est **requis** par le module (l'airbag y lit
    `capital_initial`) ; ce helper fournit le barème standard pour ne pas alourdir chaque check,
    mais **la classe, elle, n'a aucun défaut** — c'est le check 34 qui le vérifie.
    """
    return SuiviAIMD(
        config if config is not None else cfg(),
        regles if regles is not None else ReglesFTMO(),
    )


#: Une journée, au format `(variation de balance, nombre de trades CLÔTURÉS)`.
GAGNE: tuple[float, int] = (+100.0, 1)
PERD: tuple[float, int] = (-100.0, 1)
NUL: tuple[float, int] = (0.0, 1)  # des trades ont conclu, net à zéro
VIDE: tuple[float, int] = (0.0, 0)  # rien n'a conclu

BALANCE_DEPART: float = 50_000.0

#: Un lundi quelconque. Les journées d'essai sont fabriquées à partir de lui — jamais lues dans
#: les vraies données, ce module n'en a pas besoin.
JOUR_ZERO: date = date(2021, 1, 25)

#: Un compte confortable : marge énorme, donc airbag hors jeu. Sert aux checks qui parlent
#: du nominal et ne veulent pas que le plafond de survie vienne interférer.
LARGE_EQUITY: float = 50_000.0
LARGE_PLANCHER: float = 10_000.0


def jouer(suivi: SuiviAIMD, journees: list[tuple[float, int]]) -> float:
    """
    Joue une suite de journées FTMO et retourne la balance finale.

    Chaque journée est close par la frontière **suivante** — c'est le protocole réel (H2 ③), et
    chaque frontière annonce un jour **strictement postérieur**, comme le fera le simulateur.

    ⚠️ **La balance repart de celle du suivi, jamais de `BALANCE_DEPART`**, et ce détail a déjà
    produit une preuve fausse : une première version repartait de 50 000 à chaque appel, si bien
    qu'un second `jouer` sur un compte descendu à 49 800 fabriquait une variation de +200 — donc
    une journée **gagnante** là où le test croyait jouer une perdante. Deux checks passaient pour
    une raison qui n'avait rien à voir avec ce qu'ils annonçaient. C'est le piège n°4 du projet,
    rencontré dans le harnais au lieu du code.
    """
    if not suivi.journee_ouverte:
        suivi.frontiere_de_journee(JOUR_ZERO, BALANCE_DEPART)
    balance = suivi.balance_ouverture
    for variation, n_clotures in journees:
        for _ in range(n_clotures):
            suivi.marquer_cloture_de_trade()
        balance += variation
        suivi.frontiere_de_journee(suivi.jour_courant + timedelta(days=1), balance)
    return balance


# ── La classification d'une journée (H2 ①, H15) ──────────────────────────────────────────
def check_1_les_quatre_issues_existent() -> None:
    """H15 — quatre issues, pas trois. Chacune doit être atteignable."""
    obtenus = {
        classer_journee(+100.0, 1),
        classer_journee(-100.0, 1),
        classer_journee(0.0, 1),
        classer_journee(0.0, 0),
    }
    if obtenus != set(BilanJournee):
        _echec(
            f"les quatre issues ne sont pas toutes atteignables : {sorted(b.value for b in obtenus)}"
        )
    print("  1. les quatre issues du bilan existent et se distinguent  ✓")


def check_2_le_piege_du_zero_flottant() -> None:
    """
    Le piège que H15 signale : une journée dont les trades s'annulent **exactement** ne ressort
    presque jamais à `0.0` en flottant. Un `< 0` naïf en ferait une journée **perdante**, donc
    allongerait une série qui n'existe pas, donc couperait `m` sur du bruit numérique.
    """
    if classer_journee(-1e-11, 1) is not BilanJournee.NULLE:
        _echec("une variation de -1e-11 est classée perdante : le bruit flottant coupe le sizing")
    if classer_journee(+1e-11, 1) is not BilanJournee.NULLE:
        _echec("une variation de +1e-11 fait monter m : le bruit flottant récompense")
    # L'autre côté de la garde : un vrai cent n'est PAS du bruit.
    if classer_journee(-0.01, 1) is not BilanJournee.PERDANTE:
        _echec("une perte d'un cent est avalée par la tolérance : elle est trop large")
    if classer_journee(+0.01, 1) is not BilanJournee.GAGNANTE:
        _echec("un gain d'un cent est avalé par la tolérance : elle est trop large")
    print("  2. ±1e-11 est NUL, ±1 cent ne l'est pas : le zéro se teste des DEUX côtés  ✓")


def check_3_la_tolerance_est_encadree_des_deux_cotes() -> None:
    """
    La tolérance doit être **assez grande** pour avaler le bruit flottant d'une balance à 50 000,
    et **assez petite** pour ne jamais avaler un cent réel. On mesure les deux, on ne les suppose
    pas — c'est ce qui la distingue d'un nombre magique.
    """
    ulp = math.ulp(BALANCE_DEPART)  # le plus petit écart représentable à cette échelle
    marge = TOLERANCE_BILAN_NUL / ulp
    if marge < 1e6:
        _echec(
            f"la tolérance ({TOLERANCE_BILAN_NUL}) ne vaut que {marge:.0f} ULP à 50 000 : trop "
            "juste pour absorber une accumulation flottante"
        )
    if TOLERANCE_BILAN_NUL >= 0.01:
        _echec(
            f"la tolérance ({TOLERANCE_BILAN_NUL}) atteint le cent : elle avalerait un vrai "
            "résultat, et une journée gagnante deviendrait nulle"
        )
    # Et sur une journée réaliste : dix gains de 0,10 $ annulés par une perte de 1,00 $. La somme
    # exacte est zéro, la somme flottante ne l'est pas — et elle tombe du côté NÉGATIF, celui qui
    # ferait basculer la journée en « perdante » avec un test naïf.
    balance = BALANCE_DEPART
    for _ in range(10):
        balance += 0.10
    balance -= 1.00
    residu = balance - BALANCE_DEPART
    if residu == 0.0:
        _echec("le scénario ne produit plus de résidu flottant : la démonstration est devenue vide")
    if residu >= 0.0:
        _echec(f"le résidu {residu!r} n'est plus négatif : il n'exerce plus le cas dangereux")
    if classer_journee(residu, 10) is not BilanJournee.NULLE:
        _echec(f"dix +0,10 annulés par -1,00 donnent un résidu {residu!r} classé PERDANT")
    print(
        f"  3. tolérance encadrée : {marge:.0e} ULP de marge, sous le cent (résidu réel {residu:+.2e})  ✓"
    )


def check_4_nulle_nest_pas_sans_trade() -> None:
    """
    Le cœur de H15. **Même variation, deux issues** — et deux effets opposés sur la série de
    pertes. Si le code classait sur la seule variation de balance, les deux seraient confondues.
    """
    if classer_journee(0.0, 1) is not BilanJournee.NULLE:
        _echec("une journée tradée à net zéro doit être NULLE")
    if classer_journee(0.0, 0) is not BilanJournee.SANS_TRADE:
        _echec("une journée sans clôture doit être SANS_TRADE")

    # L'effet, pas seulement l'étiquette : la nulle casse la série, la vide ne la casse pas.
    a = suivi(cfg(seuil_journees_perdantes=2))
    jouer(a, [PERD, NUL, PERD, VIDE])
    if a.n_declenchements_coupe != 0:
        _echec("une journée NULLE n'a pas cassé la série de pertes")
    b = suivi(cfg(seuil_journees_perdantes=2))
    jouer(b, [PERD, VIDE, PERD, VIDE])
    if b.n_declenchements_coupe != 1:
        _echec(
            "une journée SANS_TRADE a cassé (ou prolongé) la série de pertes — H15 dit qu'elle est transparente"
        )
    print("  4. « nulle » ≠ « sans trade » : même zéro, effets opposés sur la série  ✓")


def check_5_labsence_de_cloture_prime_sur_la_variation() -> None:
    """
    L'ordre des tests EST la règle : une journée où **rien n'a conclu** est transparente, même si
    sa balance a bougé. C'est le cas du coût prélevé à l'ouverture d'un trade qui ne se ferme pas
    dans la journée (`cout_par_trade`, branché en 10.1) — **un coût n'est pas un résultat**.
    """
    if classer_journee(-7.0, 0) is not BilanJournee.SANS_TRADE:
        _echec(
            "une balance qui bouge sans aucune clôture est classée sur sa variation : un coût prélevé à l'ouverture deviendrait une journée perdante"
        )
    if classer_journee(+7.0, 0) is not BilanJournee.SANS_TRADE:
        _echec("idem à la hausse : sans clôture, il n'y a pas de résultat à créditer")
    print("  5. sans clôture, pas de résultat : la variation ne classe rien  ✓")


def check_6_les_entrees_impossibles_levent() -> None:
    """
    Deux entrées qui ne peuvent venir que d'un bug en amont, et qui seraient **muettes** sans
    garde.

    ⚠️ Le `NaN` est le plus dangereux des deux, et il a été trouvé en relisant le code APRÈS que
    les 23 preuves passaient : il échoue les deux comparaisons d'encadrement, donc il retombait
    sur « journée nulle ». Une balance corrompue aurait cassé les séries de pertes sans jamais
    rien signaler — un chiffre plausible produit par une donnée qui ne l'est pas. Les infinis
    suivent le même chemin (`inf - inf` vaut `NaN`).
    """
    for valeur in (float("nan"), float("inf"), float("-inf")):
        _leve(
            lambda v=valeur: classer_journee(v, 1),
            ValueError,
            f"une variation de balance {valeur!r}",
            contient="fini",
        )
    _leve(
        lambda: classer_journee(0.0, -1),
        ValueError,
        "un nombre de clôtures négatif",
        contient="négatif",
    )
    print("  6. NaN, ±inf et un compteur négatif lèvent — aucun ne devient une journée nulle  ✓")


# ── La mécanique : additive à la hausse, multiplicative à la baisse (H1, H4, H10) ─────────
def check_7_depart_a_m_zero_sans_ajustement() -> None:
    """H4 ① — le tout premier trade sort à `risque_base x m₀`, sans aucun ajustement préalable."""
    s = suivi(cfg(risque_base=0.01, m_initial=1.0, airbag_n=None))
    if s.m != 1.0:
        _echec(f"m au départ vaut {s.m}, attendu m₀ = 1.0 (plus de slow start depuis H4 amendée)")
    premier = s.risque_effectif(LARGE_EQUITY, LARGE_PLANCHER)
    if not math.isclose(premier, 0.01, rel_tol=1e-12):
        _echec(f"le premier trade sort à {premier}, attendu risque_base exactement")
    # Ouvrir une journée ne doit rien ajuster : la mise à jour se fait à la CLÔTURE (H2 ③).
    s.frontiere_de_journee(JOUR_ZERO, BALANCE_DEPART)
    if s.m != 1.0:
        _echec("ouvrir une journée a déjà bougé m : la mise à jour doit se faire à la clôture")
    print("  7. départ à m₀ = 1, et ouvrir une journée n'ajuste rien (H2 ③, H4 ①)  ✓")


def check_8_la_hausse_est_ADDITIVE() -> None:
    """
    ⚠️ **Ce check ne prouve quelque chose que hors du point neutre.** Depuis `m = 1`, une hausse
    additive de +0,25 et une hausse multiplicative de x1,25 donnent **le même 1,25** : une preuve
    écrite là ne distingue pas les deux formes. On part donc d'un `m` coupé à 0,5, plafond éloigné.
    """
    s = suivi(cfg(seuil_journees_perdantes=2, m_max=2.0, m_min=0.1))
    jouer(s, [PERD, PERD])
    if s.m != 0.5:
        _echec(f"m devait valoir 0.5 après une coupe, vaut {s.m}")
    jouer(s, [GAGNE])
    if s.m != 0.75:
        _echec(
            f"après une journée gagnante depuis m=0.5, m vaut {s.m} — attendu 0.75 (ADDITIF). "
            "0.625 signalerait une hausse multiplicative, 1.0 un retour au départ"
        )
    print("  8. la hausse est ADDITIVE, prouvé hors du point neutre (0,5 → 0,75)  ✓")


def check_9_le_risque_suit_m_et_rien_dautre() -> None:
    """
    H10 — l'AIMD n'ajuste QUE le pourcentage de risque, par `risque_base x m`.

    Vérifié à travers `risque_effectif` (airbag désactivé) et non par la propriété privée : c'est
    le **contrat public** qui compte, et depuis 5.4 il n'y en a qu'un seul.
    """
    s = suivi(cfg(risque_base=0.008, m_max=2.0, airbag_n=None))
    avant = s.risque_effectif(LARGE_EQUITY, LARGE_PLANCHER)
    if not math.isclose(avant, 0.008 * s.m, rel_tol=1e-12):
        _echec(f"le risque vaut {avant}, attendu risque_base x m = {0.008 * s.m}")
    jouer(s, [GAGNE])
    apres = s.risque_effectif(LARGE_EQUITY, LARGE_PLANCHER)
    if not math.isclose(apres, 0.008 * s.m, rel_tol=1e-12):
        _echec("le risque ne suit pas m")
    if apres <= avant:
        _echec("le risque n'a pas suivi la montée de m")
    print(f"  9. risque = risque_base x m ({avant:.4f} → {apres:.4f}), et rien d'autre  ✓")


def check_10_le_plafond_tient() -> None:
    """Le plafond est appliqué par `min`, donc l'accumulation flottante ne peut pas le franchir."""
    s = suivi(cfg(hausse_additive=0.1, m_max=1.25))
    jouer(s, [GAGNE] * 40)
    if s.m > 1.25:
        _echec(f"m a dépassé son plafond : {s.m!r} > 1.25")
    if not math.isclose(s.m, 1.25, rel_tol=1e-12):
        _echec(f"m devait atteindre son plafond après 40 journées gagnantes, vaut {s.m}")
    print("  10. le plafond de m tient, même après 40 hausses accumulées  ✓")


def check_11_le_plancher_tient() -> None:
    s = suivi(cfg(seuil_journees_perdantes=2, m_min=0.25))
    jouer(s, [PERD] * 40)
    if s.m < 0.25:
        _echec(f"m est passé sous son plancher : {s.m!r} < 0.25")
    if not math.isclose(s.m, 0.25, rel_tol=1e-12):
        _echec(f"m devait atteindre son plancher après 40 journées perdantes, vaut {s.m}")
    print("  11. le plancher de m tient, même après 20 coupes  ✓")


# ── L'hystérésis (H14) ───────────────────────────────────────────────────────────────────
def check_12_une_perdante_isolee_ne_coupe_JAMAIS() -> None:
    """
    Le cœur de H14, et la raison d'être de toute la mécanique. Sur le singe, **42,8 %** des
    journées sont perdantes et la journée contient 1,62 trade : une journée perdante isolée est
    l'événement le plus banal du système. Couper dessus, c'est réagir au bruit.
    """
    s = suivi(cfg(seuil_journees_perdantes=2))
    jouer(s, [PERD, GAGNE] * 15)
    if s.n_declenchements_coupe != 0:
        _echec(
            f"{s.n_declenchements_coupe} coupe(s) sur 15 journées perdantes ISOLÉES : "
            "l'hystérésis est morte, le sizing réagit au bruit"
        )
    # Et le même essai avec une journée nulle comme séparateur, qui casse aussi la série.
    t = suivi(cfg(seuil_journees_perdantes=2))
    jouer(t, [PERD, NUL] * 15)
    if t.n_declenchements_coupe != 0:
        _echec("une journée NULLE ne casse pas la série : 15 perdantes isolées ont coupé")
    print("  12. une journée perdante ISOLÉE ne coupe jamais (15 essais, deux séparateurs)  ✓")


def check_13_la_coupe_est_MULTIPLICATIVE() -> None:
    """
    ⚠️ Même précaution qu'au check 8 : depuis `m = 1`, `x0,5` et `-0,5` donnent tous deux 0,5. On
    coupe donc depuis `m = 0,75`, où les deux formes divergent (0,375 contre 0,25).
    """
    s = suivi(cfg(seuil_journees_perdantes=2, m_max=2.0, m_min=0.1, facteur_coupe=0.5))
    jouer(s, [PERD, PERD, GAGNE])
    if s.m != 0.75:
        _echec(f"m devait valoir 0.75 avant la seconde coupe, vaut {s.m}")
    jouer(s, [PERD, PERD])
    if s.m != 0.375:
        _echec(
            f"après une coupe depuis m=0.75, m vaut {s.m} — attendu 0.375 (MULTIPLICATIF). "
            "0.25 signalerait une coupe additive"
        )
    print("  13. la coupe est MULTIPLICATIVE, prouvé hors du point neutre (0,75 → 0,375)  ✓")


def check_14_le_seuil_k_est_respecte() -> None:
    """`k` journées perdantes consécutives, ni une de moins ni une de plus."""
    for k in (2, 3, 4):
        s = suivi(cfg(seuil_journees_perdantes=k, m_min=0.01))
        jouer(s, [PERD] * (k - 1))
        if s.n_declenchements_coupe != 0:
            _echec(f"k={k} : coupe déclenchée après seulement {k - 1} journées perdantes")
        jouer(s, [PERD])
        if s.n_declenchements_coupe != 1:
            _echec(f"k={k} : aucune coupe après {k} journées perdantes consécutives")
    print("  14. le seuil k se déclenche à la kᵉ journée perdante, pas avant (k = 2, 3, 4)  ✓")


def check_15_le_compteur_repart_a_zero_apres_une_coupe() -> None:
    """
    H14, lecture (a) — tranchée par Hugo le 2026-08-15. Après une coupe, il faut `k` NOUVELLES
    journées perdantes. Sans ce reset, un seuil à 2 dégénère en « coupe à chaque journée perdante
    dès la 2ᵉ » : exactement la réaction au bruit que l'hystérésis venait de supprimer.
    """
    s = suivi(cfg(seuil_journees_perdantes=2, m_min=0.01))
    jouer(s, [PERD, PERD])  # 1ʳᵉ coupe
    if s.journees_perdantes_consecutives != 0:
        _echec(f"le compteur vaut {s.journees_perdantes_consecutives} après une coupe, attendu 0")
    jouer(s, [PERD])  # la 3ᵉ perdante d'affilée ne doit PAS couper
    if s.n_declenchements_coupe != 1:
        _echec(
            "la journée perdante suivant une coupe a coupé de nouveau : le compteur n'est pas "
            "reparti à zéro (lecture (b), écartée)"
        )
    jouer(s, [PERD])  # la 4ᵉ, elle, coupe
    if s.n_declenchements_coupe != 2:
        _echec("deux nouvelles journées perdantes après une coupe n'ont pas recoupé")
    print("  15. après une coupe, le compteur repart à 0 : il faut k nouvelles perdantes  ✓")


def check_16_deux_k_journees_pour_atteindre_le_plancher() -> None:
    """
    La conséquence gravée : avec le reset de H14, atteindre le plancher depuis `m₀` demande **2k**
    journées perdantes consécutives — 4 à k=2, 6 à k=3, **8 à k=4**. Or la plus longue série
    observée sur tout le dev fait **8 journées** : à k=4, la seconde coupe est un événement unique
    dans l'historique, donc **cet axe de la grille est quasi inerte** (à constater en 6.4).
    """
    for k in (2, 3, 4):
        s = suivi(cfg(seuil_journees_perdantes=k, m_initial=1.0, facteur_coupe=0.5, m_min=0.25))
        jouer(s, [PERD] * (2 * k - 1))
        if s.m == 0.25:
            _echec(f"k={k} : le plancher est atteint en {2 * k - 1} journées, attendu {2 * k}")
        jouer(s, [PERD])
        if s.m != 0.25:
            _echec(
                f"k={k} : m vaut {s.m} après {2 * k} journées perdantes, attendu le plancher 0.25"
            )
    print("  16. le plancher de m demande 2k journées perdantes consécutives (4 / 6 / 8)  ✓")


# ── L'état : structure, indépendance, resets (H4 ②③) ─────────────────────────────────────
def check_17_deux_suivis_sont_independants() -> None:
    """
    H4 ② — la garantie est **structurelle**, pas déclarative. Le jour où deux chemins partagent un
    état, `P(funded)` cesse d'être une probabilité pour devenir un artefact d'ordonnancement — et
    le chiffre reste parfaitement plausible.
    """
    config = cfg()
    a, b = suivi(config), suivi(config)
    jouer(a, [PERD, PERD, PERD, PERD])
    if b.m != config.m_initial:
        _echec(f"le second suivi a hérité de m={b.m} : un état est partagé entre chemins")
    if b.journees_perdantes_consecutives != 0 or sum(b.bilans.values()) != 0:
        _echec("le second suivi a hérité du compteur ou du diagnostic du premier")
    if a.bilans is b.bilans:
        _echec("les deux suivis partagent le MÊME dictionnaire de bilans")
    print("  17. deux SuiviAIMD sur la même config sont totalement indépendants  ✓")


def check_18_passage_de_phase_selon_le_champ() -> None:
    """H4 ③ — la lecture est un champ explicite, à MESURER en phase 6, pas à choisir au jugé."""
    avec = suivi(cfg(seuil_journees_perdantes=2, reset_au_passage_de_phase=True))
    jouer(avec, [PERD, PERD, PERD])
    avec.passage_de_phase(avec.jour_courant, BALANCE_DEPART)
    if avec.m != avec.config.m_initial or avec.journees_perdantes_consecutives != 0:
        _echec(
            f"reset=True : m={avec.m}, compteur={avec.journees_perdantes_consecutives}, attendu m₀ et 0"
        )

    sans = suivi(cfg(seuil_journees_perdantes=2, reset_au_passage_de_phase=False))
    jouer(sans, [PERD, PERD, PERD])
    m_avant, compteur_avant = sans.m, sans.journees_perdantes_consecutives
    sans.passage_de_phase(sans.jour_courant, BALANCE_DEPART)
    if sans.m != m_avant or sans.journees_perdantes_consecutives != compteur_avant:
        _echec("reset=False : l'état a quand même été remis à zéro, le champ est décoratif")
    if m_avant == sans.config.m_initial:
        _echec("l'essai ne discrimine pas : m valait déjà m₀ avant le passage de phase")
    print("  18. le passage de phase remet l'état, ou non, selon le champ (H4 ③, F17)  ✓")


def check_19_le_passage_de_phase_ne_classe_JAMAIS_de_journee() -> None:
    """
    Le bug conçu-dehors. Au passage de phase, le simulateur remet la balance au capital de départ
    (F17) : la variation depuis l'ouverture de la journée n'est plus un résultat de trading mais
    un **saut comptable** de plusieurs milliers de dollars. La classer ferait monter ou couper `m`
    sans qu'aucun trade n'ait eu lieu, sans rien planter, et `P(funded)` resterait crédible.
    """
    s = suivi(cfg(reset_au_passage_de_phase=False, m_max=2.0))
    jouer(s, [GAGNE])  # m bouge une fois, pour que le check discrimine

    # Une journée s'ouvre bas, un trade s'y clôture, puis la phase 2 remet le compte à 50 000.
    # ⚠️ L'instantané se prend APRÈS l'ouverture de la journée basse : c'est cette frontière-là
    # qui classe la journée précédente, et la confondre avec l'effet du passage de phase ferait
    # échouer le check pour la mauvaise raison.
    s.frontiere_de_journee(s.jour_courant + timedelta(days=1), 47_000.0)
    s.marquer_cloture_de_trade()
    m_avant, bilans_avant = s.m, sum(s.bilans.values())
    s.passage_de_phase(s.jour_courant, 50_000.0)

    if sum(s.bilans.values()) != bilans_avant:
        _echec("le passage de phase a classé une journée : le saut de capital en a fabriqué une")
    if s.m != m_avant:
        _echec(f"m est passé de {m_avant} à {s.m} au passage de phase, sans aucun trade")
    if s.n_montees != 1:
        _echec(f"n_montees vaut {s.n_montees} : le saut comptable a été compté comme une montée")
    print(
        "  19. le passage de phase ABANDONNE la journée : +3 000 $ ne fabriquent pas une gagnante  ✓"
    )


def check_20_la_garde_de_progression_de_journee() -> None:
    """
    **La moitié « structurelle » de la double protection, décidée par Hugo le 2026-08-16.**

    Le pire bug du branchement (5.6) est une ligne au mauvais niveau d'indentation : prévenir
    l'AIMD à chaque **barre** au lieu de chaque **frontière de journée**. Il est parfaitement muet
    — une barre sans clôture donne « sans trade » donc transparente, une barre où un trade se
    ferme donne « gagnante » ou « perdante ». L'AIMD réagirait donc **trade par trade**, soit
    exactement le déclencheur que H2 a écarté (*« à 1,62 trade/jour, c'est du bruit pur »*).

    Une mutation de site d'appel le pinnera aussi en 5.7 — mais elle ne protège que **cette**
    version du code, tandis que la garde tient quelle que soit la version.
    """
    s = suivi(cfg())
    jouer(s, [GAGNE])
    aujourdhui = s.jour_courant

    # ① la même journée deux fois — le cas « appelé à chaque barre »
    _leve(
        lambda: s.frontiere_de_journee(aujourdhui, 50_000.0),
        RuntimeError,
        "la même journée FTMO annoncée deux fois",
        contient="strictement avancer",
    )
    # ② une journée qui RECULE — un panel mal trié mélangerait deux journées dans un bilan
    _leve(
        lambda: s.frontiere_de_journee(aujourdhui - timedelta(days=1), 50_000.0),
        RuntimeError,
        "une journée FTMO qui recule",
        contient="strictement avancer",
    )
    # ③ un passage de phase annoncé sur une AUTRE journée que celle ouverte
    _leve(
        lambda: s.passage_de_phase(aujourdhui + timedelta(days=1), 50_000.0),
        RuntimeError,
        "un passage de phase hors de la journée ouverte",
        contient="journée ouverte",
    )
    # Et le cas légitime passe toujours : la garde n'est pas un mur.
    if s.frontiere_de_journee(aujourdhui + timedelta(days=1), 50_000.0) is None:
        _echec("une frontière strictement postérieure doit classer la journée écoulée")
    print("  20. la journée doit STRICTEMENT avancer : l'appel par barre lève immédiatement  ✓")


def check_21_cloture_hors_journee_leve() -> None:
    s = suivi(cfg())
    _leve(
        s.marquer_cloture_de_trade,
        RuntimeError,
        "une clôture signalée sans journée ouverte",
        contient="frontiere_de_journee",
    )
    print("  21. signaler une clôture sans journée ouverte lève (message vérifié)  ✓")


def check_22_premiere_et_derniere_journee() -> None:
    """
    La toute première frontière n'a **rien** à classer, et la dernière journée d'une simulation
    n'est **jamais** classée : aucune frontière ne la suit. C'est un no-op par construction (`m`
    n'influence que les trades suivants, H2 ③), mais il faut le savoir en lisant le diagnostic de
    5.6 — la somme de `bilans` vaut une journée de moins que les journées vécues.
    """
    s = suivi(cfg())
    if s.frontiere_de_journee(JOUR_ZERO, BALANCE_DEPART) is not None:
        _echec("la première frontière a classé une journée qui n'existe pas")
    s.marquer_cloture_de_trade()
    if sum(s.bilans.values()) != 0 or s.m != s.config.m_initial:
        _echec("une journée en cours a déjà été comptée avant sa frontière de clôture")
    bilan = s.frontiere_de_journee(JOUR_ZERO + timedelta(days=1), BALANCE_DEPART + 100.0)
    if bilan is not BilanJournee.GAGNANTE or sum(s.bilans.values()) != 1:
        _echec(f"la journée close n'a pas été classée gagnante : {bilan}")
    print(
        "  22. la 1ʳᵉ frontière ne classe rien, la journée en cours n'est comptée qu'à sa clôture  ✓"
    )


# ── H9 et le diagnostic ──────────────────────────────────────────────────────────────────
def check_23_aucun_defaut_dans_la_config() -> None:
    """
    H9 — **aucun défaut fourni pour aucun paramètre de sizing**. `risque_base` domine le résultat
    et un airbag désactivé par défaut serait un filet absent sans que rien ne le signale : un
    champ qui peut être oublié finira par l'être.
    """
    avec_defaut = [
        f.name
        for f in fields(AIMDConfig)
        if f.default is not MISSING or f.default_factory is not MISSING
    ]
    if avec_defaut:
        _echec(f"AIMDConfig fournit un défaut pour {avec_defaut} — H9 l'interdit")
    _leve(AIMDConfig, TypeError, "AIMDConfig construite sans aucun argument")
    print(f"  23. AIMDConfig : aucun défaut sur ses {len(fields(AIMDConfig))} champs (H9)  ✓")


def check_24_le_diagnostic_distingue_lactif_de_linerte() -> None:
    """
    Sans ce diagnostic, la phase 6 ne distingue pas une config qui protège d'une config inerte
    (6.4). Deux asymétries voulues : au plafond, une gagnante n'est **pas** une montée ; au
    plancher, une coupe se **déclenche** sans faire bouger `m`.
    """
    haut = suivi(cfg(hausse_additive=0.25, m_max=1.25))
    jouer(haut, [GAGNE] * 10)
    if haut.n_montees != 1:
        _echec(
            f"n_montees vaut {haut.n_montees} : les gagnantes au plafond sont comptées comme des montées"
        )

    bas = suivi(cfg(seuil_journees_perdantes=2, m_initial=1.0, m_min=0.25))
    jouer(bas, [PERD] * 12)  # 6 déclenchements, mais le plancher est atteint dès le 2ᵉ
    if bas.n_declenchements_coupe != 6:
        _echec(f"n_declenchements_coupe vaut {bas.n_declenchements_coupe}, attendu 6")
    if bas.n_coupes != 2:
        _echec(f"n_coupes vaut {bas.n_coupes}, attendu 2 — seules les coupes EFFECTIVES comptent")
    if bas.n_declenchements_coupe < bas.n_coupes:
        _echec("invariant cassé : n_declenchements_coupe doit toujours être >= n_coupes")

    tally = suivi(cfg())
    journees = [GAGNE, PERD, NUL, VIDE, PERD, VIDE]
    jouer(tally, journees)
    if sum(tally.bilans.values()) != len(journees):
        _echec(f"{sum(tally.bilans.values())} journées comptées sur {len(journees)} closes")
    if tally.bilans[BilanJournee.SANS_TRADE] != 2 or tally.bilans[BilanJournee.NULLE] != 1:
        _echec(
            f"le décompte par issue est faux : { ({b.value: n for b, n in tally.bilans.items()}) }"
        )
    print("  24. le diagnostic sépare l'effectif du déclenché, et compte les 4 issues  ✓")


# ── L'airbag de survie (H16) — tâche 5.4 ─────────────────────────────────────────────────
def check_25_airbag_desactive_ne_touche_a_rien() -> None:
    """
    `airbag_n=None` désactive, et la désactivation est **explicite** (H9) : le champ n'a aucun
    défaut, donc elle est forcément écrite. Un filet absent par distraction ne se voit nulle part.
    """
    s = suivi(cfg(risque_base=0.02, airbag_n=None))
    # Marge minuscule : un airbag actif mordrait violemment ici.
    r = s.risque_effectif(equity=45_100.0, plancher=45_000.0)
    if r != 0.02:
        _echec(f"airbag désactivé, le risque vaut {r} au lieu du nominal 0.02")
    if s.n_declenchements_airbag != 0:
        _echec("un airbag désactivé a compté un déclenchement")
    print("  25. airbag désactivé : le nominal passe intact, même à 100 $ du plancher  ✓")


def check_26_le_scenario_de_H16_en_dollars() -> None:
    """
    **Le cas qui a fait naître l'airbag** (H16) : compte à 46 000, barrière à 45 000, il reste
    1 000 $. Deux bonnes journées viennent de remonter `m`, donc le nominal veut risquer 1 000 $ —
    **un seul trade perdant et le challenge est fini**.

    On vérifie la sémantique de `N` **en dollars**, pas la formule en fraction : après plafond,
    `capital_initial x risque x N` doit valoir **exactement la marge**. C'est la définition même
    de « combien de trades perdants au minimum pour mourir », et c'est ce qui attrape une erreur
    d'unité — la formule de H16 compare une fraction à des dollars.
    """
    regles = ReglesFTMO()
    s = suivi(cfg(risque_base=0.02, m_initial=1.0, airbag_n=2), regles)
    nominal_en_dollars = regles.capital_initial * 0.02
    if nominal_en_dollars != 1_000.0:
        _echec(
            f"le scénario ne reproduit pas H16 : nominal {nominal_en_dollars} $ au lieu de 1 000"
        )

    r = s.risque_effectif(equity=46_000.0, plancher=45_000.0)
    perte_au_stop = regles.capital_initial * r
    if not math.isclose(perte_au_stop * 2, 1_000.0, rel_tol=1e-12):
        _echec(
            f"après airbag N=2, la perte au stop vaut {perte_au_stop} $ : il faudrait "
            f"{1_000.0 / perte_au_stop:.2f} trades perdants pour mourir, attendu exactement 2"
        )
    if s.n_declenchements_airbag != 1:
        _echec("l'airbag a mordu sans le compter")
    print(
        f"  26. scénario H16 : 1 000 $ voulus → {perte_au_stop:.0f} $, soit N=2 trades pour mourir  ✓"
    )


def check_27_hors_jeu_quand_la_marge_est_large() -> None:
    """
    H16 : *« il est conditionnel par construction »*. Au matin d'une journée à 50 000, la marge
    vaut 2 500 $ (F2) ; à risque nominal modeste, l'airbag ne doit **rien** faire — sinon ce
    n'est plus un filet de survie mais un second mécanisme de sizing qui double l'AIMD.
    """
    regles = ReglesFTMO()
    s = suivi(cfg(risque_base=0.01, m_initial=1.25, airbag_n=2), regles)
    plancher = regles.plancher_journalier(50_000.0)
    if plancher != 47_500.0:
        _echec(f"le plancher journalier du matin vaut {plancher}, attendu 47 500 (F2)")
    r = s.risque_effectif(equity=50_000.0, plancher=plancher)
    if r != 0.0125:
        _echec(f"l'airbag a mordu sur une marge de 2 500 $ : risque {r} au lieu de 0.0125")
    if s.n_declenchements_airbag != 0:
        _echec("déclenchement compté alors que l'airbag n'a pas mordu")
    print("  27. marge du matin (2 500 $) : l'airbag est hors jeu, comme H16 l'annonce  ✓")


def check_28_cest_un_plafond_jamais_un_plancher() -> None:
    """
    Un `max` à la place du `min` transformerait le filet en **accélérateur** : sur un compte
    confortable, l'airbag remonterait le risque au lieu de le borner. Rien ne planterait.
    """
    s = suivi(cfg(risque_base=0.002, m_initial=1.0, airbag_n=2))
    r = s.risque_effectif(equity=60_000.0, plancher=45_000.0)  # marge 15 000 $, plafond 15 %
    if r != 0.002:
        _echec(f"le risque a été REMONTÉ à {r} : l'airbag agit comme un plancher, pas un plafond")
    print("  28. l'airbag ne remonte JAMAIS le risque : c'est un plafond  ✓")


def check_29_il_ne_double_pas_laimd() -> None:
    """
    H16 : *« quand `m` a déjà coupé, il est encore plus loin de mordre »*. Même marge, même
    config : si l'AIMD a réduit `m`, le nominal passe sous le plafond et l'airbag se tait.
    C'est ce qui garantit qu'ils ne se superposent pas.
    """
    haut = suivi(cfg(risque_base=0.02, m_initial=1.0, seuil_journees_perdantes=2, airbag_n=2))
    haut.risque_effectif(equity=46_000.0, plancher=45_000.0)
    if haut.n_declenchements_airbag != 1:
        _echec("l'airbag devait mordre à m = 1")

    bas = suivi(
        cfg(risque_base=0.02, m_initial=1.0, seuil_journees_perdantes=2, m_min=0.25, airbag_n=2)
    )
    jouer(bas, [PERD, PERD])  # coupe : m passe à 0,5, donc nominal 1 %
    if bas.m != 0.5:
        _echec(f"m devait valoir 0.5 après la coupe, vaut {bas.m}")
    r = bas.risque_effectif(equity=46_000.0, plancher=45_000.0)
    if bas.n_declenchements_airbag != 0 or r != 0.01:
        _echec(f"après coupe de m, l'airbag mord encore (risque {r}) : les deux se superposent")
    print("  29. après une coupe de m, l'airbag ne mord plus : il ne double pas l'AIMD  ✓")


def check_30_marge_nulle_ou_negative_leve() -> None:
    """
    **Le pire scénario silencieux de la tâche.** Une marge négative donne un plafond négatif,
    donc un risque négatif, donc un nombre d'unités négatif : la position partirait dans le
    **sens inverse** de la décision. Rien ne planterait, et le compte **gagnerait sur ses pires
    trades** — un P(funded) flatteur et complètement faux.
    """
    s = suivi(cfg(airbag_n=2))
    _leve(
        lambda: s.risque_effectif(equity=44_000.0, plancher=45_000.0),
        RuntimeError,
        "une marge négative",
        contient="position inversée",
    )
    _leve(
        lambda: s.risque_effectif(equity=45_000.0, plancher=45_000.0),
        RuntimeError,
        "une marge exactement nulle",
        contient="position inversée",
    )
    for valeur in (float("nan"), float("inf")):
        _leve(
            lambda v=valeur: s.risque_effectif(equity=v, plancher=45_000.0),
            ValueError,
            f"une equity {valeur!r}",
            contient="fini",
        )
    print("  30. marge nulle, négative ou non finie : lève — jamais de risque négatif  ✓")


def check_31_N_divise_reellement_le_plafond() -> None:
    """`N` se lit « combien de trades perdants au minimum pour mourir » : le plafond doit lui être
    inversement proportionnel. Sinon l'axe « 2 / 3 » de la grille phase 6 ne sépare rien."""
    regles = ReglesFTMO()
    risques = {}
    for n in (2, 3):
        s = suivi(cfg(risque_base=0.03, m_initial=1.0, airbag_n=n), regles)
        risques[n] = s.risque_effectif(equity=46_000.0, plancher=45_000.0)
    if not math.isclose(risques[2] / risques[3], 1.5, rel_tol=1e-12):
        _echec(f"N=2 donne {risques[2]}, N=3 donne {risques[3]} : le rapport n'est pas 3/2")
    print(
        f"  31. le plafond est bien divisé par N ({risques[2]:.5f} à N=2, {risques[3]:.5f} à N=3)  ✓"
    )


def check_32_capital_initial_est_LU_jamais_recopie() -> None:
    """
    Piège n°5 — *on importe, on ne recopie pas*. Un `50_000` en dur ici plafonnerait le risque
    **cinq fois trop haut** sur un compte 10k, et le seul symptôme serait un compte qui meurt
    plus souvent que le filet ne le promet.
    """
    petit = ReglesFTMO(capital_initial=10_000.0)
    s = suivi(cfg(risque_base=0.03, m_initial=1.0, airbag_n=2), petit)
    r = s.risque_effectif(equity=9_200.0, plancher=9_000.0)  # marge 200 $
    # 200 / (2 x 10 000) = 1 %. Avec un 50 000 recopié on lirait 0,2 %.
    if not math.isclose(r, 0.01, rel_tol=1e-12):
        _echec(f"risque {r} sur un compte 10k, attendu 0.01 — `capital_initial` est recopié")
    perte = petit.capital_initial * r
    if not math.isclose(perte * 2, 200.0, rel_tol=1e-12):
        _echec(f"la perte au stop vaut {perte} $ : N=2 ne tient pas sur un compte 10k")
    print("  32. `capital_initial` est lu depuis ReglesFTMO : le plafond suit un compte 10k  ✓")


def check_33_le_nominal_est_prive() -> None:
    """
    Exigence posée en 5.3 et tenue en 5.4 : `risque_effectif` est le **seul** accès public au
    risque. Un site d'appel qui lirait le nominal contournerait l'airbag **en silence**, et le
    compte mourrait sur un trade que le filet était censé retenir.
    """
    s = suivi(cfg(airbag_n=2))
    if hasattr(s, "risque_nominal"):
        _echec("`risque_nominal` est redevenu public : l'airbag est contournable sans le savoir")
    public = [n for n in dir(s) if not n.startswith("_") and "risque" in n]
    if public != ["risque_effectif"]:
        _echec(f"accès publics au risque : {public}, attendu ['risque_effectif'] et lui seul")
    print("  33. `risque_effectif` est le SEUL accès public au risque (nominal privé)  ✓")


def check_34_aucun_defaut_meme_pour_lairbag() -> None:
    """
    H9, revérifié après l'ajout de `airbag_n` : *« un airbag désactivé par défaut serait un filet
    absent sans que rien ne le signale »*. Et `regles` est **requis** côté `SuiviAIMD`, contrairement
    à `SuiviChallenge` — un barème implicite plafonnerait un compte 10k comme un 50k.
    """
    avec_defaut = [
        f.name
        for f in fields(AIMDConfig)
        if f.default is not MISSING or f.default_factory is not MISSING
    ]
    if avec_defaut:
        _echec(f"AIMDConfig fournit un défaut pour {avec_defaut} — H9 l'interdit")
    _leve(
        lambda: SuiviAIMD(cfg()),  # pyright: ignore[reportCallIssue]
        TypeError,
        "SuiviAIMD construit sans barème",
    )
    print("  34. aucun défaut : ni pour `airbag_n` (H9), ni pour le barème de `SuiviAIMD`  ✓")


# ── La validation de la config (H9) — tâche 5.5 ──────────────────────────────────────────
def _refuse(quoi: str, contient: str, attendu: type[Exception] = ValueError, **champs) -> None:
    """Construire une `AIMDConfig` avec ces champs doit lever, **et pour la bonne raison**."""
    _leve(lambda: cfg(**champs), attendu, quoi, contient=contient)


def check_35_risque_base_est_une_fraction() -> None:
    for valeur in (0.0, 1.0, 1.5, -0.01):
        _refuse(f"risque_base={valeur}", "fraction", risque_base=valeur)
    print("  35. risque_base hors ]0, 1[ refusé : ni sizing nul, ni compte joué en entier  ✓")


def check_36_hausse_additive_strictement_positive() -> None:
    """À 0, `m` ne peut plus que descendre : l'AIMD dégénère et rien ne le signale."""
    for valeur in (0.0, -0.25):
        _refuse(f"hausse_additive={valeur}", "ne peut", hausse_additive=valeur)
    print(
        "  36. hausse_additive <= 0 refusée : une AIMD qui ne monte jamais n'est plus une AIMD  ✓"
    )


def check_37_le_seuil_k_vaut_au_moins_deux() -> None:
    """H14 — *« une journée perdante isolée ne coupe JAMAIS »*. À k=1, l'hystérésis n'existe plus."""
    for valeur in (1, 0, -2):
        _refuse(f"k={valeur}", "hystérésis", seuil_journees_perdantes=valeur)
    print("  37. k < 2 refusé : sans hystérésis, le sizing coupe sur du bruit (H14)  ✓")


def check_38_facteur_coupe_est_une_fraction() -> None:
    """Au-dessus de 1, la « coupe » **augmente** `m` : le frein devient un accélérateur."""
    for valeur in (1.0, 1.5, 0.0, -0.5):
        _refuse(f"facteur_coupe={valeur}", "accélérateur", facteur_coupe=valeur)
    print("  38. facteur_coupe hors ]0, 1[ refusé : au-dessus de 1 le frein accélère  ✓")


def check_39_bornes_de_m_et_la_config_DEGENEREE() -> None:
    """
    ⚠️ **L'égalité `m_min == m_max` doit PASSER.** C'est la config dégénérée qui reproduit le
    sizing constant au centime près — et c'est un **critère de recette de 5.6**. La refuser
    casserait le seul contrôle qui prouvera que brancher le sizing n'a rien changé au passage.
    """
    for valeur in (0.0, -0.5):
        _refuse(f"m_min={valeur}", "inverse", m_min=valeur)
    _refuse("m_min > m_max", "dégénérée", m_min=1.5, m_max=1.0, m_initial=1.2)

    degeneree = cfg(m_min=1.0, m_max=1.0, m_initial=1.0, airbag_n=None)
    s = suivi(degeneree)
    jouer(s, [GAGNE] * 5 + [PERD] * 6)
    if s.m != 1.0:
        _echec(f"la config dégénérée a bougé : m = {s.m}, attendu 1.0 fixe")
    if s.risque_effectif(LARGE_EQUITY, LARGE_PLANCHER) != degeneree.risque_base:
        _echec("la config dégénérée ne reproduit pas le sizing constant")
    print("  39. m_min <= 0 et bornes croisées refusés ; m_min == m_max ACCEPTÉ (dégénérée)  ✓")


def check_40_m_initial_dans_les_bornes() -> None:
    _refuse("m_initial sous m_min", "m_initial", m_initial=0.1, m_min=0.25, m_max=1.25)
    _refuse("m_initial au-dessus de m_max", "m_initial", m_initial=2.0, m_min=0.25, m_max=1.25)
    print("  40. m_initial hors [m_min, m_max] refusé : sinon `m` se recale en silence  ✓")


def check_41_airbag_n_au_moins_un_ou_None() -> None:
    """La désactivation s'écrit `None` (H9), jamais par une valeur qui a l'air neutre."""
    for valeur in (0, -1):
        _refuse(f"airbag_n={valeur}", "explicite", airbag_n=valeur)
    if cfg(airbag_n=None).airbag_n is not None:
        _echec("`airbag_n=None` doit rester None : c'est la désactivation explicite")
    if cfg(airbag_n=1).airbag_n != 1:
        _echec("airbag_n=1 est faible mais légitime : il borne le risque à la marge entière")
    print("  41. airbag_n < 1 refusé ; `None` est la SEULE désactivation admise (H9)  ✓")


def check_42_les_compteurs_sont_des_ENTIERS() -> None:
    """
    Un `k` flottant ne planterait pas : la comparaison marcherait et la coupe tomberait une
    journée plus tard. La grille de la phase 6 étant construite par programme, un `2.0` peut
    arriver sans qu'on l'ait voulu.
    """
    _refuse("k=2.5", "entier", TypeError, seuil_journees_perdantes=2.5)
    _refuse("k=2.0 (flottant)", "entier", TypeError, seuil_journees_perdantes=2.0)
    _refuse("airbag_n=2.5", "entier", TypeError, airbag_n=2.5)
    _refuse("k=True (bool)", "entier", TypeError, seuil_journees_perdantes=True)
    print("  42. k et airbag_n doivent être des entiers vrais (ni flottant, ni booléen)  ✓")


def check_43_le_NaN_est_refuse_sur_CHAQUE_champ_flottant() -> None:
    """
    **Le piège n°8 transposé à la validation, et il tient à la FORME du test.**

    Ces deux lignes se lisent pareil et ne font pas la même chose ::

        if not 0 < x < 1:       # NaN REFUSÉ  : `0 < nan` est False, donc `not` déclenche
        if x <= 0 or x >= 1:    # NaN ACCEPTÉ : les DEUX comparaisons sont False

    On le vérifie d'abord **sur les comparaisons elles-mêmes**, pour que la propriété soit
    démontrée et pas seulement affirmée, puis on exige un refus sur **chaque** champ flottant.

    Ce qu'un `NaN` accepté produirait : un nombre d'unités `NaN`, donc une balance `NaN`, donc
    **plus aucun bust jamais détecté** — toute comparaison avec `NaN` est fausse — et un
    `P(funded)` effondré sans raison lisible.
    """
    nan = float("nan")
    if (nan <= 0.0) or (nan >= 1.0):
        _echec("la disjonction attraperait le NaN : la démonstration n'a plus d'objet")
    if 0.0 < nan < 1.0:
        _echec("l'encadrement accepte le NaN : la garde ne peut pas fonctionner")

    # Un champ par entrée, écrit en clair plutôt que par dépaquetage de dictionnaire : le
    # dépaquetage prive pyright de tout contrôle de type, et c'est justement lui qui attrape la
    # classe d'erreur de l'incident du 2026-08-03.
    champs_flottants = (
        ("risque_base", lambda: cfg(risque_base=nan)),
        ("hausse_additive", lambda: cfg(hausse_additive=nan)),
        ("facteur_coupe", lambda: cfg(facteur_coupe=nan)),
        ("m_initial", lambda: cfg(m_initial=nan)),
        ("m_min", lambda: cfg(m_min=nan)),
        ("m_max", lambda: cfg(m_max=nan)),
    )
    for champ, construire in champs_flottants:
        try:
            construire()
        except ValueError, TypeError:
            continue
        _echec(
            f"un NaN passe dans `{champ}` : la borne est écrite en disjonction, pas en encadrement"
        )
    print("  43. un NaN est refusé sur les 6 champs flottants (la FORME de la borne le décide)  ✓")


def check_44_la_garde_de_plafond_de_risque() -> None:
    """
    Le seul contrôle qui a besoin du barème : `risque_base x m_max` doit rester **strictement**
    sous le plancher journalier. À égalité, un trade perdant amène l'équity **sur** le plancher —
    et `observer` buste au plancher (`<=`, classe F18) : la config serait mortelle tout en
    paraissant à la limite du légal.

    Et la borne est **LUE** dans `ReglesFTMO` : on le prouve en changeant le barème, pas en
    relisant le code. Un `0.05` recopié survivrait à un changement de règlement.
    """
    trop = cfg(risque_base=0.04, m_max=1.25)  # 5,00 % pile : mortel
    _leve(
        lambda: SuiviAIMD(trop, ReglesFTMO()),
        ValueError,
        "risque_base x m_max exactement au plancher",
        contient="un seul trade perdant",
    )
    # Juste en dessous : accepté.
    suivi(cfg(risque_base=0.039, m_max=1.25))

    # La borne suit le barème. Sur un règlement à 10 % de perte journalière, 6,25 % passe.
    large = ReglesFTMO(plancher_journalier_pct=0.09)
    suivi(cfg(risque_base=0.05, m_max=1.25), large)
    # Et sur un règlement plus serré, une config jusque-là légale devient refusée.
    serre = ReglesFTMO(plancher_journalier_pct=0.01)
    _leve(
        lambda: SuiviAIMD(cfg(risque_base=0.01, m_max=1.25), serre),
        ValueError,
        "une config légale à 5 % mais illégale à 1 %",
        contient="plancher journalier",
    )
    print("  44. risque_base x m_max < plancher journalier, borne LUE depuis ReglesFTMO  ✓")


def check_45_chaque_refus_a_son_PROPRE_message() -> None:
    """
    H9 exige que chaque valeur absurde soit refusée **pour sa propre raison**. Un message générique
    ferait exactement ce que le piège n°4 décrit : une preuve passerait sur la mauvaise erreur, et
    on croirait avoir vérifié une garde qui n'existe pas.

    On collecte ici les messages de toutes les fautes possibles et on vérifie qu'ils sont **tous
    distincts**. C'est un contrôle systématique, pas un échantillon.
    """
    fautes: dict[str, dict] = {
        "risque_base": {"risque_base": 0.0},
        "hausse_additive": {"hausse_additive": 0.0},
        "seuil_type": {"seuil_journees_perdantes": 2.5},
        "seuil_valeur": {"seuil_journees_perdantes": 1},
        "facteur_coupe": {"facteur_coupe": 1.0},
        "m_min": {"m_min": 0.0},
        "bornes_croisees": {"m_min": 1.5, "m_max": 1.0, "m_initial": 1.2},
        "m_initial": {"m_initial": 2.0},
        "airbag_type": {"airbag_n": 2.5},
        "airbag_valeur": {"airbag_n": 0},
    }
    messages: dict[str, str] = {}
    for nom, champs in fautes.items():
        try:
            cfg(**champs)
        except (ValueError, TypeError) as e:
            messages[nom] = str(e)
        else:
            _echec(f"la faute « {nom} » n'est pas refusée du tout")
    if len(set(messages.values())) != len(fautes):
        doublons = [n for n in messages if list(messages.values()).count(messages[n]) > 1]
        _echec(f"messages identiques pour {doublons} : les refus ne sont pas discriminants")
    print(f"  45. les {len(fautes)} refus ont {len(fautes)} messages DISTINCTS (H9, piège n°4)  ✓")


def main() -> None:
    check_93_les_invariants_INTERNES_du_suivi_mordent()
    print("\n" + "=" * 78)
    print("PREUVE DU NOYAU AIMD v2 — tâche 5.3 (H1, H2, H4, H9, H10, H14, H15)")
    print("=" * 78 + "\n")

    print("  ── la classification d'une journée (H2 ①, H15) ──")
    check_1_les_quatre_issues_existent()
    check_2_le_piege_du_zero_flottant()
    check_3_la_tolerance_est_encadree_des_deux_cotes()
    check_4_nulle_nest_pas_sans_trade()
    check_5_labsence_de_cloture_prime_sur_la_variation()
    check_6_les_entrees_impossibles_levent()

    print("\n  ── la mécanique : additive à la hausse, multiplicative à la baisse (H1, H10) ──")
    check_7_depart_a_m_zero_sans_ajustement()
    check_8_la_hausse_est_ADDITIVE()
    check_9_le_risque_suit_m_et_rien_dautre()
    check_10_le_plafond_tient()
    check_11_le_plancher_tient()

    print("\n  ── l'hystérésis (H14) ──")
    check_12_une_perdante_isolee_ne_coupe_JAMAIS()
    check_13_la_coupe_est_MULTIPLICATIVE()
    check_14_le_seuil_k_est_respecte()
    check_15_le_compteur_repart_a_zero_apres_une_coupe()
    check_16_deux_k_journees_pour_atteindre_le_plancher()

    print("\n  ── l'état : structure, indépendance, resets (H4 ②③) ──")
    check_17_deux_suivis_sont_independants()
    check_18_passage_de_phase_selon_le_champ()
    check_19_le_passage_de_phase_ne_classe_JAMAIS_de_journee()
    check_20_la_garde_de_progression_de_journee()
    check_21_cloture_hors_journee_leve()
    check_22_premiere_et_derniere_journee()

    print("\n  ── H9 et le diagnostic ──")
    check_23_aucun_defaut_dans_la_config()
    check_24_le_diagnostic_distingue_lactif_de_linerte()

    print("\n  \u2500\u2500 l'airbag de survie (H16) \u2500\u2500")
    check_25_airbag_desactive_ne_touche_a_rien()
    check_26_le_scenario_de_H16_en_dollars()
    check_27_hors_jeu_quand_la_marge_est_large()
    check_28_cest_un_plafond_jamais_un_plancher()
    check_29_il_ne_double_pas_laimd()
    check_30_marge_nulle_ou_negative_leve()
    check_31_N_divise_reellement_le_plafond()
    check_32_capital_initial_est_LU_jamais_recopie()
    check_33_le_nominal_est_prive()
    check_34_aucun_defaut_meme_pour_lairbag()

    print("\n  \u2500\u2500 la validation de la config (H9) \u2500\u2500")
    check_35_risque_base_est_une_fraction()
    check_36_hausse_additive_strictement_positive()
    check_37_le_seuil_k_vaut_au_moins_deux()
    check_38_facteur_coupe_est_une_fraction()
    check_39_bornes_de_m_et_la_config_DEGENEREE()
    check_40_m_initial_dans_les_bornes()
    check_41_airbag_n_au_moins_un_ou_None()
    check_42_les_compteurs_sont_des_ENTIERS()
    check_43_le_NaN_est_refuse_sur_CHAQUE_champ_flottant()
    check_44_la_garde_de_plafond_de_risque()
    check_45_chaque_refus_a_son_PROPRE_message()

    print("\n" + "=" * 78)
    print("LES 46 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_les_invariants_INTERNES_du_suivi_mordent() -> None:
    """
    Trois gardes qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    ⚠️ Ce sont des **invariants d'ordre d'appel**, pas des validations d'entrée : lire l'état de
    la journée avant d'en avoir ouvert une rendrait un `m` calculé sur une ancre absente. Elles
    lèvent `RuntimeError`, pas `ValueError` — l'appelant a tort, pas la donnée.
    """
    s = suivi()
    _leve(lambda: s.jour_courant, RuntimeError, "jour courant sans journée ouverte", "frontiere")
    _leve(
        lambda: s.balance_ouverture,
        RuntimeError,
        "balance d'ouverture sans journée ouverte",
        "frontiere",
    )
    _leve(
        lambda: s.passage_de_phase(date(2021, 3, 1), 50_000.0),
        RuntimeError,
        "passage de phase hors journée ouverte",
        "frontiere",
    )
    # anti-vacuité : une fois la journée ouverte, les trois marchent
    s.frontiere_de_journee(date(2021, 3, 1), balance=50_000.0)
    if s.jour_courant != date(2021, 3, 1) or s.balance_ouverture != 50_000.0:
        _echec("anti-vacuité : l'état n'est pas lisible une fois la journée ouverte")
    s.passage_de_phase(date(2021, 3, 1), 50_000.0)  # LA journée ouverte, pas la suivante
    print("  93. invariants du suivi AIMD : jour, balance d'ouverture, passage de phase  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
