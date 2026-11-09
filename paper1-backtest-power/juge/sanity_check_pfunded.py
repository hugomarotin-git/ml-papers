"""
Preuve du comportement de P(funded) — tâche 4.8, décisions F0, F9, F13.

Les preuves précédentes vérifiaient la **mécanique** : un stop ferme bien un trade, un plancher
ferme bien un compte. Celle-ci vérifie autre chose — que le chiffre produit **réagit
correctement à une structure qu'on a plantée exprès**.

C'est la différence entre « le code fait ce qu'il dit » et « le code mesure ce qu'on croit ».
Un P(funded) peut être mécaniquement irréprochable et rester aveugle au clustering des pertes,
ou insensible au taux de réussite. Rien ne le signalerait : le nombre resterait entre 0 et 1.

────────────────────────────────────────────────────────────────────────────────────────
Ce que chaque famille de preuves plante, et ce qu'elle attend
────────────────────────────────────────────────────────────────────────────────────────
1. **Un edge connu** → P(funded) doit croître avec lui, sans exception. C'est le contrôle
   positif du juge, dans l'esprit d'A10 : si la chaîne ne voit pas un edge planté à la main,
   elle ne verra jamais un edge réel.
2. **Des mauvaises passes forcées** → le risque de bust doit monter.
3. **La taille de bloc** → sur une série qui contient du clustering, elle doit **changer le
   verdict** (F9). Et — c'est le point le plus important de ce fichier — **sur une série sans
   clustering, elle ne doit presque rien changer**. Sans ce second test, le premier ne prouve
   rien : on ne saurait pas si l'effet vient du clustering ou de la taille de bloc elle-même.

Les seuils sont **relatifs**, jamais absolus : on vérifie des ordres et des sens de variation,
pas des valeurs. Une preuve qui fixerait « P(funded) = 34 % » se casserait au premier
raffinement du simulateur sans rien avoir prouvé de plus.

Lancer :  .venv/bin/python juge/sanity_check_pfunded.py
"""

from __future__ import annotations

import itertools
import random
import sys

from methodes_pfunded import TradeRealise, methode_b, plus_longue_serie_perdante
from regles_ftmo import ReglesFTMO

R = ReglesFTMO()
RISQUE = 0.01
N_CHEMINS = 300


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def serie(taux_reussite: float, n: int, seed: int, par_journee: int = 1) -> list[TradeRealise]:
    """
    Une série de trades à taux de réussite imposé, RR 1:1.

    `par_journee` groupe les trades : à 1 chaque trade ouvre sa journée FTMO, à 3 ils vont par
    paquets de trois dans la même journée — donc partagent le même plancher journalier.

    ⚠️ Le nombre de gains est **exact**, pas tiré au sort : `round(taux x n)` gains, puis
    mélange. Ma première version tirait chaque trade indépendamment, si bien que le taux
    *réalisé* variait de plusieurs points d'une seed à l'autre — et ce bruit se confondait avec
    ce que les preuves mesurent. À 50 % annoncés, deux seeds donnaient 25 % et 74 % de P(funded).
    """
    n_gains = round(taux_reussite * n)
    rendements = [1.0] * n_gains + [-1.0] * (n - n_gains)
    random.Random(seed).shuffle(rendements)
    return [
        TradeRealise(rendement=r, nouvelle_journee=i % par_journee == 0)
        for i, r in enumerate(rendements)
    ]


def serie_par_regimes(n_bons: int, n_mauvais: int, n_cycles: int) -> list[TradeRealise]:
    """
    Une série **volontairement clusterisée** : des paquets de gains, puis des paquets de pertes.

    Le taux de réussite global est le même qu'une série mélangée de mêmes proportions — seule
    la *disposition* change. C'est exactement ce que la taille de bloc doit savoir capturer.
    """
    trades: list[TradeRealise] = []
    for _ in range(n_cycles):
        trades += [TradeRealise(1.0, True) for _ in range(n_bons)]
        trades += [TradeRealise(-1.0, True) for _ in range(n_mauvais)]
    return trades


def melanger(trades: list[TradeRealise], seed: int) -> list[TradeRealise]:
    """Mêmes trades, mêmes proportions, **clustering détruit**. Le contrôle du check 6."""
    copie = list(trades)
    random.Random(seed).shuffle(copie)
    return [TradeRealise(t.rendement, True) for t in copie]


def p_funded(trades: list[TradeRealise], bloc: int, seed: int = 1) -> float:
    return methode_b(trades, RISQUE, taille_bloc=bloc, n_chemins=N_CHEMINS, seed=seed).p_funded


def p_bust(trades: list[TradeRealise], bloc: int, seed: int = 1) -> float:
    return methode_b(trades, RISQUE, taille_bloc=bloc, n_chemins=N_CHEMINS, seed=seed).p_bust


# ── 1. Le contrôle positif : P(funded) suit l'edge planté ─────────────────────────────────
def check_1_pfunded_croit_avec_le_taux_de_reussite() -> None:
    """
    **Le contrôle positif du juge** (esprit A10). On plante un edge connu et croissant ; le
    chiffre doit suivre. S'il ne suit pas, la chaîne est aveugle et un NO-GO ne voudra rien dire.
    """
    mesures = []
    for taux in (0.40, 0.50, 0.60, 0.70):
        t = serie(taux, 200, seed=int(taux * 100))
        mesures.append((taux, p_funded(t, bloc=max(8, plus_longue_serie_perdante(t)))))

    for (t1, p1), (t2, p2) in itertools.pairwise(mesures):
        if p2 <= p1:
            _echec(
                f"P(funded) ne croît pas avec l'edge : {100 * t1:.0f} % -> {100 * p1:.1f} % puis "
                f"{100 * t2:.0f} % -> {100 * p2:.1f} %. Le juge est aveugle à l'edge"
            )
    detail = " · ".join(f"{100 * t:.0f}%->{100 * p:.0f}%" for t, p in mesures)
    print(f"  1. P(funded) croît strictement avec le taux de réussite  ({detail})  ✓")


def check_2_un_systeme_sans_edge_nest_PAS_a_50() -> None:
    """
    **F0 rendu mesurable.** À 50 % de réussite et RR 1:1, l'espérance est nulle — mais
    P(funded) est **bien en dessous de 50 %**, parce qu'un first-passage pénalise la variance :
    il faut franchir +10 % puis +5 % SANS toucher les planchers en chemin.

    C'est le fondement de I13 : un trade à espérance nulle n'est pas neutre, il fait baisser
    P(funded). Si cette preuve tombe, le juge ne mesure pas un first-passage.
    """
    t = serie(0.50, 200, seed=11)
    p = p_funded(t, bloc=max(8, plus_longue_serie_perdante(t)))
    if p >= 0.50:
        _echec(
            f"P(funded) = {100 * p:.1f} % à espérance nulle : un first-passage doit pénaliser la "
            "variance, donc rester nettement sous 50 %"
        )
    print(f"  2. à espérance nulle, P(funded) = {100 * p:.1f} % — bien sous 50 % (F0, I13)  ✓")


def check_3_edge_ecrasant_finance_presque_toujours() -> None:
    """La borne haute : si presque tout gagne, le compte doit être financé presque toujours."""
    t = serie(0.95, 200, seed=3)
    p = p_funded(t, bloc=max(3, plus_longue_serie_perdante(t)))
    if p < 0.90:
        _echec(f"P(funded) = {100 * p:.1f} % avec 95 % de réussite : la borne haute est cassée")
    print(f"  3. edge écrasant (95 %) -> P(funded) = {100 * p:.0f} %  ✓")


def check_4_systeme_desespere_ne_finance_jamais() -> None:
    t = serie(0.05, 200, seed=4)
    d = methode_b(
        t, RISQUE, taille_bloc=max(20, plus_longue_serie_perdante(t)), n_chemins=100, seed=4
    )
    if d.p_funded > 0.02:
        _echec(f"P(funded) = {100 * d.p_funded:.1f} % avec 5 % de réussite : impossible")
    if d.p_bust < 0.90:
        _echec(f"P(bust) = {100 * d.p_bust:.1f} % : un système désespéré doit buster")
    print(f"  4. système désespéré (5 %) -> P(funded) = {100 * d.p_funded:.1f} %  ✓")


# ── 2. Les mauvaises passes forcées ──────────────────────────────────────────────────────
def check_5_mauvaises_passes_forcees_font_monter_le_bust() -> None:
    """
    Deux séries de **même taux de réussite global**. L'une a ses pertes dispersées, l'autre les
    a concentrées en paquets. Le risque de bust doit être plus élevé dans la seconde.

    C'est la propriété que le juge existe pour capturer : à P&L total identique, l'ordre décide.
    """
    concentree = serie_par_regimes(n_bons=14, n_mauvais=6, n_cycles=10)  # 70 % de réussite
    dispersee = melanger(concentree, seed=5)

    bloc = max(plus_longue_serie_perdante(concentree), plus_longue_serie_perdante(dispersee))
    b_conc = p_bust(concentree, bloc=bloc, seed=5)
    b_disp = p_bust(dispersee, bloc=bloc, seed=5)

    if b_conc <= b_disp:
        _echec(
            f"pertes concentrées P(bust)={100 * b_conc:.1f} % <= dispersées {100 * b_disp:.1f} % : "
            "le juge ne voit pas le clustering des pertes, alors que c'est sa raison d'être"
        )
    print(
        f"  5. mauvaises passes forcées : P(bust) {100 * b_disp:.0f} % -> {100 * b_conc:.0f} %  ✓"
    )


# ── 3. La sensibilité à la taille de bloc, ET son contrôle ───────────────────────────────
def check_6_la_taille_de_bloc_COMPTE_sur_une_serie_clusterisee() -> None:
    """
    **F9 rendu mesurable — et ma première attente était fausse.**

    J'avais écrit que P(bust) devait croître *monotonement* avec la taille de bloc. La mesure dit
    le contraire : sur une série faite de cycles « 14 gains puis 6 pertes », le bust culmine
    quand le bloc vaut la longueur de la mauvaise passe, puis **redescend** quand le bloc
    dépasse la longueur du cycle — un bloc de 36 sur des cycles de 20 contient forcément les
    deux régimes, donc il les moyenne et dilue le clustering.

    **F9 est un PLANCHER, pas un « plus c'est long, mieux c'est ».** Un bloc trop court ne peut
    pas reproduire la pire passe ; un bloc trop long la noie dans le reste. Ce qu'on vérifie ici
    est donc que la taille de bloc **change réellement le verdict** sur une série structurée —
    le check 7 vérifiant qu'elle ne change rien quand il n'y a rien à capturer.
    """
    t = serie_par_regimes(n_bons=14, n_mauvais=6, n_cycles=10)
    mini = plus_longue_serie_perdante(t)
    mesures = [(b, p_bust(t, bloc=b, seed=6)) for b in (mini, mini * 3, mini * 6)]
    ecart = max(p for _, p in mesures) - min(p for _, p in mesures)

    if ecart < 0.02:
        _echec(
            f"série CLUSTERISÉE : P(bust) ne bouge que de {100 * ecart:.1f} points avec la taille "
            f"de bloc ({mesures}). Le bootstrap est alors aveugle au clustering, ce qui est "
            "précisément ce que la méthode B existe pour capturer"
        )
    if mesures[0][1] < max(p for _, p in mesures):
        _echec(
            f"le bloc au PLANCHER F9 ({mini}) devrait être le plus sévère, or {mesures}. "
            "Si un bloc plus long durcit encore, le plancher n'est pas au bon endroit"
        )
    detail = " · ".join(f"bloc {b}->{100 * p:.0f}%" for b, p in mesures)
    print(f"  6. série CLUSTERISÉE : la taille de bloc change le verdict  ({detail})  ✓")


def check_7_bloc_plus_long_sur_serie_SANS_clustering_ne_change_presque_rien() -> None:
    """
    **Le contrôle du check 6, et le plus important des deux.**

    Sur une série sans structure, la taille de bloc ne doit presque rien changer : il n'y a rien
    à préserver. Sans ce test, le check 6 ne prouverait rien — on ne saurait pas si l'effet
    observé vient du **clustering** ou de la taille de bloc elle-même (par exemple d'un artefact
    de bord, ou d'une longueur de séquence qui change).

    C'est la même logique que les tests négatifs du filtre intraday (B12) : montrer que l'outil
    REFUSE de réagir quand il n'y a rien à voir.
    """
    t = melanger(serie_par_regimes(n_bons=14, n_mauvais=6, n_cycles=10), seed=7)
    mini = plus_longue_serie_perdante(t)
    mesures = [(b, p_bust(t, bloc=b, seed=7)) for b in (mini, mini * 3, mini * 6)]
    ecart = max(p for _, p in mesures) - min(p for _, p in mesures)

    if ecart > 0.15:
        _echec(
            f"série SANS clustering : P(bust) varie de {100 * ecart:.0f} points avec la taille de "
            f"bloc ({mesures}). L'effet du check 6 ne peut alors pas être attribué au clustering"
        )
    detail = " · ".join(f"bloc {b}->{100 * p:.0f}%" for b, p in mesures)
    print(f"  7. série SANS clustering : le bloc ne change presque rien  ({detail})  ✓")


# ── 4. Stabilité et bornes ───────────────────────────────────────────────────────────────
def check_8_estimation_stable_entre_seeds() -> None:
    """
    Une proportion sur 300 chemins doit être reproductible d'un tirage à l'autre. Si deux seeds
    donnent des réponses très différentes, `n_chemins` est trop petit et tout le reste du
    fichier mesure du bruit.
    """
    t = serie(0.55, 200, seed=8)
    bloc = max(8, plus_longue_serie_perdante(t))
    valeurs = [p_funded(t, bloc=bloc, seed=s) for s in (101, 202, 303)]
    ecart = max(valeurs) - min(valeurs)
    if ecart > 0.12:
        _echec(
            f"P(funded) varie de {100 * ecart:.0f} points entre seeds ({valeurs}) : "
            f"{N_CHEMINS} chemins ne suffisent pas, le reste du fichier mesure du bruit"
        )
    print(f"  8. estimation stable entre seeds (écart {100 * ecart:.1f} points)  ✓")


def check_9_proportions_dans_les_bornes() -> None:
    for taux in (0.2, 0.5, 0.8):
        t = serie(taux, 120, seed=int(taux * 10))
        d = methode_b(
            t, RISQUE, taille_bloc=max(10, plus_longue_serie_perdante(t)), n_chemins=100, seed=9
        )
        for nom, v in (("p_funded", d.p_funded), ("p_bust", d.p_bust), ("p_censure", d.p_censure)):
            if not 0.0 <= v <= 1.0:
                _echec(f"{nom} = {v} hors de [0, 1] à {taux}")
        if d.n_funded + d.n_bust + d.n_censures != d.n_chemins:
            _echec("les trois comptages ne couvrent pas l'ensemble des chemins")
    print("  9. toutes les proportions restent dans [0, 1] et les comptages sont complets  ✓")


def check_10_le_risque_deplace_le_verdict() -> None:
    """
    À edge identique, monter le risque rapproche des planchers : P(bust) doit monter.

    C'est ce que l'AIMD exploitera (H1) — et si le juge n'y était pas sensible, toute la
    calibration de la phase 5 mesurerait du vide.
    """
    t = serie(0.55, 200, seed=10)
    bloc = max(8, plus_longue_serie_perdante(t))
    prudent = methode_b(t, 0.005, taille_bloc=bloc, n_chemins=N_CHEMINS, seed=10)
    agressif = methode_b(t, 0.04, taille_bloc=bloc, n_chemins=N_CHEMINS, seed=10)
    if agressif.p_bust <= prudent.p_bust:
        _echec(
            f"risque 0,5 % -> P(bust) {100 * prudent.p_bust:.1f} %, risque 4 % -> "
            f"{100 * agressif.p_bust:.1f} % : le juge est insensible au sizing"
        )
    print(
        f" 10. monter le risque monte P(bust) : {100 * prudent.p_bust:.0f} % -> "
        f"{100 * agressif.p_bust:.0f} %  ✓"
    )


def check_11_le_bruit_decroit_avec_le_nombre_de_chemins() -> None:
    """
    **Le chiffre que la phase 5 et la phase 6 devront connaître.**

    P(funded) est une proportion : son bruit d'échantillonnage décroît en `1/sqrt(n)`. Tant
    qu'on ne l'a pas mesuré, on ne sait pas quel écart entre deux configurations est réel et
    lequel est du tirage — or c'est exactement ce que fera la calibration AIMD (H5-H8) et la
    comparaison au twin-null (G2).

    On vérifie ici que le bruit **diminue** quand on ajoute des chemins. S'il ne diminuait pas,
    le bootstrap serait biaisé, pas bruité, et augmenter `n_chemins` ne servirait à rien.
    """
    t = serie(0.55, 200, seed=88)
    bloc = max(8, plus_longue_serie_perdante(t))
    seeds = (11, 22, 33, 44)

    etendues: list[tuple[int, float]] = []
    for n in (50, 400):
        valeurs = [
            methode_b(t, RISQUE, taille_bloc=bloc, n_chemins=n, seed=s).p_funded for s in seeds
        ]
        etendues.append((n, max(valeurs) - min(valeurs)))

    (n_petit, bruit_petit), (n_grand, bruit_grand) = etendues
    if bruit_grand >= bruit_petit:
        _echec(
            f"le bruit ne décroît pas : {n_petit} chemins -> {100 * bruit_petit:.1f} points, "
            f"{n_grand} chemins -> {100 * bruit_grand:.1f} points. Un estimateur bruité doit se "
            "resserrer quand on l'alimente ; s'il ne bouge pas, il est BIAISÉ, pas bruité"
        )
    print(
        f" 11. le bruit décroît avec les chemins : {n_petit} -> ±{100 * bruit_petit / 2:.1f} pts, "
        f"{n_grand} -> ±{100 * bruit_grand / 2:.1f} pts  ✓"
    )
    print(
        f"      ⚠️ à retenir pour les phases 5 et 6 : à {N_CHEMINS} chemins, un écart de moins "
        "de ~10 points entre deux configurations n'est pas interprétable."
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU COMPORTEMENT DE P(funded) — tâche 4.8 (F0, F9, F13)")
    print("=" * 78 + "\n")

    print("  ── contrôle positif : le juge voit-il un edge planté ? ──")
    check_1_pfunded_croit_avec_le_taux_de_reussite()
    check_2_un_systeme_sans_edge_nest_PAS_a_50()
    check_3_edge_ecrasant_finance_presque_toujours()
    check_4_systeme_desespere_ne_finance_jamais()

    print("\n  ── mauvaises passes forcées ──")
    check_5_mauvaises_passes_forcees_font_monter_le_bust()

    print("\n  ── sensibilité à la taille de bloc, et son CONTRÔLE (F9) ──")
    check_6_la_taille_de_bloc_COMPTE_sur_une_serie_clusterisee()
    check_7_bloc_plus_long_sur_serie_SANS_clustering_ne_change_presque_rien()

    print("\n  ── stabilité et bornes ──")
    check_8_estimation_stable_entre_seeds()
    check_9_proportions_dans_les_bornes()
    check_10_le_risque_deplace_le_verdict()
    check_11_le_bruit_decroit_avec_le_nombre_de_chemins()

    print("\n" + "=" * 78)
    print("LES 11 PREUVES PASSENT.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
