"""
Le twin-null générique — tâche 6.1, décisions G1, G8, G9.

**La barre à battre.** Le null est le *jumeau exact* du système réel — même instrument, mêmes
barrières, même horizon, même sizing, même cadence — et **seule la direction est un pile ou
face** (G1). Si une série sans aucun signal directionnel obtient déjà un bon `P(funded)`, alors
un bon `P(funded)` ne prouve rien.

────────────────────────────────────────────────────────────────────────────────────────
Pourquoi les entrées sont un ARGUMENT et non quelque chose que ce module fabrique
────────────────────────────────────────────────────────────────────────────────────────
**G9 : il y a DEUX twin-null, pas un**, et ce module les sert tous les deux — c'est la raison
d'être de sa signature.

1. **La table par cadence** (tâche 6.5), produite *avant* que le modèle existe : elle donne
   l'ordre de grandeur de la barre et sert à calibrer l'AIMD. C'est une **approximation**.
2. **Le twin-null final**, qui reçoit la **liste exacte des entrées du modèle** — et **lui seul
   donne le verdict**.

La différence n'est pas cosmétique. Le plancher de confiance (I13) rend la cadence irrégulière :
certains jours deux trades, d'autres aucun. Or deux trades le même jour **partagent le même
plancher de perte journalier**. « 780 trades groupés sur les jours que le modèle a aimés » n'a pas
le même risque que « 780 trades étalés à 0,76 par jour ». **La forme dans le temps compte, pas
seulement le nombre.**

⚠️ **Ce module ne tire JAMAIS les instants d'entrée.** Le null à entrées aléatoires est une brique
séparée, explicitement **non faite** (G8). Confondre les deux, c'est comparer le modèle à un
système qui ne lui ressemble plus.

────────────────────────────────────────────────────────────────────────────────────────
Rien n'est recalculé : le « jumeau exact » est STRUCTUREL
────────────────────────────────────────────────────────────────────────────────────────
Le prix d'entrée et les deux barrières sont **lus** dans le panel labellisé, jamais recalculés
(piège n°5 : *on importe, on ne recopie pas*). Trois conséquences, et chacune serait une décision
fragile si elle était écrite à la main :

- **les barrières sont identiques à celles du système réel**, au bit près. « Même distance » n'est
  pas une intention, c'est la même colonne ;
- **la symétrie et le RR 1:1 sont hérités** du label, qui les tient de `K_ATR` ;
- **l'absence de look-ahead sur l'ATR est héritée elle aussi** — ce module ne touche pas à l'ATR,
  donc il ne peut pas le regarder trop tôt. La propriété est prouvée là où elle se joue
  (`sanity_check_label`, où le look-ahead est testé en TRONQUANT réellement la série).

Recalculer les barrières ici aurait créé un second endroit où la largeur du triple-barrière est
décidée. Le jour où `K_ATR` change, le null aurait continué sur l'ancienne valeur et la barre à
battre aurait décrit un système qui n'existe plus — sans que rien ne plante.

────────────────────────────────────────────────────────────────────────────────────────
Ce qui est écarté, et pourquoi c'est compté
────────────────────────────────────────────────────────────────────────────────────────
Une entrée demandée sur une barre **sans label** est écartée : pas de barrières, donc pas de trade
possible. C'est le cas des barres non contiguës (C9) et du début de série où l'ATR n'existe pas
encore. Le système réel ne peut pas y entrer non plus — le null reste donc son jumeau.

**Les écartées sont comptées et leur motif rapporté.** Un null silencieusement amputé de 30 % de
ses entrées produirait une barre calculée sur une cadence qui n'est pas celle qu'on croit — et
G3 dit précisément que comparer un modèle à une barre calculée pour une autre fréquence est un
faux test.

────────────────────────────────────────────────────────────────────────────────────────
Ce que ce module NE fait pas
────────────────────────────────────────────────────────────────────────────────────────
- **Il ne restreint pas au dev.** ⛔ C'est à l'appelant d'appeler `periodes.assert_dev_seulement`
  avant de lui passer un panel. Le faire ici interdirait l'usage légitime du hold-out à la tâche
  10.5 — et cacherait la décision au lieu de la rendre visible.
- **Il n'ordonnance rien** : l'occupation du compte (un seul trade à la fois, F14) est le travail
  de `ordonnanceur`, qui tourne ensuite et ignore tout du sizing (H11).
- **Il ne stocke rien** (G5, A8) : tout se recalcule à seed identique.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from simulateur import LONG, SHORT, Decision

# Le pont vers `src/` est déjà branché par `chemins` (importé par `simulateur`) ; on le refait
# ici pour que ce module reste importable seul, sans dépendre de l'ordre des imports.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Les colonnes du panel labellisé dont ce module a besoin. Toutes viennent du label : il n'en
#: calcule aucune.
COLONNES_REQUISES: tuple[str, ...] = (
    "DateTime",
    "label_entree_dt",
    "prix_entree",
    "barriere_haute",
    "barriere_basse",
    "raison_sans_label",
)


@dataclass(frozen=True)
class TirageNull:
    """
    Le résultat d'un tirage : les décisions du singe, et ce qui a été écarté.

    Gelé : un tirage est une observation, pas un objet qu'on ajuste après coup.
    """

    decisions: list[Decision]
    seed: int
    n_demandes: int
    #: Motif -> nombre d'entrées écartées. Vide si tout est passé. **Rapporté, jamais tu** : un
    #: null amputé en silence produirait une barre pour une cadence qui n'est pas la sienne (G3).
    ecartes: dict[str, int] = field(default_factory=dict)

    @property
    def n_ecartes(self) -> int:
        return sum(self.ecartes.values())

    @property
    def part_long(self) -> float:
        """La part de longs. Doit tendre vers 0,5 — c'est la seule chose qui soit tirée (G1)."""
        if not self.decisions:
            return float("nan")
        return sum(1 for d in self.decisions if d.direction == LONG) / len(self.decisions)

    def resume(self) -> str:
        ecartes = ", ".join(f"{n} {m}" for m, n in sorted(self.ecartes.items())) or "aucune"
        return (
            f"twin-null seed={self.seed} · {len(self.decisions)} décisions sur "
            f"{self.n_demandes} entrées demandées · écartées : {ecartes} · "
            f"{self.part_long:.1%} de longs"
        )


@dataclass(frozen=True)
class BarreDuNull:
    """
    La barre à battre, lue sur un ensemble de `P(funded)` produits par le null (G2).

    **Le seuil est le p95, pas la médiane** : « le meilleur score qu'un singe chanceux atteint ».
    Battre la médiane, c'est seulement être dans le nuage du hasard.
    """

    p50: float
    p95: float
    n: int
    #: Combien de tirages dépassent **strictement** le p95. C'est la mesure qui dit si la barre
    #: repose sur la population ou sur une poignée d'extrêmes.
    #:
    #: ⚠️ La formulation naïve — « le p95 vaut-il le maximum ? » — ne marche pas : avec
    #: l'interpolation linéaire de `numpy`, le p95 tombe **entre** deux valeurs et n'égale jamais
    #: le maximum dès qu'il y a au moins deux tirages. Un tel drapeau ne se lèverait **jamais** :
    #: du code qu'aucun test ne peut exercer, donc une fausse confiance. Trouvé en écrivant 6.3.
    n_au_dessus: int

    @property
    def p95_repose_sur_les_extremes(self) -> bool:
        """
        Vrai quand **un seul tirage au plus** dépasse le p95.

        La barre ne décrit alors plus « le meilleur score qu'un singe chanceux atteint » (G2) mais
        **le singe le plus chanceux observé** — une statistique d'extrême, qui bougera de tout ce
        que le prochain tirage voudra. **Mesuré, pas supposé** : c'est à 6.5 de choisir un nombre
        de tirages qui rende ce drapeau faux, et de le **constater**.
        """
        return self.n_au_dessus <= 1

    def resume(self) -> str:
        alerte = (
            f"  ATTENTION : {self.n_au_dessus} tirage(s) au-dessus du p95, barre d'extrême"
            if self.p95_repose_sur_les_extremes
            else ""
        )
        return (
            f"barre du null sur {self.n} tirages : p50 {self.p50:.3f} · p95 {self.p95:.3f}{alerte}"
        )


def barre_du_null(p_fundeds: Sequence[float]) -> BarreDuNull:
    """
    Réduit un ensemble de `P(funded)` du null à la barre à battre (G2, G4).

    ⚠️ **À calculer AVANT de connaître le score du modèle** (G4) : la calculer après coup, c'est
    s'exposer à choisir la barre qui arrange.

    Rien n'est stocké (G5, A8) : la barre se recalcule à la demande.
    """
    valeurs = np.asarray(list(p_fundeds), dtype=float)
    if valeurs.size < 2:
        raise ValueError(
            f"Il faut au moins 2 tirages pour parler de dispersion, reçu {valeurs.size}. Un "
            "percentile sur un seul point est ce point : la barre serait un chiffre unique "
            "déguisé en statistique."
        )
    if not np.all(np.isfinite(valeurs)):
        # Un NaN traverserait `percentile` sans lever et rendrait la barre silencieusement fausse.
        raise ValueError(
            f"{int((~np.isfinite(valeurs)).sum())} valeur(s) non finie(s) parmi les P(funded). "
            "Un NaN se propagerait dans le percentile sans rien faire planter."
        )
    p50 = float(np.percentile(valeurs, 50))
    p95 = float(np.percentile(valeurs, 95))
    return BarreDuNull(
        p50=p50,
        p95=p95,
        n=int(valeurs.size),
        n_au_dessus=int((valeurs > p95).sum()),
    )


def tirer_directions(
    panel_labelle: pd.DataFrame,
    candidats: Sequence[datetime],
    seed: int,
) -> TirageNull:
    """
    Attribue une direction **au hasard** à des instants de **DÉCISION** donnés (G1).

    ⚠️ **`candidats` sont les barres où la décision se prend, PAS celles où l'on entre.** La
    distinction est celle du label, et elle a coûté un bug : `DateTime` y désigne la barre
    candidate, tandis que `label_entree_dt` — candidat + 1 barre — est celle où la position
    s'ouvre, et `prix_entree` est l'**Open de cette barre-là** (C5).

    Dater la décision sur la barre candidate tout en lui donnant le prix de la suivante produit
    une entrée à un prix qui n'a jamais existé à cet instant. C'est précisément ce que la garde
    C5 du simulateur refuse — et c'est elle qui l'a attrapé, sur les vraies données, parce que le
    premier jet de ce module confondait les deux. Le paramètre porte donc le nom de ce qu'il est.

    *(La contiguïté C9 garantit que candidat et entrée partagent la même journée FTMO — vérifié
    sur les 23 880 labels du panel : zéro exception. La cadence « n trades par jour » reste donc
    exacte malgré le décalage d'une barre.)*

    `candidats` doit être **triée et sans doublon** : c'est une donnée du problème, pas quelque
    chose qu'on répare ici. Un tri silencieux masquerait une séquence mal construite en amont, et
    un doublon ferait entrer deux fois au même instant — ce que l'occupation du compte interdit.

    `seed` est **obligatoire, sans défaut**. La comparaison de configurations doit être appariée
    (H8) : à seed identique, les séries du null sont rigoureusement les mêmes d'une config à
    l'autre, ce qui isole l'effet des paramètres. Un défaut ferait dépendre la barre à battre
    d'une valeur que personne n'a choisie.

    ⚠️ **Le tirage se fait sur TOUTES les entrées demandées, puis on filtre.** L'ordre compte :
    tirer au fil de l'eau en sautant les écartées ferait dépendre la direction d'une entrée de ce
    qui lui est arrivé **avant** elle. Une entrée écartée en amont décalerait toutes les suivantes,
    et deux cadences partageant un préfixe cesseraient d'être comparables — ce qui viderait H8 de
    son sens sans rien faire planter.
    """
    manquantes = [c for c in COLONNES_REQUISES if c not in panel_labelle.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du panel labellisé : {manquantes}. Le twin-null ne CALCULE ni "
            "prix d'entrée ni barrières — il les lit, pour être le jumeau exact du système réel "
            "(G1). Passer un panel non labellisé le forcerait à les inventer."
        )
    if seed < 0:
        raise ValueError(
            f"seed doit être >= 0, reçu {seed}. Une seed négative n'est pas reproductible partout."
        )
    liste = list(candidats)
    if any(b <= a for a, b in itertools.pairwise(liste)):
        raise ValueError(
            "Les instants d'entrée doivent être STRICTEMENT croissants (triés, sans doublon). "
            "Ils ne sont ni triés ni dédoublonnés ici volontairement : l'ordre réel est une "
            "donnée du problème, et un doublon ferait entrer deux fois au même instant."
        )

    # ⚠️ La garde `NaT` passe AVANT le contrôle de tri, et l'ordre n'est pas cosmétique : un
    # `NaT` rend `is_monotonic_increasing` faux, donc le tri lèverait en premier et annoncerait
    # un problème de tri là où il y a une donnée manquante (cf. la même correction dans `cadence`).
    horodatages = pd.to_datetime(pd.Series(panel_labelle["DateTime"]))
    if bool(horodatages.isna().any()):
        raise ValueError(
            f"{int(horodatages.isna().sum())} horodatage(s) manquant(s) dans le panel. Un `NaT` "
            "rend fausse toute comparaison : le panel paraîtrait trié et les entrées seraient "
            "appariées n'importe où."
        )
    if not horodatages.is_monotonic_increasing:
        raise ValueError("Le panel labellisé n'est pas trié par DateTime croissant.")
    dt_barres = np.asarray(horodatages.values)
    # Les dates d'ENTRÉE, converties une fois pour toutes : une conversion par ligne rendrait
    # un `Timestamp | NaT` que pyright refuse a juste titre, et le NaT est justement le cas
    # qu'il faut refuser explicitement plutôt que caster.
    dt_entrees = np.asarray(
        pd.to_datetime(pd.Series(panel_labelle["label_entree_dt"])).dt.to_pydatetime(),
        dtype=object,
    )

    # Un tirage par entrée DEMANDÉE, en une fois : la direction d'une entrée ne dépend alors que
    # de sa position dans la liste, jamais de ce qui a été écarté avant elle.
    rng = np.random.default_rng(seed)
    directions = np.where(rng.integers(0, 2, size=len(liste)) == 1, LONG, SHORT)

    decisions: list[Decision] = []
    ecartes: dict[str, int] = {}

    def _ecarter(motif: str) -> None:
        ecartes[motif] = ecartes.get(motif, 0) + 1

    for rang, instant in enumerate(liste):
        # Recherche binaire (F12) : cette boucle tourne des milliers de fois par campagne.
        i = int(np.searchsorted(dt_barres, np.datetime64(instant), side="left"))
        if i >= len(dt_barres) or dt_barres[i] != np.datetime64(instant):
            _ecarter("instant absent du panel")
            continue
        ligne = panel_labelle.iloc[i]
        prix = float(ligne["prix_entree"])
        haute = float(ligne["barriere_haute"])
        basse = float(ligne["barriere_basse"])
        if not (np.isfinite(prix) and np.isfinite(haute) and np.isfinite(basse)):
            # Pas de label sur cette barre : le système réel ne peut pas y entrer non plus, donc
            # le null ne le doit pas. Le motif vient du label, il n'est pas deviné ici.
            motif = ligne["raison_sans_label"]
            _ecarter(str(motif) if pd.notna(motif) else "barrières indisponibles")
            continue
        # ⚠️ La décision est datée sur la barre d'ENTRÉE, lue dans le label — jamais sur la
        # barre candidate. `prix_entree` est l'Open de cette barre-là (C5).
        dt_entree = dt_entrees[i]
        if pd.isna(dt_entree):
            raise ValueError(
                f"Le candidat du {instant} porte des barrières mais aucune date d'entrée. Le "
                "panel est incohérent : sans elle, la décision serait datée sur la barre de "
                "décision et le juge simulerait une entrée à un prix qui n'a jamais existé (C5)."
            )
        decisions.append(
            Decision(
                dt_entree=dt_entree,
                direction=int(directions[rang]),
                prix_entree=prix,
                barriere_haute=haute,
                barriere_basse=basse,
            )
        )

    return TirageNull(decisions=decisions, seed=seed, n_demandes=len(liste), ecartes=ecartes)


def tirer_directions_sur_decisions(decisions: Sequence[Decision], seed: int) -> TirageNull:
    """
    Le jumeau d'une liste de `Decision` — tâche **C4** (2026-09-23) : **seule la direction est
    tirée** (G1), tout le reste est REFLÉTÉ depuis la décision du modèle.

    `tirer_directions` lit les barrières dans les colonnes du label triple-barrière. Une décision
    sans objectif, à sortie propre (F20), n'a pas ces colonnes : refléter des colonnes ne veut
    plus rien dire, refléter la **décision** si. Pour chaque décision, le singe garde :

    - l'instant d'entrée, le prix d'entrée, la sortie propre (`dt_sortie`) et le score ;
    - la **distance** du stop, posée du côté qui fait perdre pour la direction TIRÉE ;
    - la distance de l'objectif s'il en a un — et **aucun objectif s'il n'en a pas** (F20).

    ⚠️ Un tirage par décision, en une fois, sur les décisions triées par entrée : la direction ne
    dépend que du rang (H8), même règle que `tirer_directions`. Sur un label triple-barrière
    sans entrée écartée, les deux tireurs rendent **les mêmes directions et les mêmes
    barrières** — la preuve `check_91` le vérifie bit à bit : ce n'est pas un second null, c'est
    le même, qui sait maintenant refléter une décision sans objectif.
    """
    if not decisions:
        raise ValueError("Aucune décision : le null n'a rien à refléter.")
    # (Formulée autrement que dans `tirer_directions` pour que chaque garde garde SON motif de
    #  mutation — deux lignes identiques rendraient l'ancien motif double, donc absent.)
    if not seed >= 0:
        raise ValueError(
            f"seed doit être >= 0, reçu {seed}. Une seed négative n'est pas reproductible partout."
        )
    ordre = sorted(decisions, key=lambda d: d.dt_entree)
    if any(b.dt_entree <= a.dt_entree for a, b in itertools.pairwise(ordre)):
        raise ValueError(
            "Deux décisions partagent un instant d'entrée : le null entrerait deux fois au même "
            "instant, ce que l'occupation du compte interdit (H11)."
        )
    tirages = np.random.default_rng(seed).integers(0, 2, size=len(ordre))
    directions = np.where(tirages == 1, LONG, SHORT)
    miroir: list[Decision] = []
    for d, direction in zip(ordre, directions, strict=True):
        prix = d.prix_entree
        d_stop = d.distance_stop
        cible = d.cible
        d_cible = None if cible is None else abs(cible - prix)
        if int(direction) == LONG:
            haute, basse = (None if d_cible is None else prix + d_cible), prix - d_stop
        else:
            haute, basse = prix + d_stop, (None if d_cible is None else prix - d_cible)
        miroir.append(
            Decision(
                dt_entree=d.dt_entree,
                direction=int(direction),
                prix_entree=prix,
                barriere_haute=haute,
                barriere_basse=basse,
                score=d.score,
                dt_sortie=d.dt_sortie,
            )
        )
    return TirageNull(decisions=miroir, seed=seed, n_demandes=len(ordre), ecartes={})
