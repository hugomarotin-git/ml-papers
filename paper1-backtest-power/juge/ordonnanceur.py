"""
L'ordonnanceur d'occupation — tâche 4.9, décisions H11 et F14.

Le modèle proposera **plus de candidats que le compte ne peut en porter**. Un seul trade à la
fois y tient (H11), et la règle de décision en produit jusqu'à deux par journée FTMO (I12). Ce
module transforme donc une liste de *candidats* en la liste des trades **réellement pris**.

────────────────────────────────────────────────────────────────────────────────────────
Il tourne AVANT toute simulation, et c'est le point (H11)
────────────────────────────────────────────────────────────────────────────────────────
La sélection dépend de l'instant, de l'occupation et du score — **jamais de la taille des
positions**. Elle se précalcule donc une seule fois, et toutes les configurations de sizing
comparées ensuite tournent sur **exactement la même séquence de trades**. Sans ça, comparer
deux configurations d'AIMD reviendrait à comparer deux systèmes différents (H8).

────────────────────────────────────────────────────────────────────────────────────────
À égalité d'instant, le MIEUX NOTÉ (F14)
────────────────────────────────────────────────────────────────────────────────────────
Quand plusieurs candidats sont valides au même instant, prendre le premier de la liste jetterait
l'information de confiance que le modèle vient de produire — et le résultat dépendrait de
l'ordre de tri en amont, ce qui n'est pas une décision de trading.

Corollaire imposé par F14 et **asserté ici** : le score doit exister avant l'ordonnanceur. Un
départage entre candidats sans score lève, plutôt que de retomber silencieusement sur l'ordre
d'arrivée.

────────────────────────────────────────────────────────────────────────────────────────
Pourquoi glouton dans le temps, et pourquoi c'est la SEULE option honnête
────────────────────────────────────────────────────────────────────────────────────────
Prendre un trade à *t* peut bloquer un meilleur candidat à *t+1*. Un ordonnancement « optimal »
ferait donc mieux — mais il choisirait en connaissant l'avenir. **C'est du look-ahead**, et le
gain qu'il produirait serait entièrement fictif.

Le glouton dans le temps n'est pas une simplification : c'est la seule politique qu'un trader
puisse réellement exécuter. À l'instant où le candidat se présente, l'avenir n'existe pas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from methodes_pfunded import resoudre_trades
from simulateur import Decision


@dataclass(frozen=True)
class Ordonnancement:
    """
    Le résultat : ce qui est pris, ce qui est écarté, et **pourquoi**.

    Les rejets ne sont pas jetés — ils sont comptés et ventilés. Une cadence qui s'effondre
    parce que le compte est toujours occupé est une information sur le système, pas un détail
    d'implémentation : elle change la colonne du twin-null à laquelle on se compare (G3).
    """

    retenus: list[Decision]
    n_candidats: int
    n_rejetes_occupation: int
    n_rejetes_departage: int

    @property
    def taux_de_retention(self) -> float:
        return len(self.retenus) / self.n_candidats if self.n_candidats else 0.0

    def resume(self) -> str:
        return (
            f"  candidats          : {self.n_candidats}\n"
            f"  retenus            : {len(self.retenus)}  ({100 * self.taux_de_retention:.1f} %)\n"
            f"  écartés, compte occupé  : {self.n_rejetes_occupation}\n"
            f"  écartés, moins bien notés : {self.n_rejetes_departage}"
        )


def ordonnancer(candidats: list[Decision], barres: pd.DataFrame) -> Ordonnancement:
    """
    Transforme des candidats en séquence réellement exécutable.

    Un trade occupe le compte de sa barre d'entrée jusqu'à sa barre de sortie **incluse** : le
    candidat suivant ne peut donc entrer qu'à la barre d'après. La durée d'occupation vient de
    `resoudre_trades`, qui résout chaque candidat **isolément** — c'est légitime parce que la
    résolution d'un trade ne dépend pas de ceux qu'on a pris avant (H11), seulement du marché.

    Les candidats sont supposés triés par date d'entrée ; ils ne sont pas retriés en silence,
    pour la même raison que dans `jouer_chemin` : l'ordre réel est une donnée du problème.
    """
    if not candidats:
        return Ordonnancement([], 0, 0, 0)

    dts = [c.dt_entree for c in candidats]
    if dts != sorted(dts):
        raise ValueError(
            "Les candidats ne sont pas triés par date d'entrée. Ils ne sont pas retriés ici "
            "volontairement : un tri silencieux masquerait une séquence mal construite en amont."
        )

    # Résolution isolée de chaque candidat : donne l'occupation, indépendamment de la sélection.
    resolus = resoudre_trades(candidats, barres)

    # Regroupement par instant d'entrée, en préservant l'ordre des instants.
    par_instant: dict[int, list[int]] = {}
    for i, t in enumerate(resolus):
        par_instant.setdefault(t.barre_entree, []).append(i)

    retenus: list[Decision] = []
    n_occupation = n_departage = 0
    libre_a_partir_de = -1

    for barre in sorted(par_instant):
        indices = par_instant[barre]

        if barre < libre_a_partir_de:
            n_occupation += len(indices)
            continue

        if len(indices) > 1:
            # F14 — le mieux noté, jamais le premier arrivé.
            scores = [candidats[i].score for i in indices]
            if any(math.isnan(s) for s in scores):
                raise ValueError(
                    f"{len(indices)} candidats se présentent à la même barre ({barre}) et au "
                    "moins un n'a PAS de score. F14 impose de départager par le score, et son "
                    "corollaire est que le score doit exister avant l'ordonnanceur — sinon on "
                    "retomberait en silence sur l'ordre d'arrivée, qui n'est pas une décision "
                    "de trading."
                )
            gagnant = max(indices, key=lambda i: candidats[i].score)
            n_departage += len(indices) - 1
        else:
            gagnant = indices[0]

        retenus.append(candidats[gagnant])
        libre_a_partir_de = resolus[gagnant].barre_sortie + 1

    return Ordonnancement(
        retenus=retenus,
        n_candidats=len(candidats),
        n_rejetes_occupation=n_occupation,
        n_rejetes_departage=n_departage,
    )
