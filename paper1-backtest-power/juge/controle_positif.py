"""
Le contrôle POSITIF — tâche 7.6, décision A10.

Le twin-null est le contrôle **négatif** : il prouve qu'un système sans signal ne marque pas de
points, donc il protège du **faux GO**. Rien, jusqu'ici, ne protège du **faux NO-GO** — un edge
réel masqué par un bug de pipeline. Et les deux ne coûtent pas pareil : un faux GO fait perdre de
l'argent, un faux NO-GO fait **abandonner un edge sans jamais le savoir**. Sans contrôle positif,
un NO-GO peut vouloir dire « pas d'edge » **ou** « chaîne cassée », et on ne peut pas trancher.

────────────────────────────────────────────────────────────────────────────────────────
Le principe : planter un edge de force CONNUE, vérifier que la chaîne le retrouve
────────────────────────────────────────────────────────────────────────────────────────
On prend les mêmes candidats que le null, mais au lieu de tirer la direction à pile ou face, on
l'aligne sur l'issue réelle du label avec une probabilité `edge` **choisie**. Un trade dont le
label dit « haut » a pour direction gagnante LONG ; on la prend avec probabilité `edge`, l'opposée
sinon. Sur les trades à direction gagnante claire (haut / bas), le **taux de réussite directionnel
vaut donc `edge` par construction** — `edge = 0,5` reproduit exactement le null, `edge > 0,5` est
un système qui sait quelque chose.

On fait ensuite passer ces décisions par **la même chaîne que tout le reste** — ordonnanceur →
juge (`methode_a`) → `P(funded)` — et on compare au null (l'edge 0,5, joué par la même chaîne).
Un edge est **détectable** quand son `P(funded)` dépasse la barre du null (le p95, G2) de façon
lisible. En balayant des edges décroissants, on mesure le **plus petit edge détectable** : la
réponse à « la chaîne verrait-elle le plancher de décision de 55 %, ou est-il noyé dans le
bruit ? ».

────────────────────────────────────────────────────────────────────────────────────────
Portée en phase 7, et ce qui attend la phase 9
────────────────────────────────────────────────────────────────────────────────────────
Ici l'edge est injecté **directement au niveau des décisions** — comme si un modèle parfait sur
une variable cachée avait décidé. C'est le contrôle positif de la chaîne d'**ÉVALUATION** (juge +
null + DSR), la seule qui existe en phase 7. ⚠️ Le contrôle positif de l'**APPRENTISSAGE** — un
vrai modèle retrouve-t-il l'edge depuis des features synthétiques ? — appartient à la phase 9,
quand le modèle existera ; il rejouera cette même mécanique en amont. Distinguer les deux évite de
croire la chaîne entière validée alors que seule sa moitié aval l'est.

⚠️ **Les nombres 0,58 / 0,55 / 0,53 sont des PARAMÈTRES de contrôle, pas des résultats de modèle**
(table rase) : on choisit la force de l'edge planté, on ne la découvre pas.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from aimd import AIMDConfig
from cadence import construire_calendrier
from calibration import BUDGET_UN_AN, UNITES_MINIMUM_PROVISOIRE
from methodes_pfunded import methode_a
from ordonnanceur import ordonnancer
from regles_ftmo import ReglesFTMO
from simulateur import LONG, SHORT, Decision
from trajectoires import SEUIL_CENSURE_ALARMANT
from twin_null import barre_du_null

#: L'edge du null : direction à pile ou face, aucun signal. `injecter_edge(0,5)` == le twin-null.
EDGE_NULL: float = 0.5


@dataclass(frozen=True)
class TirageEdge:
    """Des décisions à edge planté, et de quoi vérifier que l'edge est bien celui demandé."""

    decisions: list[Decision]
    edge: float
    seed: int
    n_demandes: int
    #: Direction alignée sur l'issue du label, sur les trades à gagnante claire (haut/bas). Compté
    #: pour vérifier que le taux réalisé colle à l'edge visé — sinon l'injection ment sur sa force.
    n_gagnante_possible: int
    n_direction_gagnante: int
    ecartes: dict[str, int] = field(default_factory=dict)

    @property
    def taux_reussite_realise(self) -> float:
        """Part des trades à gagnante claire où la direction PRISE est la gagnante — doit ≈ edge."""
        if self.n_gagnante_possible == 0:
            return float("nan")
        return self.n_direction_gagnante / self.n_gagnante_possible


def _direction_gagnante(issue: str) -> int | None:
    """LONG si le label dit « haut », SHORT s'il dit « bas », None sinon (double/timeout perdent)."""
    if issue == "haut":
        return LONG
    if issue == "bas":
        return SHORT
    return None


def injecter_edge(
    panel_labelle: pd.DataFrame,
    candidats: Sequence[datetime],
    edge: float,
    seed: int,
) -> TirageEdge:
    """
    Fabrique des décisions dont le taux de réussite directionnel vaut `edge` (A10).

    ⚠️⚠️ **CETTE FONCTION TRICHE EXPRÈS : elle lit l'issue RÉELLE du label — donc le futur — pour
    aligner la direction.** C'est le mécanisme même du contrôle positif (un modèle synthétique de
    force connue), et ce serait un **look-ahead massif** si on l'employait comme un vrai décideur.
    Elle n'a sa place que dans le CONTRÔLE : jamais en production, jamais dans le pipeline de la
    phase 9. Un vrai modèle décide depuis des features causales (I9), il ne connaît pas l'issue.

    Même squelette que `twin_null.tirer_directions` — mêmes gardes (NaT, tri strict, écartés
    comptés et lus du label), mêmes barrières **lues** jamais recalculées (piège n°5) — mais la
    direction n'est pas un pile ou face : sur un trade à gagnante claire, on prend la gagnante avec
    probabilité `edge`. ⚠️ `edge = 0,5` reproduit le null (contrôle négatif), et la preuve le
    vérifie : à même seed, la distribution doit coïncider avec celle du twin-null — mêmes trades,
    mêmes écartés. ⚠️ **Pas le même tirage** : les deux fonctions consomment leur RNG autrement
    (ici deux tirages d'un bloc, là un seul), donc à seed égale les directions sont indépendantes
    (mesuré le 2026-09-15 : 504 directions communes sur 1 020). Le null d'une campagne est un
    **nouvel échantillon** du null, jamais celui de la table 6.5 — la barre de G2 se re-mesure
    dans la cellule, elle ne se transporte pas (piège n°10), et l'écart entre deux échantillons de
    60 tirages (p95 0,447 le 05/09 contre 0,401 le 15/09 à 1/j, 2 %) est le bruit de la barre.

    ⚠️ **La duplication de la logique de `tirer_directions` est assumée** : la factoriser
    imposerait à `twin_null` (module prouvé et stable) de connaître la notion d'edge, qui lui est
    étrangère. La contrepartie est une **assertion de cohérence** en preuve — à `edge = 0,5`, part
    de longs et comptes d'écartés identiques au null.
    """
    if not 0.0 <= edge <= 1.0:
        raise ValueError(
            f"edge est une probabilité dans [0, 1], reçu {edge!r}. 0,5 = null (pile ou face), "
            "au-dessus = un système qui sait, en dessous = un système à contre-emploi."
        )
    if seed < 0:
        raise ValueError(f"seed doit être >= 0, reçu {seed}.")
    liste = list(candidats)
    if any(b <= a for a, b in itertools.pairwise(liste)):
        raise ValueError(
            "Les instants doivent être STRICTEMENT croissants (triés, sans doublon) — comme pour "
            "le null : l'ordre réel est une donnée du problème, jamais réparé ici."
        )
    for colonne in (
        "DateTime",
        "issue",
        "prix_entree",
        "barriere_haute",
        "barriere_basse",
        "label_entree_dt",
    ):
        if colonne not in panel_labelle.columns:
            raise ValueError(f"Colonne « {colonne} » absente du panel labellisé.")

    horodatages = pd.to_datetime(pd.Series(panel_labelle["DateTime"]))
    if bool(horodatages.isna().any()):
        raise ValueError(
            f"{int(horodatages.isna().sum())} horodatage(s) manquant(s) : un `NaT` rendrait le "
            "panel faussement trié et placerait les entrées n'importe où."
        )
    if not horodatages.is_monotonic_increasing:
        raise ValueError("Le panel labellisé n'est pas trié par DateTime croissant.")
    dt_barres = np.asarray(horodatages.values)
    dt_entrees = np.asarray(
        pd.to_datetime(pd.Series(panel_labelle["label_entree_dt"])).dt.to_pydatetime(), dtype=object
    )
    issues = np.asarray(panel_labelle["issue"], dtype=object)

    rng = np.random.default_rng(seed)
    # Deux tirages indépendants tirés d'un coup : « prend-on la gagnante ? » et le pile ou face de
    # secours (double/timeout, ou trade où l'on renonce à la gagnante). Tirer d'un bloc rend la
    # direction d'une entrée indépendante de ce qui a été écarté avant elle (piège du twin-null).
    prend_gagnante = rng.random(len(liste)) < edge
    pile_ou_face = np.where(rng.integers(0, 2, size=len(liste)) == 1, LONG, SHORT)

    decisions: list[Decision] = []
    ecartes: dict[str, int] = {}
    n_gagnante_possible = n_direction_gagnante = 0

    def _ecarter(motif: str) -> None:
        ecartes[motif] = ecartes.get(motif, 0) + 1

    for rang, instant in enumerate(liste):
        i = int(np.searchsorted(dt_barres, np.datetime64(instant), side="left"))
        if i >= len(dt_barres) or dt_barres[i] != np.datetime64(instant):
            _ecarter("instant absent du panel")
            continue
        ligne = panel_labelle.iloc[i]
        prix = float(ligne["prix_entree"])
        haute = float(ligne["barriere_haute"])
        basse = float(ligne["barriere_basse"])
        if not (np.isfinite(prix) and np.isfinite(haute) and np.isfinite(basse)):
            motif = ligne.get("raison_sans_label")
            _ecarter(motif if isinstance(motif, str) and motif else "barrières indisponibles")
            continue
        dt_entree = dt_entrees[i]
        if pd.isna(dt_entree):
            raise ValueError(
                f"Le candidat du {instant} porte des barrières mais aucune date d'entrée (C5)."
            )

        gagnante = _direction_gagnante(str(issues[i]))
        if gagnante is None:
            # double / timeout : aucune direction ne gagne, pile ou face pur.
            direction = int(pile_ou_face[rang])
        else:
            n_gagnante_possible += 1
            if prend_gagnante[rang]:
                direction = gagnante
                n_direction_gagnante += 1
            else:
                direction = -gagnante  # l'opposée de la gagnante

        decisions.append(
            Decision(
                dt_entree=dt_entree,
                direction=direction,
                prix_entree=prix,
                barriere_haute=haute,
                barriere_basse=basse,
            )
        )

    return TirageEdge(
        decisions=decisions,
        edge=edge,
        seed=seed,
        n_demandes=len(liste),
        n_gagnante_possible=n_gagnante_possible,
        n_direction_gagnante=n_direction_gagnante,
        ecartes=ecartes,
    )


@dataclass(frozen=True)
class PointControle:
    """Ce qu'un edge planté donne au juge, comparé au null — jamais un chiffre seul."""

    edge: float
    n_replications: int
    #: `P(funded)` (borne basse) de l'edge : médiane et p95 sur les réplications.
    p50: float
    p95: float
    taux_reussite_median: float
    #: La barre du null (p95 de l'edge 0,5, joué par la MÊME chaîne). Le point de comparaison G2.
    barre_null_p95: float
    censure_mediane: float
    #: La censure du NULL qui a fourni la barre. ⚠️ Sans elle, une barre issue d'un null qui ne
    #: conclut pas (censure 40 à 96 % à 0,5 % de risque, mesuré le 2026-09-15) vaut ~0 — et
    #: n'importe quel edge « la bat » sans rien prouver. La barre voyage avec sa fiabilité (F13).
    censure_null_mediane: float

    @property
    def interpretable(self) -> bool:
        """
        Faux dès que la censure médiane de l'edge OU celle du null dépasse le seuil de F13.

        Même garde que `table_null.CelluleDeTable.interpretable` et `calibration.PointDeGrille`,
        pour la même raison : un chiffre de fiabilité **rapporté** qu'aucune garde n'impose finit
        lu comme un résultat. Trouvé à la première campagne sur les vraies données (2026-09-15) :
        à 0,5 trade/jour et 0,5 % de risque, le null a 95,7 % de censure, sa barre vaut 0,001, et
        un edge à 99,2 % de censure ressortait « DÉTECTABLE » avec un p50 de 0,008. Les deux
        termes comptent : un edge fort conclut vite (censure ~0) alors que le null de la même
        cellule n'a pas conclu — comparer les deux serait comparer un réalisé à une convention.
        """
        return (
            self.censure_mediane <= SEUIL_CENSURE_ALARMANT
            and self.censure_null_mediane <= SEUIL_CENSURE_ALARMANT
        )

    @property
    def detectable(self) -> bool:
        """
        Vrai quand la cellule est interprétable ET que la MÉDIANE de l'edge dépasse la barre du
        null (son p95). Un edge dont un tirage typique bat déjà le meilleur singe chanceux est vu
        par la chaîne ; sinon il est noyé. L'edge 0,5 lui-même ne doit JAMAIS être détectable —
        c'est le contrôle négatif intégré. ⚠️ Non interprétable n'est pas « noyé » : c'est « la
        question n'a pas été posée » — `resume` distingue les deux.
        """
        return self.interpretable and self.p50 > self.barre_null_p95

    def resume(self) -> str:
        if not self.interpretable:
            etat = "NON INTERPRÉTABLE — censure > seuil F13"
        else:
            etat = "DÉTECTABLE" if self.detectable else "noyé dans le bruit"
        return (
            f"edge {self.edge:.0%} (réussite réalisée {self.taux_reussite_median:.0%}) : "
            f"P(funded) p50 {self.p50:.3f} p95 {self.p95:.3f} vs barre null {self.barre_null_p95:.3f} "
            f"→ {etat} * censure {self.censure_mediane:.3f} (null {self.censure_null_mediane:.3f})"
        )


def _bornes_pour_edge(
    panel_labelle: pd.DataFrame,
    candidats: list[datetime],
    edge: float,
    config: AIMDConfig,
    seeds: Sequence[int],
    pas: int,
    regles: ReglesFTMO | None,
    budget_journees: int | None,
    unites_minimum: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Pour un edge : les bornes basses, censures et taux réalisés, une valeur par seed.

    Le sizing est **le même que pour le null** (la config passée) — c'est la condition d'une
    comparaison honnête : edge et null ne diffèrent que par la DIRECTION, jamais par la taille.
    """
    return _bornes_pour_tirages(
        panel_labelle,
        lambda seed: injecter_edge(panel_labelle, candidats, edge, seed),
        config,
        seeds,
        pas,
        regles,
        budget_journees,
        unites_minimum,
    )


def _bornes_pour_tirages(
    panel_labelle: pd.DataFrame,
    tirer: Callable[[int], TirageEdge],
    config: AIMDConfig,
    seeds: Sequence[int],
    pas: int,
    regles: ReglesFTMO | None,
    budget_journees: int | None,
    unites_minimum: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Le balayage lui-même, indépendant de la FAÇON de planter l'edge (C3, 2026-09-23).

    Un seul injecteur vit ici depuis C6 (2026-09-25) : `injecter_edge` aligne la **direction**
    (triple-barrière). Le balayage reste séparé de l'injecteur parce que c'est la chaîne aval —
    ordonnanceur, `methode_a`, borne basse — qu'on mesure, pas ce qu'un injecteur fabrique : un
    futur injecteur (un filtre de jours, G8) s'y branche sans dupliquer la boucle.
    """
    bornes: list[float] = []
    censures: list[float] = []
    taux: list[float] = []
    for seed in seeds:
        tirage = tirer(seed)
        taux.append(tirage.taux_reussite_realise)
        ordo = ordonnancer(tirage.decisions, panel_labelle)
        dist = methode_a(
            ordo.retenus,
            panel_labelle,
            aimd=config,
            pas=pas,
            regles=regles,
            budget_journees=budget_journees,
            unites_minimum=unites_minimum,
        )
        bornes.append(dist.p_funded_borne_basse)
        censures.append(dist.p_censure)
    return bornes, censures, taux


def mesurer_controle(
    panel_labelle: pd.DataFrame,
    trades_par_jour: float,
    config: AIMDConfig,
    edges: Sequence[float],
    seeds_directions: Sequence[int],
    seed_calendrier: int,
    pas: int = 1,
    regles: ReglesFTMO | None = None,
    budget_journees: int | None = BUDGET_UN_AN,
    unites_minimum: float = UNITES_MINIMUM_PROVISOIRE,
) -> list[PointControle]:
    """
    Joue chaque edge à travers la chaîne complète et le compare au null (A10).

    Le **calendrier est construit une fois** (comme la table du null) : tous les edges et le null
    jouent exactement les mêmes instants — seule la direction change. Le null (`EDGE_NULL = 0,5`)
    est **toujours mesuré**, quelle que soit la liste `edges` : c'est lui qui fournit la barre, et
    son détectable doit rester Faux (contrôle négatif intégré). Le p95 du null sur ces réplications
    est la barre à battre (G2).
    """
    if not edges:
        raise ValueError("Aucun edge à mesurer.")
    calendrier = construire_calendrier(panel_labelle, trades_par_jour, seed_calendrier)

    # Chaque edge (dont le null) est calculé une seule fois, mémorisé par sa valeur.
    a_mesurer = sorted({EDGE_NULL, *edges})
    mesures = {
        edge: _bornes_pour_edge(
            panel_labelle,
            calendrier.candidats,
            edge,
            config,
            seeds_directions,
            pas,
            regles,
            budget_journees,
            unites_minimum,
        )
        for edge in a_mesurer
    }
    # La barre : le p95 des bornes du null, joué par la MÊME chaîne (G2) — et sa fiabilité, la
    # censure médiane du null, qui décide si la barre a le droit d'être lue (F13).
    barre = barre_du_null(mesures[EDGE_NULL][0]).p95
    censure_null = float(np.median(mesures[EDGE_NULL][1]))

    points: list[PointControle] = []
    for edge in a_mesurer:
        bornes, censures, taux = mesures[edge]
        points.append(
            PointControle(
                edge=edge,
                n_replications=len(seeds_directions),
                p50=float(np.median(bornes)),
                p95=float(np.percentile(bornes, 95)),
                taux_reussite_median=float(np.nanmedian(taux)),
                barre_null_p95=barre,
                censure_mediane=float(np.median(censures)),
                censure_null_mediane=censure_null,
            )
        )
    return points


def plus_petit_edge_detectable(points: Sequence[PointControle]) -> float | None:
    """
    Le plus petit edge (> null) dont le `P(funded)` dépasse la barre du null (G2).

    `None` si aucun edge testé n'est détectable — un résultat en soi : la chaîne ne verrait aucun
    des edges plantés, il faudrait plus de trades ou une meilleure métrique avant de chercher un
    edge réel (A10). Un point **non interprétable** (censure au-dessus du seuil F13, côté edge ou
    côté null) n'est jamais détectable, donc jamais rendu ici : on écarte, on ne lit pas. ⚠️ Le null lui-même (edge 0,5) est exclu : s'il ressortait « détectable », ce
    serait un bug du comparateur, pas un edge — la preuve le vérifie séparément.
    """
    detectables = sorted(p.edge for p in points if p.edge > EDGE_NULL and p.detectable)
    return detectables[0] if detectables else None
