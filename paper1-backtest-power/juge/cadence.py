"""
La cadence paramétrable — tâche 6.2, décisions G3, I7, G9.

Ce module ne juge rien et ne tire aucune direction : il **fabrique des calendriers de candidats**
que le twin-null (6.1) transformera ensuite en décisions. C'est le manque du build précédent, où
la cadence était **figée à 1 trade par jour, sans réglage** — et donc où la barre à battre
n'existait que pour une seule fréquence.

────────────────────────────────────────────────────────────────────────────────────────
Pourquoi une TABLE et pas un chiffre (G3)
────────────────────────────────────────────────────────────────────────────────────────
**La barre à battre dépend du nombre de trades.** Peu de trades → dispersion large → p95 élevé.
Comparer un modèle à une barre calculée pour une autre fréquence est un **faux test**, et ça ne
se voit pas : les deux chiffres sont des `P(funded)`, ils se comparent sans broncher.

Le null se produit donc comme une **table par cadence** (6.5), autour de trois points :
**~500 · ~1 020 · ~2 040 trades** sur les 1 020 journées FTMO du dev, soit ~0,5 · ~1 · ~2 par
jour. Le plafond de 2 vient de **I7** — contrainte de survie sur un compte à drawdown serré.

────────────────────────────────────────────────────────────────────────────────────────
⚠️ Ce module N'EST PAS l'« option 2 » du twin-null (G8)
────────────────────────────────────────────────────────────────────────────────────────
La frontière est fine et elle compte :

- **G8, option 2 — non faite** : un null qui tirerait **ses propres entrées à chaque chemin**, de
  sorte que la variance du *choix des instants* entrerait dans la distribution. Le null cesserait
  alors d'être le jumeau du système réel sur la dimension « quand ».
- **Ce module** : un constructeur de calendrier **extérieur** au null. Il produit **un** calendrier
  par couple (cadence, seed), et le null le reçoit du dehors — exactement comme il recevra la
  liste des entrées du modèle pour le verdict final (G9).

Autrement dit : le null ne choisit toujours pas quand entrer. C'est nous qui lui donnons un
emploi du temps, et nous disons lequel.

────────────────────────────────────────────────────────────────────────────────────────
Pourquoi les entrées sont RÉPARTIES dans la journée, et pas posées à heure fixe
────────────────────────────────────────────────────────────────────────────────────────
Poser l'entrée toujours sur la première barre de la journée serait déterministe, donc tentant.
**Ce serait aussi un biais systématique**, pour deux raisons mesurables :

1. **le plancher journalier est au plus large en début de journée.** Un null qui entre toujours à
   l'ouverture voit toujours 2 500 $ de marge (F2), donc **l'airbag ne mordrait jamais** (H16) et
   le départage stop / barrière de compte (F5) ne jouerait pratiquement pas. Le null serait
   structurellement plus sûr que le système qu'il est censé imiter — et la barre serait fausse
   dans le sens le plus dangereux : trop facile à battre ;
2. **l'amplitude d'une barre H1 dépend de l'heure.** Toujours entrer à la même heure, c'est
   choisir un régime de volatilité, alors que le label a été construit sur toutes les heures.

La position dans la journée est donc **tirée**, avec sa propre seed. ⚠️ **Cette seed n'est pas
celle des directions** : ce sont deux modules et deux tirages. Pour la comparaison appariée de la
phase 6 (**H8**), c'est le **calendrier qui doit rester identique** d'une configuration d'AIMD à
l'autre — on fixe sa seed et on ne fait varier que les paramètres de sizing.

────────────────────────────────────────────────────────────────────────────────────────
Ce qui est demandé n'est pas ce qui sera joué
────────────────────────────────────────────────────────────────────────────────────────
Un calendrier est une **demande**. L'ordonnanceur (4.9) tranchera ensuite : un seul trade à la
fois, donc une entrée qui tombe pendant qu'une position est ouverte est **ignorée** (F14). Le
nombre de trades réellement joués est donc **inférieur** au nombre d'entrées demandées, et
c'est **lui** qui indexe la table (G3), jamais la demande.

**À mesurer, jamais à supposer** : la tâche 6.5 doit lire le nombre réalisé et calibrer la demande
dessus. Annoncer une table « à 2 040 trades » construite sur 2 040 *demandes* serait exactement le
faux test que G3 nomme.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Plafond de cadence — **I7**, décision de Hugo du 2026-08-03 : 1 à 2 trades par jour au
#: maximum, contrainte de survie sur un compte FTMO à drawdown serré. Au-delà, on ne décrit plus
#: le système qu'on a l'intention de trader.
TRADES_PAR_JOUR_MAX: float = 2.0

#: Colonnes lues. `jour_ftmo` est **lue**, jamais recalculée : une seule notion de journée dans
#: tout le projet (B11, piège n°1).
COLONNES_REQUISES: tuple[str, ...] = ("DateTime", "jour_ftmo", "prix_entree", "barriere_haute")


@dataclass(frozen=True)
class Calendrier:
    """Un emploi du temps d'entrées, et de quoi juger s'il vaut ce qu'il annonce."""

    candidats: list[datetime]
    trades_par_jour_demande: float
    seed: int
    n_journees: int
    #: Journées où il n'y avait pas assez de barres labellisées pour honorer l'allocation.
    #: **Rapporté, jamais tu** : un calendrier amputé décrit une autre cadence que celle demandée.
    n_journees_incompletes: int
    n_candidats_manquants: int

    @property
    def cadence_obtenue(self) -> float:
        return len(self.candidats) / self.n_journees if self.n_journees else float("nan")

    def resume(self) -> str:
        manque = (
            f" · {self.n_candidats_manquants} candidat(s) manquant(s) sur "
            f"{self.n_journees_incompletes} journée(s) trop courte(s)"
            if self.n_candidats_manquants
            else ""
        )
        return (
            f"calendrier seed={self.seed} · {self.trades_par_jour_demande:g} trade(s)/jour "
            f"demandé(s) · {len(self.candidats)} candidats sur {self.n_journees} journées FTMO "
            f"(cadence obtenue {self.cadence_obtenue:.3f}){manque}"
        )


def construire_calendrier(
    panel_labelle: pd.DataFrame,
    trades_par_jour: float,
    seed: int,
) -> Calendrier:
    """
    Fabrique un calendrier de **candidats** à la cadence demandée.

    ⚠️ **Ce sont des barres de DÉCISION, pas des barres d'entrée.** Le twin-null les convertira en
    décisions qui s'ouvrent à la barre **suivante** (`label_entree_dt`), dont l'Open est
    `prix_entree` (C5). La confusion entre les deux a coûté un bug, attrapé par la garde C5 du
    simulateur sur les vraies données : le nom porte désormais ce qu'il est.

    *(La contiguïté C9 garantit que candidat et entrée partagent la même journée FTMO — vérifié
    sur les 23 880 labels du panel, zéro exception. La cadence « n trades par jour » reste donc
    exacte malgré le décalage d'une barre.)*

    **Le total est fixé, pas laissé au hasard** : `round(cadence x journées)` entrées au total.
    Chaque journée reçoit `floor(cadence)` entrées, et le reste est distribué sur un
    sous-ensemble de journées **tiré** — sinon une cadence de 0,5 donnerait un total qui varie
    d'un tirage à l'autre, et deux points de la table ne seraient plus comparables entre eux.

    Les entrées d'une journée sont tirées **parmi ses barres labellisées**, sans remise. Une barre
    sans label n'a pas de barrières : le système réel ne peut pas y entrer, donc le null non plus
    (G1). Les tirer quand même reviendrait à demander une cadence que le twin-null raboterait
    ensuite en silence.

    `seed` est **obligatoire, sans défaut** : rien n'est stocké (A8, G5), tout se recalcule à
    l'identique. Un défaut ferait dépendre la barre à battre d'une valeur que personne n'a choisie.
    """
    manquantes = [c for c in COLONNES_REQUISES if c not in panel_labelle.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du panel labellisé : {manquantes}. Ce module ne CALCULE ni la "
            "journée FTMO (B11) ni les barrières — il les lit."
        )
    if not math.isfinite(trades_par_jour) or not 0 < trades_par_jour <= TRADES_PAR_JOUR_MAX:
        raise ValueError(
            f"trades_par_jour doit être fini et dans ]0, {TRADES_PAR_JOUR_MAX:g}] — reçu "
            f"{trades_par_jour!r}. Le plafond vient de I7 : 1 à 2 trades par jour au maximum, "
            "contrainte de survie sur un compte à drawdown serré. Au-delà, la table décrirait un "
            "système qu'on n'a pas l'intention de trader."
        )
    if seed < 0:
        raise ValueError(f"seed doit être >= 0, reçu {seed}.")
    # ⚠️ **L'ORDRE de ces deux gardes compte** — trouvé en écrivant la preuve (6.3). Un `NaT`
    # rend `is_monotonic_increasing` faux, donc le contrôle de tri levait EN PREMIER et annonçait
    # « panel non trié » : un message qui envoie chercher un bug de tri là où il y a une donnée
    # manquante. La garde `NaT`, elle, devenait inatteignable — du code qu'aucun test ne peut
    # exercer donne une fausse confiance. Chaque faute doit lever pour SA propre raison.
    horodatages = pd.to_datetime(pd.Series(panel_labelle["DateTime"]))
    if bool(horodatages.isna().any()):
        raise ValueError(
            f"{int(horodatages.isna().sum())} horodatage(s) manquant(s) dans le panel. Un `NaT` "
            "rend fausse toute comparaison : le panel paraîtrait trié et les entrées seraient "
            "placées n'importe où, sans que rien ne le signale."
        )
    if not horodatages.is_monotonic_increasing:
        raise ValueError("Le panel labellisé n'est pas trié par DateTime croissant.")

    # `np.asarray(..., dtype=float)` d'abord : `pd.to_numeric` rend une Series au type flou, que
    # `np.isfinite` accepte à l'exécution mais que pyright refuse — or c'est pyright qui attrape
    # la classe d'erreur de l'incident du 2026-08-03. On ne contourne pas le vérificateur, on lui
    # donne de quoi vérifier.
    prix = np.asarray(pd.to_numeric(panel_labelle["prix_entree"], errors="coerce"), dtype=float)
    haute = np.asarray(pd.to_numeric(panel_labelle["barriere_haute"], errors="coerce"), dtype=float)
    labellees = np.isfinite(prix) & np.isfinite(haute)
    eligibles = panel_labelle.loc[labellees]
    if len(eligibles) == 0:
        raise ValueError("Aucune barre labellisée dans le panel : aucun calendrier n'est possible.")

    # `jour_ftmo` est LUE. Les journées sont prises dans l'ordre chronologique, jamais trié ici :
    # le panel est déjà trié, et un tri silencieux masquerait un panel mal construit.
    jours = pd.to_datetime(pd.Series(eligibles["jour_ftmo"])).dt.date.to_numpy()
    dts = np.asarray(
        pd.to_datetime(pd.Series(eligibles["DateTime"])).dt.to_pydatetime(), dtype=object
    )
    frontieres = np.flatnonzero(np.concatenate(([True], jours[1:] != jours[:-1])))
    groupes = np.split(np.arange(len(jours)), frontieres[1:])
    n_journees = len(groupes)

    total = round(trades_par_jour * n_journees)
    base = math.floor(trades_par_jour)
    reste = total - base * n_journees

    rng = np.random.default_rng(seed)
    # Le sous-ensemble de journées qui reçoit une entrée de plus est tiré AVANT les positions
    # intra-journée : l'ordre des tirages fait partie du déterminisme, il ne doit pas dépendre
    # du contenu des journées.
    bonus = np.zeros(n_journees, dtype=int)
    if reste > 0:
        bonus[rng.choice(n_journees, size=reste, replace=False)] = 1

    candidats: list[datetime] = []
    n_incompletes = n_manquantes = 0
    for j, indices in enumerate(groupes):
        voulu = base + int(bonus[j])
        if voulu == 0:
            continue
        if voulu > len(indices):
            n_incompletes += 1
            n_manquantes += voulu - len(indices)
            voulu = len(indices)
        choisis = rng.choice(len(indices), size=voulu, replace=False)
        for pos in sorted(int(x) for x in np.asarray(choisis).ravel()):
            candidats.append(dts[int(indices[pos])])

    return Calendrier(
        candidats=candidats,
        trades_par_jour_demande=trades_par_jour,
        seed=seed,
        n_journees=n_journees,
        n_journees_incompletes=n_incompletes,
        n_candidats_manquants=n_manquantes,
    )
