"""
Le simulateur de compte FTMO — tâche 4.2, décisions F1, F2, F3.

Il produit les observations que `regles_ftmo.SuiviChallenge` juge. Le partage des rôles est
strict : **ce module tient le compte, l'autre dit si c'est gagné.** Aucune règle de challenge
n'est réimplémentée ici.

────────────────────────────────────────────────────────────────────────────────────────
Équity = balance + flottant (F3)
────────────────────────────────────────────────────────────────────────────────────────
La **balance** ne bouge qu'à la clôture d'un trade. L'**équity** bouge à chaque barre, parce
qu'elle inclut le P&L de la position ouverte.

C'est cette distinction qui fait tout l'intérêt du module : un compte peut sauter sur du
flottant sans qu'aucun trade ne soit clôturé. Ne suivre que le réalisé, c'est manquer
exactement les busts qui arrivent pour de vrai.

Et l'asymétrie de F15 s'appuie dessus : les planchers se jugent sur l'équity, la cible sur la
balance.

────────────────────────────────────────────────────────────────────────────────────────
La journée FTMO, encore et toujours (B11)
────────────────────────────────────────────────────────────────────────────────────────
Le simulateur ne calcule **jamais** une journée : il lit la colonne `jour_ftmo` produite par
`ftmo_cleaner`, et annonce chaque changement à `SuiviChallenge`. Une seule notion de journée
dans tout le projet, calculée à un seul endroit.

Le plancher journalier se repose à chaque frontière de journée sur la **balance** de début de
journée, moins 5 % du capital INITIAL — le montant autorisé est fixe, c'est l'ancre qui bouge
(règle FTMO réelle, alignée le 2026-08-10). Le flottant ne déplace pas la ligne ; il compte
dans la violation, testée sur l'équity (F3).

────────────────────────────────────────────────────────────────────────────────────────
Résolution intra-barre : le côté défavorable D'ABORD (F4)
────────────────────────────────────────────────────────────────────────────────────────
Une barre H1 ne dit pas dans quel ordre le High et le Low ont été touchés. Convention, la même
que celle du label (C7) : **on suppose le pire**. Sur une barre où le stop et la cible sont tous
deux atteignables, c'est le stop qui compte.

Même logique pour l'équity intra-barre : le **pire niveau** de la barre est celui qui est
présenté aux planchers, jamais le Close. Un compte qui traverse son plancher en séance est
fermé, même s'il termine la barre au-dessus.

────────────────────────────────────────────────────────────────────────────────────────
Le départage stop / barrière de compte, par proximité à l'entrée (F5) — tâche 4.3
────────────────────────────────────────────────────────────────────────────────────────
Sur une même barre, deux choses peuvent tuer la position : son **stop**, et une **barrière de
compte** (le plancher journalier ou le plancher total, celui des deux qui mord).

Ce sont deux objets de nature différente — l'un est un niveau de prix, l'autre un niveau
d'équity — et tant qu'ils ne sont pas exprimés dans la même unité, « lequel arrive en premier »
n'a aucun sens. `Position.prix_pour_equity` convertit la barrière de compte en **prix** : celui
auquel l'équity vaudrait exactement le plancher.

Une fois les deux en prix, la règle est mécanique. Le prix part de l'entrée et se déplace de
façon **monotone** : il croise donc nécessairement le plus **proche** en premier. F5 se réduit à
« la plus petite distance à l'entrée gagne » — ce n'est pas une convention, c'est le seul ordre
physiquement possible. À égalité stricte, c'est le compte qui l'emporte : convention pessimiste.

**Ce que ça corrige par rapport à la version 4.2.** Elle présentait au compte l'équity calculée
au **pire prix de la barre**, même quand le stop l'avait déjà fermée en chemin. Une barre qui
plonge loin sous le stop faisait donc voir au compte une perte que le stop avait empêchée —
un bust fabriqué. Désormais, si le stop est le plus proche, l'équity la plus basse vue par le
compte est celle **au stop**, pas au bas de la barre.

**Quand la barrière de compte gagne, elle LIQUIDE la position** au prix du plancher : c'est ce
que fait FTMO quand la limite journalière est franchie, et c'est pourquoi la balance finale d'un
bust vaut exactement le plancher. La version 4.2 laissait la position ouverte et la balance
intacte, ce qui décrivait un compte fermé mais toujours investi.

⚠️ **Quand ce départage mord-il vraiment ?** Le plancher n'est plus proche que le stop que si
`balance - plancher < risque du trade` — donc jamais pour le premier trade d'une journée à
risque modeste, et systématiquement après avoir déjà perdu dans la journée. C'est exactement la
situation où un juge naïf se trompe : il laisserait le trade aller au bout de son stop alors que
le compte est déjà mort en chemin.

⚠️ **Le stop est exécuté EXACTEMENT à son niveau, jamais moins bien.** Sur une barre qui plonge
loin sous le stop — un gap, une ouverture de dimanche — l'exécution réelle serait pire. C'est
donc une hypothèse **optimiste**, la seule de ce module, et il faut le savoir : elle se corrigera
avec le slippage (tâche 10.1), pas avant.

────────────────────────────────────────────────────────────────────────────────────────
Une propriété à connaître avant de lire un P(funded)
────────────────────────────────────────────────────────────────────────────────────────
Avec un risque `r` par trade et un plancher journalier à 5 %, **un trade seul ne peut pas faire
buster sur du flottant tant que `r <= 5 %`** : la perte flottante vaut
`capital x r x (mouvement / distance_stop)`, et le mouvement reste inférieur à la distance du
stop tant que celui-ci n'est pas touché — donc la perte flottante d'un trade est bornée par son
risque.

Autrement dit, à la cadence visée (1-2 trades/jour, risque de l'ordre du pour cent), les busts
viendront de **séries de pertes réalisées**, pas d'un flottant catastrophique. C'est cohérent
avec F9 : la taille de bloc de la méthode B doit couvrir les pires séries, parce que ce sont
elles qui tuent le compte.

────────────────────────────────────────────────────────────────────────────────────────
Ce qui n'est délibérément PAS branché
────────────────────────────────────────────────────────────────────────────────────────
- **Les coûts** (F7) : `cout_par_trade` existe, vaut zéro, et n'est pas estimé. Tout P(funded)
  produit avant la tâche 10.1 est hors spread, hors slippage, hors commission — et le spread du
  dev est structurellement moins cher que celui du déploiement (B10).
- **La granularité de lot** : le sizing est continu. F1 note que le choix du compte 50k tenait
  en partie à la granularité ; l'arrondi au pas de lot se branchera avec les coûts, puisqu'il
  demande la spec de contrat du broker. `pas_de_lot=None` le dit explicitement plutôt que de
  laisser croire que la question est réglée.
*(Cette liste portait jusqu'au 2026-08-18 un cinquième point — « le sizing adaptatif n'est PAS
branché ». Il est branché depuis la tâche 5.6, et le laisser aurait été le piège n°0 du projet :
une doc qui décrit un état qui n'existe plus envoie coder la mauvaise chose. Les quatre garanties
qu'elle listait comme « à faire revenir » sont revenues, chacune avec sa preuve et sa mutation :
source unique H9, création de l'état par le simulateur, garde F11 exécutable, et le contrôle
« config dégénérée ≡ sizing constant au centime près ».)*
- **L'ordonnancement fin** (F14, tâche 4.9) : un seul trade à la fois, et si plusieurs décisions
  tombent pendant qu'une position est ouverte, elles sont **ignorées et comptées**. Le choix du
  mieux noté viendra avec l'ordonnanceur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from aimd import AIMDConfig, SuiviAIMD
from chemins import horizon_du_label
from regles_ftmo import Issue, ReglesFTMO, SuiviChallenge


#: **La barrière temporelle du juge EST l'horizon du label** (F6, C3) — tâche 4.4.
#:
#: Elle n'est pas « égale à 6 », elle est **importée** de `label_triple_barriere`. La différence
#: n'est pas cosmétique : un `6` recopié ici fonctionnerait parfaitement aujourd'hui et
#: divergerait en silence le jour où l'horizon du label change. Le juge jugerait alors des
#: trades d'une durée que le label ne décrit pas, sans qu'aucune erreur ne soit levée.
#:
#: C3 le dit : `DEFAULT_HORIZON` est la source unique, toute autre borne d'horizon du projet
#: doit la suivre. La suivre veut dire l'importer, pas la recopier.
#:
#: ⚠️⚠️ **Lu à l'APPEL, plus à l'IMPORT — tâche C1, 2026-09-23.** Tant que ce module évaluait
#: `horizon_du_label()` en tête, importer le juge importait le label triple-barrière : le juge
#: ne pouvait pas survivre à la mort de ce label (date de mort C6), alors qu'une stratégie dont
#: chaque décision porte sa propre sortie (`dt_sortie`) n'a **aucun** besoin de lui. La source
#: unique est conservée pour le monde qui en dépend ; la dépendance n'est plus payée par celui
#: qui ne l'utilise pas. `check_93` vérifie les deux : la garde mord toujours, et l'import du
#: juge ne charge plus l'ancien label.
def horizon_global() -> int:
    """L'horizon du label, pour une décision qui ne porte pas sa propre sortie (F6, C3)."""
    h = horizon_du_label()
    if h < 1:
        raise RuntimeError(f"Horizon du label incohérent ({h}) : il faut au moins 1 barre.")
    return h


#: Colonnes attendues dans le panel de barres.
COLONNES_BARRES: tuple[str, ...] = ("DateTime", "Open", "High", "Low", "Close", "jour_ftmo")

LONG: int = 1
SHORT: int = -1


@dataclass(frozen=True)
class Decision:
    """
    Un trade décidé, **avant** toute simulation.

    La séquence des décisions est indépendante du sizing et peut donc être précalculée une fois
    pour toutes (H11) : seule la TAILLE dépend de la simulation en cours. C'est ce qui permettra
    de comparer plusieurs configurations de sizing sur exactement les mêmes trades.

    Les champs correspondent un pour un aux colonnes du label (`prix_entree`, `barriere_haute`,
    `barriere_basse`, `label_entree_dt`) : le juge évalue les trades que le label décrit, sans
    recalculer aucune barrière.

    ────────────────────────────────────────────────────────────────────────────────────────
    ⚠️⚠️ `Decision` EST le contrat entre un label et le juge — décision 0.1 de Hugo, 2026-09-23
    ────────────────────────────────────────────────────────────────────────────────────────
    Le juge ne lit pas des colonnes, il reçoit des `Decision`. Chaque label construit les siennes
    dans son vocabulaire ; le juge n'en connaît qu'un. C'est ce qui permet à deux labels de
    coexister sans adaptateur qui renomme, et sans valeur truquée (P12).

    Deux généralisations, tâche **C1** (2026-09-23), pour qu'une stratégie sans take-profit et à
    sortie fixe soit descriptible **honnêtement** — ni `+inf` pour dire « pas d'objectif », ni
    horizon bricolé. Écrites pour le swing, conservées après sa mort (C6, 2026-09-25) : elles
    généralisent le contrat, elles ne l'adaptent pas :

    - **la barrière d'objectif est optionnelle** : `barriere_haute` (long) ou `barriere_basse`
      (short) peut valoir `None`, et alors **rien ne fait gagner avant la sortie**. La barrière du
      côté du **stop**, elle, est obligatoire : sans elle « risquer 1 % » n'a pas de définition,
      et le sizing divise par sa distance ;
    - **`dt_sortie`** : la décision peut porter sa propre sortie temporelle — la position est
      fermée à l'**Open** de cette barre-là, avant tout ce qui s'y passe (Cb2). `None` signifie
      « l'horizon global du label s'applique », le comportement d'avant, inchangé.
    """

    dt_entree: datetime
    direction: int
    prix_entree: float
    barriere_haute: float | None
    barriere_basse: float | None
    #: Score du modèle. Inutilisé ici ; l'ordonnanceur (4.9) départagera dessus (F14).
    score: float = float("nan")
    #: Sortie temporelle PROPRE à la décision (C1). `None` : l'horizon global du label.
    dt_sortie: datetime | None = None

    def __post_init__(self) -> None:
        if self.direction not in (LONG, SHORT):
            raise ValueError(f"direction doit valoir {LONG} (long) ou {SHORT} (short)")
        # ⚠️ La barrière du côté du STOP ne peut pas manquer : c'est elle qui définit le risque,
        #    donc la taille de la position. Une décision sans stop n'est pas « sans objectif »,
        #    elle est sans dimension.
        stop = self.barriere_basse if self.direction == LONG else self.barriere_haute
        if stop is None or not math.isfinite(stop):
            raise ValueError(
                f"la barrière du côté du stop est absente ou non finie ({stop!r}) : sans stop, le "
                "risque n'a pas de définition et le sizing ne peut rien dimensionner"
            )
        # Le stop est du côté qui fait perdre — sinon la distance de stop serait négative et le
        # sizing produirait une position INVERSÉE sans rien planter.
        du_bon_cote = stop < self.prix_entree if self.direction == LONG else stop > self.prix_entree
        if not du_bon_cote:
            raise ValueError(
                f"stop {stop} du mauvais côté de l'entrée {self.prix_entree} pour la direction "
                f"{self.direction}"
            )
        haute, basse = self.barriere_haute, self.barriere_basse
        if haute is not None and basse is not None and not basse < self.prix_entree < haute:
            raise ValueError(
                f"barrières incohérentes : basse={basse} < entrée={self.prix_entree} < "
                f"haute={haute} n'est pas vérifié"
            )
        if self.dt_sortie is not None and not self.dt_sortie > self.dt_entree:
            raise ValueError(
                f"dt_sortie ({self.dt_sortie}) doit être STRICTEMENT après dt_entree "
                f"({self.dt_entree}) : une sortie à la barre d'entrée est un trade qui n'existe pas"
            )

    @property
    def stop(self) -> float:
        """Le niveau qui fait perdre : bas pour un long, haut pour un short. Jamais absent."""
        s = self.barriere_basse if self.direction == LONG else self.barriere_haute
        assert s is not None  # garanti par __post_init__
        return s

    @property
    def cible(self) -> float | None:
        """Le niveau qui fait gagner — ou `None` : rien ne fait gagner avant la sortie (C1)."""
        return self.barriere_haute if self.direction == LONG else self.barriere_basse

    @property
    def distance_stop(self) -> float:
        """Distance en prix entre l'entrée et le stop. Sert à dimensionner la position."""
        return abs(self.prix_entree - self.stop)


@dataclass
class Position:
    """Une position ouverte. `unites` est telle que P&L = unites x mouvement de prix."""

    decision: Decision
    unites: float
    barre_ouverture: int
    #: Indice de la barre où la position est fermée à l'Open (C1) — `None` : horizon global.
    barre_sortie: int | None = None

    def pnl(self, prix: float) -> float:
        """P&L à un prix donné, signé par la direction."""
        return self.decision.direction * (prix - self.decision.prix_entree) * self.unites

    def pire_prix(self, haut: float, bas: float) -> float:
        """Le prix le plus défavorable de la barre — F4, on suppose toujours le pire."""
        return bas if self.decision.direction == LONG else haut

    def meilleur_prix(self, haut: float, bas: float) -> float:
        return haut if self.decision.direction == LONG else bas

    def distance_adverse(self, niveau: float) -> float:
        """
        Distance d'un niveau de prix depuis l'entrée, **comptée du côté qui fait mal**.

        Positive si le niveau est du mauvais côté, négative s'il est du bon. Cette mesure unique
        est ce qui permet de comparer un stop et une barrière de compte sans se soucier du sens
        du trade : F5 se réduit alors à « le plus petit gagne ».
        """
        return (self.decision.prix_entree - niveau) * self.decision.direction

    def prix_pour_equity(self, balance: float, equity_visee: float) -> float:
        """
        Le **prix** auquel l'équity du compte vaudrait `equity_visee`.

        C'est la conversion qui rend F5 applicable : une barrière de compte est un niveau
        d'équity, un stop est un niveau de prix. Tant qu'on ne les exprime pas dans la même
        unité, « lequel est touché en premier » n'a pas de sens.

        Inversion de `pnl` : `equity = balance + direction x (prix - entree) x unites`.
        """
        ecart = (equity_visee - balance) / (self.decision.direction * self.unites)
        return self.decision.prix_entree + ecart


@dataclass
class ResultatSimulation:
    """Ce qu'une simulation rapporte. Aucune moyenne : des comptages et des trajectoires."""

    issue: Issue
    balance_finale: float
    n_trades: int
    n_barres: int
    n_decisions_ignorees: int
    motif_bust: str | None
    #: Journées de trading comptées **dans la phase en cours** — c'est le compteur de la règle
    #: FTMO, il repart à zéro en phase 2. Ne JAMAIS s'en servir comme durée du challenge.
    jours_tradees: int
    #: Journées FTMO **écoulées depuis le début du challenge**, phases confondues. C'est la
    #: durée, et c'est elle qu'on médiane (tâche 4.6).
    n_journees: int = 0
    #: Trajectoire de l'équity à chaque barre, pour inspection et pour l'AIMD (phase 5).
    equity: list[float] = field(default_factory=list)
    #: Le risque **effectif** appliqué à chaque trade, dans l'ordre — airbag compris. C'est la
    #: preuve la plus directe qu'une config **fait réellement quelque chose** : un sizing
    #: adaptatif qui ne bouge jamais est un sizing constant qui ne le dit pas.
    risques: list[float] = field(default_factory=list)

    # ── diagnostic AIMD (tâche 5.6) ─────────────────────────────────────────────────────
    # Tous à zéro quand le sizing est constant. Sans eux, la phase 6 ne distingue pas une
    # config qui protège d'une config inerte (6.4) — et c'est exactement ce qu'elle doit
    # trancher sur les axes « seuil k » et « airbag ».
    #: Montées **effectives** de `m` (une gagnante au plafond n'en est pas une).
    n_montees: int = 0
    #: Coupes **effectives** de `m`.
    n_coupes: int = 0
    #: Déclenchements du seuil `k`, effet ou non. Toujours `>= n_coupes` : au plancher, la coupe
    #: se déclenche sans faire bouger `m`. C'est ce compteur qui dira si **k = 4 est inerte**.
    n_declenchements_coupe: int = 0
    #: Morsures de l'airbag (H16). S'il ne mord jamais sur le null, c'est une décoration — et
    #: H16 exige de le **vérifier**, pas de le supposer.
    n_declenchements_airbag: int = 0
    #: Décompte des quatre issues de journée, par nom. Répond à « combien de journées vides ? »
    #: (I13). ⚠️ Somme = journées **closes**, soit une de moins que les journées vécues : la
    #: dernière journée d'un chemin n'est jamais classée, aucune frontière ne la suit.
    bilans_journee: dict[str, int] = field(default_factory=dict)

    def resume(self) -> str:
        motif = f" ({self.motif_bust})" if self.motif_bust else ""
        return (
            f"{self.issue.value}{motif} · balance {self.balance_finale:,.0f} · "
            f"{self.n_trades} trade(s) sur {self.n_barres} barres · "
            f"{self.n_decisions_ignorees} décision(s) ignorée(s) (compte occupé)"
        )


def _valider_barres(barres: pd.DataFrame) -> None:
    manquantes = [c for c in COLONNES_BARRES if c not in barres.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du panel de barres : {manquantes}. Le simulateur ne CALCULE "
            "jamais la journée FTMO — il lit `jour_ftmo`, produit par `ftmo_cleaner`, pour qu'il "
            "n'existe qu'une seule notion de journée dans le projet (B11)."
        )
    if len(barres) == 0:
        raise ValueError("Panel de barres vide.")
    dt = pd.Series(barres["DateTime"])
    if not dt.is_monotonic_increasing:
        raise ValueError("Les barres ne sont pas triées par DateTime croissant.")


def simuler(
    decisions: list[Decision],
    barres: pd.DataFrame,
    risque_par_trade_pct: float | None = None,
    aimd: AIMDConfig | None = None,
    regles: ReglesFTMO | None = None,
    cout_par_trade: float = 0.0,
    unites_minimum: float = 0.0,
    pas_de_lot: float | None = None,
) -> ResultatSimulation:
    """
    Rejoue un challenge barre par barre.

    **Le sizing est obligatoire et sans défaut** (H9) : `risque_par_trade_pct` est la fraction
    du capital INITIAL risquée sur chaque trade, donc la perte encaissée si le stop est touché.
    Ne rien passer lève — « à peu près dimensionné » n'existe pas.

    **SOURCE UNIQUE de sizing** (H9, critère de recette de 5.6) : soit `risque_par_trade_pct`
    (constant), soit `aimd` (adaptatif) — **jamais les deux, jamais aucun**. Les deux ensemble
    laisseraient le code choisir en silence lequel gagne ; aucun des deux ferait tomber le
    dimensionnement sur un défaut implicite. Les deux refus ont leur **propre message**.

    **L'état du sizing est créé ICI**, et c'est ce qui rend H4 ② *structurelle* : `SuiviAIMD` est
    construit à chaque appel, donc **aucun objet ne peut être partagé entre deux chemins**. Une
    convention écrite n'aurait pas suffi — le jour où deux chemins partagent une mémoire,
    `P(funded)` cesse d'être une probabilité pour devenir un artefact d'ordonnancement, et le
    chiffre reste parfaitement plausible.

    **La barrière temporelle n'est pas un paramètre de `simuler`.** Soit la décision porte sa
    sortie (`dt_sortie`, C1) et la position est fermée à l'**Open** de cette barre ; soit elle
    n'en porte pas, et l'horizon du label s'applique, lu à sa source (F6, C3). En faire un
    réglage rouvrirait la porte que 4.4 a fermée : on pourrait juger des trades d'une durée que
    le label ne décrit pas, et le seul symptôme serait un P(funded) légèrement faux.
    """
    if risque_par_trade_pct is None and aimd is None:
        raise ValueError(
            "Le sizing est obligatoire : fournir `risque_par_trade_pct` (constant) OU `aimd` "
            "(adaptatif). Ne rien passer ferait tomber le dimensionnement sur une valeur par "
            "défaut, ce que H9 interdit."
        )
    if risque_par_trade_pct is not None and aimd is not None:
        raise ValueError(
            "Deux sources de sizing fournies (`risque_par_trade_pct` ET `aimd`) : il en faut "
            "EXACTEMENT une (H9). Les accepter toutes les deux laisserait le code choisir en "
            "silence laquelle l'emporte, et le P(funded) resterait plausible."
        )
    if risque_par_trade_pct is not None and not 0 < risque_par_trade_pct < 1:
        raise ValueError(
            f"risque_par_trade_pct est une FRACTION dans ]0, 1[, reçu {risque_par_trade_pct}. "
            "Aucun défaut n'est fourni (H9) : un risque hérité d'un périmètre abandonné survit "
            "sans que rien ne le signale."
        )
    # ⚠️ **`cout_par_trade` n'était validé nulle part — corrigé le 2026-08-18** (piège n°8).
    # Il vaut zéro jusqu'à la tâche 10.1, mais c'est précisément 10.1 qui commencera à y passer
    # de vraies valeurs. Un NaN mettrait la BALANCE à NaN dès le premier trade, et à partir de là
    # `observer` ne déclencherait plus jamais — toute comparaison avec NaN est fausse, y compris
    # le `<=` du plancher. Aucun bust, aucune cible : tous les chemins resteraient EN_COURS.
    # Un coût négatif serait un gain à l'ouverture, qui gonflerait le compte à chaque trade.
    if not math.isfinite(cout_par_trade) or cout_par_trade < 0:
        raise ValueError(
            f"cout_par_trade doit être un nombre fini >= 0, reçu {cout_par_trade!r}. Un NaN "
            "mettrait la balance à NaN et plus aucun bust ne serait jamais détecté ; un coût "
            "négatif serait un gain à chaque ouverture."
        )
    if not math.isfinite(unites_minimum) or unites_minimum < 0:
        raise ValueError(
            f"unites_minimum doit être un nombre fini >= 0, reçu {unites_minimum!r}. Zéro dit "
            "que la granularité de lot n'est PAS branchée (tâche 10.1) ; un négatif n'a aucun sens."
        )
    if pas_de_lot is not None:
        raise NotImplementedError(
            "L'arrondi au pas de lot demande la spec de contrat du broker et se branchera avec "
            "les coûts (tâche 10.1). Le laisser à None dit que la question n'est PAS réglée."
        )
    _valider_barres(barres)

    regles = regles if regles is not None else ReglesFTMO()
    suivi = SuiviChallenge(regles)
    # ⚠️ **Critère de recette ② de 5.6 — l'état AIMD est créé ICI, jamais reçu.** C'est ce qui
    # rend H4 ② structurelle plutôt que déclarative : un nouvel objet par appel de `simuler`,
    # donc par chemin, donc aucune mémoire ne peut franchir la frontière d'un challenge.
    # `regles` lui est passé explicitement — l'airbag y lit `capital_initial` (piège n°5).
    suivi_aimd: SuiviAIMD | None = SuiviAIMD(aimd, regles) if aimd is not None else None

    dt_barres = np.asarray(pd.Series(barres["DateTime"]).values)
    ouv = np.asarray(barres["Open"], dtype=float)
    haut = np.asarray(barres["High"], dtype=float)
    bas = np.asarray(barres["Low"], dtype=float)
    clo = np.asarray(barres["Close"], dtype=float)
    jours = np.asarray(pd.to_datetime(pd.Series(barres["jour_ftmo"])).dt.date)

    # Recherche binaire, jamais de scan linéaire (F12) : la boucle tourne des centaines de
    # milliers de fois sur une campagne.
    par_barre: dict[int, list[Decision]] = {}
    sorties: dict[Decision, int] = {}
    for d in sorted(decisions, key=lambda x: x.dt_entree):
        i = int(np.searchsorted(dt_barres, np.datetime64(d.dt_entree), side="left"))
        if i >= len(dt_barres) or dt_barres[i] != np.datetime64(d.dt_entree):
            raise ValueError(
                f"La barre d'entrée {d.dt_entree} n'existe pas dans le panel. Une décision doit "
                "pointer une barre réelle — sinon le juge évalue un trade impossible."
            )
        # Le prix d'entrée doit être l'Open de sa barre (C5). Le label le garantit ; un jeu de
        # décisions bricolé ailleurs, non. Sans ce contrôle, le juge simulerait des entrées à un
        # prix qui n'a jamais existé, et le seul symptôme serait un P(funded) faux.
        if not np.isclose(d.prix_entree, ouv[i], rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"Décision du {d.dt_entree} : prix d'entrée {d.prix_entree} != Open de la barre "
                f"({ouv[i]}). L'entrée se fait à l'Open de t+1 (C5) — un écart signale des "
                "décisions construites sur d'autres données que celles qu'on simule."
            )
        par_barre.setdefault(i, []).append(d)
        # C1 — la sortie temporelle propre à la décision doit pointer une barre réelle, après
        # l'entrée. Sinon le juge fermerait un trade à un instant qui n'existe pas dans le panel.
        if d.dt_sortie is not None:
            j = int(np.searchsorted(dt_barres, np.datetime64(d.dt_sortie), side="left"))
            if j >= len(dt_barres) or dt_barres[j] != np.datetime64(d.dt_sortie):
                raise ValueError(
                    f"La barre de sortie {d.dt_sortie} n'existe pas dans le panel : la décision "
                    "du {d.dt_entree} serait fermée à un instant qui n'a jamais été coté."
                )
            if j <= i:
                raise ValueError(
                    f"Décision du {d.dt_entree} : la barre de sortie {d.dt_sortie} n'est pas "
                    "après la barre d'entrée dans le panel."
                )
            sorties[d] = j

    # ⚠️ L'horizon global n'est lu — et le label triple-barrière importé — QUE si une décision en
    #    a besoin. Un jeu de décisions qui portent toutes leur sortie ne touche jamais l'ancien
    #    label : c'est ce qui rend sa mort (C6) inoffensive pour le juge.
    horizon: int | None = horizon_global() if any(d.dt_sortie is None for d in decisions) else None

    balance = regles.capital_initial
    position: Position | None = None
    jour_courant: date | None = None
    n_trades = n_ignorees = n_journees = 0
    trajectoire: list[float] = []
    risques: list[float] = []

    for i in range(len(barres)):
        # ── 1. Frontière de journée FTMO : le plancher journalier se recalcule ──────────
        if jours[i] != jour_courant:
            jour_courant = jours[i]
            n_journees += 1
            # L'ancre est la BALANCE à la frontière (règle FTMO réelle, alignée le 2026-08-10),
            # pas l'équity : un flottant ne déplace pas la ligne du jour — il compte dans la
            # violation, testée sur l'équity (F3). Aucun trade issu du label ne peut de toute
            # façon enjamber la frontière de 17:00 NY : la contiguïté C9 l'interdit.
            suivi.nouvelle_journee(jours[i], balance)
            # ⚠️ **C'est ICI que le sizing adaptatif s'ajuste, et nulle part ailleurs** (H2 ③).
            # La journée écoulée est classée sur sa variation de BALANCE (réalisé seul, H2 ①) et
            # la suivante s'ouvre — un seul appel, une seule balance, parce que la balance de
            # clôture d'une journée EST celle d'ouverture de la suivante.
            # Placer cet appel hors du `if` ferait réagir l'AIMD **trade par trade**, soit le
            # déclencheur que H2 a explicitement écarté. La garde de progression de journée lève
            # dans ce cas — c'est la moitié « B » de la double protection décidée le 16/08.
            if suivi_aimd is not None:
                suivi_aimd.frontiere_de_journee(jours[i], balance)

        # ── 1 bis. Sortie temporelle PROPRE à la décision : à l'Open, AVANT tout le reste ────
        # C1 (Cb2). On quitte la position à l'Open de sa barre de sortie : ni le High ni le Low
        # de cette barre ne peuvent plus rien déclencher, et le compte est libre dès cet instant —
        # une décision qui entre à cette même barre s'ouvre juste après, à ce même Open.
        if position is not None and position.barre_sortie == i:
            balance += position.pnl(ouv[i])
            position = None
            if suivi_aimd is not None:
                suivi_aimd.marquer_cloture_de_trade()

        # ── 2. Ouverture éventuelle, à l'Open de la barre ───────────────────────────────
        for d in par_barre.get(i, []):
            if position is not None:
                n_ignorees += 1  # compte occupé (H11) — l'ordonnanceur viendra en 4.9
                continue
            # Le multiplicateur courant est **lu**, pas calculé : la v2 ajuste à la frontière
            # de journée, jamais à l'ouverture d'un trade (H2 ③).
            #
            # ⚠️ **`risque_effectif` est appelée EXACTEMENT UNE FOIS PAR TRADE** — elle
            # incrémente `n_declenchements_airbag`. Un second appel (pour journaliser, pour un
            # assert) ferait lire une config comme **plus protectrice qu'elle ne l'est** en
            # phase 6, or c'est précisément la mesure qui doit dire si l'airbag est utile ou
            # décoratif (H16, 6.4).
            #
            # ⚠️ **`balance` et non l'équity** : aucune position n'est ouverte à cet instant —
            # le `continue` ci-dessus le garantit (un seul trade à la fois, H11) — donc les deux
            # sont égales. Si cette contrainte tombait un jour, c'est l'ÉQUITY qu'il faudrait
            # passer : c'est elle qui peut mourir. Le plancher est **lu** depuis le juge, jamais
            # recalculé, et c'est le plus proche des deux (piège n°5, H16).
            if suivi_aimd is not None:
                risque = suivi_aimd.risque_effectif(balance, suivi.plancher_effectif())
            else:
                risque = risque_par_trade_pct
            assert risque is not None  # garanti par la garde de source unique
            # H10 : seul le NOMBRE D'UNITÉS change. La distance du stop reste celle du label,
            # portée par l'ATR — deux mécanismes d'adaptation à la volatilité, jamais mêlés.
            unites = (regles.capital_initial * risque) / d.distance_stop
            # ⚠️ **LE PLANCHER DE TAILLE — décision de Hugo, 2026-08-22, lecture A.**
            #
            # En dessous du lot minimum du broker, l'ordre ne peut pas être passé : ce n'est pas
            # une convention de modélisation, c'est un fait de marché. Un compte réel ne se met
            # pas en pause pour autant — il passe le **minimum**, et donc **il sur-risque** par
            # rapport à ce que le sizing voulait.
            #
            # Sans ce plancher, l'airbag (H16) crée une **asymptote** : son plafond vaut
            # `marge / N`, donc chaque perte retire une FRACTION de ce qui reste et le plancher
            # n'est jamais atteint. Mesuré le 2026-08-21 sur 193 chemins : **zéro bust** avec
            # airbag contre 133 sans, pour un nombre de financés identique (59 contre 60).
            # `P(funded)` affichait alors **1,000** — il n'y avait plus aucun bust au
            # dénominateur. Le compte ne survivait pas, il tradait 5 $ sans jamais conclure.
            #
            # Le risque **effectif** est recalculé depuis les unités réellement prises : sinon
            # `risques` rapporterait ce qu'on voulait risquer, pas ce qui était en jeu.
            if unites < unites_minimum:
                unites = unites_minimum
                risque = (unites * d.distance_stop) / regles.capital_initial
            risques.append(risque)
            position = Position(
                decision=d, unites=unites, barre_ouverture=i, barre_sortie=sorties.get(d)
            )
            balance -= cout_par_trade
            n_trades += 1
            suivi.marquer_journee_tradee()

        # ── 3. Résolution intra-barre : le défavorable d'abord (F4), puis F5 ────────────
        if position is not None:
            d = position.decision
            pire = position.pire_prix(haut[i], bas[i])
            meilleur = position.meilleur_prix(haut[i], bas[i])

            # F5 — LE DÉPARTAGE. Le stop du trade et la barrière de compte sont deux niveaux de
            # PRIX du même côté de l'entrée. Le prix se déplace de façon monotone depuis
            # l'entrée : il croise donc forcément le plus PROCHE en premier. Ce n'est pas une
            # convention arbitraire, c'est le seul ordre physiquement possible.
            niveau_plancher = suivi.plancher_effectif()
            prix_plancher = position.prix_pour_equity(balance, niveau_plancher)
            d_stop = position.distance_adverse(d.stop)
            d_plancher = position.distance_adverse(prix_plancher)
            d_atteinte = position.distance_adverse(pire)

            # La barre d'entrée compte dans l'horizon (C6) : t+1..t+6 font 6 barres, donc le
            # « +1 ». Se tromper ici décale la clôture forcée d'une barre entière.
            # C1 : l'horizon global ne concerne QUE les décisions sans sortie propre — une
            # décision qui porte `dt_sortie` vit jusqu'à sa barre, quelle que soit sa durée.
            trop_long = (
                position.barre_sortie is None
                and horizon is not None
                and (i - position.barre_ouverture + 1) >= horizon
            )
            # C1 : sans objectif, rien ne fait gagner avant la sortie.
            cible = d.cible
            touche_cible = (
                False
                if cible is None
                else ((meilleur >= cible) if d.direction == LONG else (meilleur <= cible))
            )

            prix_sortie: float | None = None
            if d_atteinte >= min(d_stop, d_plancher):
                # Le mouvement adverse a franchi au moins un des deux niveaux.
                if d_plancher <= d_stop:
                    # Le compte meurt AVANT que le stop ne ferme le trade. À égalité stricte, le
                    # compte l'emporte : la convention pessimiste vaut aussi ici.
                    #
                    # L'équity au prix du plancher EST le plancher, par définition de
                    # `prix_pour_equity` — on l'affecte, on ne la reconstruit pas via `pnl()`.
                    # L'aller-retour division/multiplication ressort à ±1e-11 du plancher, et
                    # une équity à plancher + 1e-11 ne déclenche PAS le `<=` d'`observer` :
                    # le bust serait raté une fois sur deux, silencieusement (classe F18).
                    equity_pire = niveau_plancher
                    prix_sortie = prix_plancher
                else:
                    # Le stop ferme le trade en premier. L'équity ne descend PAS jusqu'au bas de
                    # la barre : la position n'existe plus au-delà du stop. C'est ce que la
                    # version 4.2 faisait à tort — elle présentait au compte une perte que le
                    # stop avait déjà empêchée.
                    equity_pire = balance + position.pnl(d.stop)
                    prix_sortie = d.stop
            else:
                # Rien n'est franchi : la position flotte jusqu'au pire prix de la barre.
                equity_pire = balance + position.pnl(pire)
                if touche_cible:
                    prix_sortie = cible
                elif trop_long:
                    prix_sortie = clo[i]  # clôture forcée au marché (F6)

            if prix_sortie is not None:
                balance += position.pnl(prix_sortie)
                position = None
                # ⚠️ À la FERMETURE, jamais à l'ouverture : le résultat d'une position tombe dans
                # la journée où elle se ferme (H2 ①). C'est ce compteur — et non la variation de
                # balance — qui sépare une journée « nulle » d'une journée « sans trade » (H15).
                if suivi_aimd is not None:
                    suivi_aimd.marquer_cloture_de_trade()
        else:
            equity_pire = balance

        # ── 4. Le juge tranche ──────────────────────────────────────────────────────────
        equity_fin = balance + (position.pnl(clo[i]) if position else 0.0)
        trajectoire.append(equity_fin)
        phase_avant = suivi.phase
        issue = suivi.observer(equity=min(equity_pire, equity_fin), balance=balance)
        if issue is not Issue.EN_COURS:
            break

        # ── 5. Passage en phase 2 : le compte est NEUF (F17) ────────────────────────────
        # Le simulateur doit repartir du capital initial, fermer ce qui traînait, et
        # redéclarer une journée. `SuiviChallenge` a d'ailleurs invalidé la sienne exprès :
        # sans ce bloc, l'`observer` de la barre suivante lèverait — le garde-fou fait son
        # travail plutôt que de laisser la phase 2 hériter du plancher de la phase 1.
        if phase_avant != suivi.phase:
            position = None  # le compte de la phase 1 est clos, sa position avec
            balance = regles.capital_de_depart(suivi.phase)
            suivi.nouvelle_journee(jours[i], balance)
            # Critère de recette ⑤ — compte neuf (F17), le capital repart de zéro et la
            # confiance aussi (H4 ③). ⚠️ `passage_de_phase` **abandonne** la journée en cours au
            # lieu de la clore : la balance vient de sauter au capital de départ, donc la classer
            # fabriquerait une journée gagnante de plusieurs milliers de dollars sans qu'aucun
            # trade n'ait eu lieu.
            if suivi_aimd is not None:
                suivi_aimd.passage_de_phase(jours[i], balance)

    return ResultatSimulation(
        issue=suivi.issue,
        balance_finale=balance,
        n_trades=n_trades,
        n_barres=len(trajectoire),
        n_decisions_ignorees=n_ignorees,
        motif_bust=suivi.motif_bust.value if suivi.motif_bust else None,
        jours_tradees=suivi.jours_tradees,
        n_journees=n_journees,
        equity=trajectoire,
        risques=risques,
        n_montees=suivi_aimd.n_montees if suivi_aimd else 0,
        n_coupes=suivi_aimd.n_coupes if suivi_aimd else 0,
        n_declenchements_coupe=suivi_aimd.n_declenchements_coupe if suivi_aimd else 0,
        n_declenchements_airbag=suivi_aimd.n_declenchements_airbag if suivi_aimd else 0,
        # Recopié en `dict[str, int]` plutôt que de rendre l'objet d'état : un résultat de
        # simulation ne doit pas porter une référence mutable vers la machine qui l'a produit.
        bilans_journee=({b.value: n for b, n in suivi_aimd.bilans.items()} if suivi_aimd else {}),
    )
