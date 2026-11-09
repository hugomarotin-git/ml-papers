"""
La métrique de payoff — tâche 4.1, décisions F0, F1, F2, F3.

**Ce module définit ce que « réussir » veut dire. Il ne simule rien.** Le simulateur
(tâche 4.2) produira des observations d'équity ; ce module dit, pour chaque observation, si le
challenge est gagné, perdu, ou toujours en cours. Les deux sont séparés pour que celui-ci soit
prouvable seul, sans données de marché.

────────────────────────────────────────────────────────────────────────────────────────
Le payoff est un FIRST-PASSAGE, pas un Sharpe (F0)
────────────────────────────────────────────────────────────────────────────────────────
`P(funded)` = probabilité d'**atteindre la cible avant de toucher la barrière**. C'est l'ordre
d'arrivée des événements qui compte, jamais la moyenne des rendements.

Un Sharpe élevé sur une trajectoire qui touche le plancher en chemin vaut **zéro** sur un compte
prop firm : le compte est fermé, la suite de la trajectoire n'existe pas. C'est pour ça qu'aucune
statistique de dispersion ne peut remplacer cette mesure, et pour ça qu'un trade à espérance
nulle n'est pas neutre — il ajoute de la variance, donc du risque de toucher la barrière en
chemin, donc il **fait baisser** `P(funded)` (cf. I13).

Conséquence pratique sur ce module : tout se joue dans l'**ordre** des tests. Un plancher est
toujours évalué **avant** une cible sur la même observation.

────────────────────────────────────────────────────────────────────────────────────────
Les deux planchers, à ne jamais confondre (F2)
────────────────────────────────────────────────────────────────────────────────────────
- **Plancher TOTAL** : -10 % du capital **INITIAL**. Statique. Il ne bouge **jamais**, ni quand
  le compte monte, ni quand il descend.
- **Plancher JOURNALIER** : balance de début de journée **moins 5 % du capital INITIAL** — un
  montant FIXE (2 500 $ sur 50k), quel que soit l'historique. C'est l'ancre qui bouge (la
  balance à minuit serveur), jamais le montant autorisé.

⚠️ **Aligné sur le règlement réel le 2026-08-10** (décision Hugo, source FTMO Academy) : la
lecture précédente — -5 % de l'ÉQUITY de début de journée — accordait 2 700 $ de marge à 54k de
balance et 2 400 $ à 48k, là où FTMO donne 2 500 $ partout. Optimiste précisément au-dessus de
50k, où vivent les chemins financés. Deux nuances à ne pas perdre :

- l'ancre officielle est la **BALANCE** à minuit CE(S)T, pas l'équity — un flottant porté au
  passage de journée ne déplace pas la ligne (il compte en revanche dans la violation, F3) ;
- le test de violation reste sur l'**équity**, flottant compris — ça, F3 l'avait toujours dit.

Ils se croisent : en début de challenge le journalier est le plus contraignant ; après une bonne
passe, le total peut le redevenir. Les deux sont testés à chaque observation, sans hypothèse sur
lequel mord.

⚠️ **La journée est la JOURNÉE FTMO** (B11), qui court de 18:00 à 16:00 NY et traverse minuit.
Sa frontière — 17:00 NY, minuit serveur — est **le même instant** que la remise à zéro du
plancher de perte FTMO. Le build précédent groupait par date calendaire : 7 heures d'écart tous
les jours, et un `P(funded)` faux sans qu'aucune erreur ne soit levée. Ce module n'accepte donc
pas de date : il exige qu'on lui **annonce** le changement de journée, et il refuse de travailler
si on ne l'a pas fait.

────────────────────────────────────────────────────────────────────────────────────────
Trois lectures de la spec qui changent le résultat — tranchées au PESSIMISTE
────────────────────────────────────────────────────────────────────────────────────────
Aucune n'est écrite noir sur blanc dans F1/F2/F3. Chacune est donc un **champ explicite** de
`ReglesFTMO`, pas une valeur en dur : on pourra mesurer ce que coûte l'autre lecture au lieu
d'en débattre.

1. **`cible_sur_equity = False`** — la cible se juge sur la **balance** (trades clôturés), les
   planchers sur l'**équity** (flottant inclus, F3). Asymétrie voulue : un profit flottant qu'on
   rend n'a jamais existé, alors qu'une perte flottante, elle, ferme le compte pour de bon.
   Compter la cible sur l'équity ferait « passer » des challenges sur un profit jamais encaissé.

2. **`cible_et_jours_simultanes = True`** — une phase se valide à l'instant où la cible **et**
   les 4 journées minimum sont réunies. Atteindre la cible au 2ᵉ jour ne valide rien : il faut
   continuer, donc rester exposé, et la cible doit être **de nouveau** satisfaite au 4ᵉ jour.
   C'est la lecture qui correspond à la réalité — on ne peut pas retirer un compte non validé —
   et c'est aussi la seule cohérente avec un first-passage.

3. **`phase2_repart_du_capital_initial = True`** — la phase 2 redémarre à 50 000, pas sur
   l'équity de fin de phase 1, et ses deux planchers se recalculent sur ce capital-là. Le gain
   de la phase 1 n'est pas un coussin : c'est un compte neuf.

────────────────────────────────────────────────────────────────────────────────────────
Ce que ce module ne fait PAS
────────────────────────────────────────────────────────────────────────────────────────
Pas de barres, pas de trades, pas de sizing, pas de coûts. **Les coûts de transaction ne sont
pas branchés** (F7) : tout `P(funded)` produit avant la tâche 10.1 est hors spread, hors
slippage, hors commission — et le spread du dev est structurellement moins cher que celui de la
période de déploiement (B10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

#: Règles standard FTMO 2-step, compte 50k (F1). Ce sont des faits externes — les règles du
#: broker — pas des seuils calibrés : A8 ne s'y applique pas. Elles restent néanmoins des champs
#: validés plutôt que des constantes disséminées, pour qu'un changement de barème se voie.
CAPITAL_INITIAL: float = 50_000.0
CIBLE_PHASE_1_PCT: float = 0.10
CIBLE_PHASE_2_PCT: float = 0.05
PLANCHER_TOTAL_PCT: float = 0.10
PLANCHER_JOURNALIER_PCT: float = 0.05
JOURS_MINIMUM: int = 4


class Issue(Enum):
    """
    L'issue d'un challenge. **Trois valeurs, pas deux.**

    `EN_COURS` n'est pas un détail de mise en œuvre : c'est une observation **censurée**, et la
    confondre avec un échec fausse `P(funded)` vers le bas, tandis que la jeter le fausse vers le
    haut. Les deux méthodes de la tâche 4.7 doivent la rapporter séparément (F10, F13).
    """

    FUNDED = "funded"
    BUST = "bust"
    EN_COURS = "en_cours"


class Phase(Enum):
    """Où en est le challenge. `TERMINE` couvre les deux fins possibles, funded comme bust."""

    UN = 1
    DEUX = 2
    TERMINE = 0


class MotifBust(Enum):
    """
    Pourquoi le compte a sauté. **À rapporter, jamais à agréger** : lequel des deux planchers
    mord est l'information dont l'**airbag** a besoin (H16) — il vise la marge jusqu'au plancher
    le **plus proche des deux**, parce que chez FTMO les deux tuent. Les confondre revient à
    jeter cette information.

    *(Ce commentaire citait H2 jusqu'au 2026-08-15 : le déclencheur du sizing était alors la
    distance à la barrière. Il est devenu le résultat net de la journée — mais le besoin de
    distinguer les deux planchers, lui, n'a pas disparu.)*
    """

    PLANCHER_JOURNALIER = "plancher_journalier"
    PLANCHER_TOTAL = "plancher_total"


@dataclass(frozen=True)
class ReglesFTMO:
    """
    Le barème d'un challenge, validé à la construction.

    Gelé (`frozen`) pour qu'une simulation ne puisse pas modifier ses propres règles en cours de
    route — c'est le genre de mutation qui produit un résultat flatteur et introuvable.
    """

    capital_initial: float = CAPITAL_INITIAL
    cible_phase1_pct: float = CIBLE_PHASE_1_PCT
    cible_phase2_pct: float = CIBLE_PHASE_2_PCT
    plancher_total_pct: float = PLANCHER_TOTAL_PCT
    plancher_journalier_pct: float = PLANCHER_JOURNALIER_PCT
    jours_minimum: int = JOURS_MINIMUM

    #: Les trois lectures pessimistes, cf. l'en-tête. Modifiables pour être MESURÉES, pas pour
    #: être choisies au jugé.
    cible_sur_equity: bool = False
    cible_et_jours_simultanes: bool = True
    phase2_repart_du_capital_initial: bool = True

    def __post_init__(self) -> None:
        # ⚠️ **Écrit en négation d'encadrement, pas en disjonction** — corrigé le 2026-08-18.
        # `nan <= 0` vaut False : la forme `if self.capital_initial <= 0` ACCEPTAIT un `NaN`.
        # Conséquence, et elle est totale : cible, plancher total et plancher journalier
        # devenaient tous `NaN`, donc `observer` ne déclenchait PLUS JAMAIS — toute comparaison
        # avec `NaN` est fausse, y compris le `<=` du plancher. Aucun challenge n'aurait pu
        # buster ni être funded : tous seraient restés EN_COURS, et `P(funded)` se serait
        # calculé sur zéro chemin conclu. `not x > 0` refuse le `NaN` (piège n°8).
        if not self.capital_initial > 0:
            raise ValueError(
                f"capital_initial doit être un nombre fini > 0, reçu {self.capital_initial!r}. "
                "Un NaN rendrait cible et planchers non comparables : plus aucun bust ne serait "
                "jamais détecté."
            )
        for nom, valeur in (
            ("cible_phase1_pct", self.cible_phase1_pct),
            ("cible_phase2_pct", self.cible_phase2_pct),
            ("plancher_total_pct", self.plancher_total_pct),
            ("plancher_journalier_pct", self.plancher_journalier_pct),
        ):
            if not 0 < valeur < 1:
                raise ValueError(f"{nom} est une FRACTION dans ]0, 1[, reçu {valeur}")
        if self.jours_minimum < 0:
            raise ValueError(f"jours_minimum doit être >= 0, reçu {self.jours_minimum}")
        if self.plancher_journalier_pct >= self.plancher_total_pct:
            # Sinon le plancher total ne peut jamais mordre en premier, et toute la logique de
            # « lequel des deux est le plus contraignant » devient du code mort silencieux.
            raise ValueError(
                f"plancher_journalier_pct ({self.plancher_journalier_pct}) doit être < "
                f"plancher_total_pct ({self.plancher_total_pct}) — sinon le plancher total est "
                "inatteignable et l'arbitrage entre les deux ne veut plus rien dire."
            )

    def capital_de_depart(self, phase: Phase) -> float:
        """Le capital sur lequel se calculent cible et planchers, pour la phase donnée."""
        if phase is Phase.TERMINE:
            raise ValueError("Une phase terminée n'a pas de capital de départ.")
        if phase is Phase.UN or self.phase2_repart_du_capital_initial:
            return self.capital_initial
        raise NotImplementedError(
            "phase2_repart_du_capital_initial=False n'est pas implémenté : le capital de départ "
            "de la phase 2 dépendrait alors de l'équity de fin de phase 1, que ce module ne "
            "connaît pas. C'est au simulateur (4.2) de le fournir s'il faut mesurer cette lecture."
        )

    def cible(self, phase: Phase) -> float:
        """
        Le niveau à atteindre, en valeur absolue.

        ⚠️ **Forme additive `base + base * pct`, jamais `base * (1 + pct)`.** Les deux sont
        équivalentes en algèbre, pas en flottant : `50_000 * 1.10` vaut `55000.00000000001`,
        tandis que `50_000 + 50_000 * 0.10` vaut exactement `55000.0`. Avec la première, une
        balance de 55 000,00 pile **ne validerait pas** la phase — un seuil décalé de 1e-11,
        parfaitement invisible, qui ne se manifesterait que sur les cas limites.
        """
        base = self.capital_de_depart(phase)
        pct = self.cible_phase1_pct if phase is Phase.UN else self.cible_phase2_pct
        return base + base * pct

    def plancher_total(self, phase: Phase) -> float:
        """
        Le plancher **statique** de la phase : -10 % du capital de départ.

        Il ne dépend d'aucun état courant — c'est précisément ce qui le distingue du journalier,
        et l'erreur classique est de le faire suivre l'équity.
        """
        base = self.capital_de_depart(phase)
        return base - base * self.plancher_total_pct

    def plancher_journalier(self, balance_debut_journee: float) -> float:
        """
        Le plancher **mobile** : balance de début de journée FTMO **moins 5 % du capital
        INITIAL** — le montant autorisé est fixe (2 500 $ sur 50k), c'est l'ancre qui bouge.

        Règle FTMO réelle (alignée le 2026-08-10) : « balance à minuit CE(S)T - 5 % du capital
        initial ». L'ancienne lecture (-5 % de l'équity du jour) faisait respirer le montant
        avec le compte — optimiste au-dessus de 50k, pessimiste en dessous.
        """
        return balance_debut_journee - self.capital_initial * self.plancher_journalier_pct


class SuiviChallenge:
    """
    **La métrique elle-même** : la machine à états qui décide, observation par observation, si la
    cible est atteinte avant la barrière.

    Elle ne connaît ni barres ni trades — seulement une suite d'observations `(equity, balance)`
    et des frontières de journée FTMO annoncées. C'est ce qui la rend prouvable sans données de
    marché, et c'est le point de la séparation 4.1 / 4.2.

    Emploi ::

        suivi = SuiviChallenge(ReglesFTMO())
        suivi.nouvelle_journee(balance=50_000.0)    # obligatoire avant toute observation
        suivi.marquer_journee_tradee()              # dès qu'un trade s'ouvre ce jour-là
        issue = suivi.observer(equity=50_120.0, balance=50_000.0)

    Trois garde-fous, tous là parce qu'ils correspondent à un bug déjà rencontré ou évident :

    - **observer sans avoir annoncé de journée lève une erreur.** C'est ce qui rend impossible le
      bug du build précédent — grouper les journées à minuit calendaire au lieu de 17:00 NY. Ici,
      il n'y a pas de date du tout : c'est l'appelant qui possède la notion de `jour_ftmo` (B11),
      et il doit la déclarer.
    - **observer après la fin lève une erreur.** Continuer à trader un compte fermé produit un
      `P(funded)` faux et parfaitement silencieux.
    - **le passage en phase 2 invalide la journée en cours.** La phase 2 est un compte NEUF : le
      simulateur est forcé de redéclarer une journée avec le capital de départ, il ne peut pas
      hériter par distraction du plancher journalier de la phase 1.
    """

    def __init__(self, regles: ReglesFTMO | None = None) -> None:
        self.regles: ReglesFTMO = regles if regles is not None else ReglesFTMO()
        self.phase: Phase = Phase.UN
        self.issue: Issue = Issue.EN_COURS
        self.motif_bust: MotifBust | None = None
        self.jours_tradees: int = 0
        self._jour: date | None = None
        self._balance_debut_journee: float | None = None
        self._journee_deja_comptee: bool = False
        self._cible_deja_atteinte: bool = False

    # ── état ────────────────────────────────────────────────────────────────────────────
    @property
    def termine(self) -> bool:
        """Vrai dès que l'issue est scellée, dans un sens ou dans l'autre."""
        return self.issue is not Issue.EN_COURS

    @property
    def equity_depart_phase(self) -> float:
        """Le capital de départ de la phase en cours."""
        return self.regles.capital_de_depart(self.phase)

    def plancher_effectif(self) -> float:
        """
        **Le plancher qui mord réellement** : le plus HAUT des deux, donc le plus contraignant.

        Une équity qui descend croise forcément le plus haut en premier — c'est le même argument
        de monotonie que F5, transposé au niveau du compte. Ce niveau unique est aussi celui que
        l'**airbag** devra lire pour mesurer la marge restante avant la mort (H16). *(Citait
        H2 et H3 jusqu'au 2026-08-15, toutes deux périmées depuis — H2 réécrite, H3 abrogée.)*
        """
        if self._balance_debut_journee is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `nouvelle_journee` d'abord. Le plancher "
                "journalier n'existe pas tant que la journée n'a pas été annoncée."
            )
        return max(
            self.regles.plancher_journalier(self._balance_debut_journee),
            self.regles.plancher_total(self.phase),
        )

    # ── pilotage ────────────────────────────────────────────────────────────────────────
    def nouvelle_journee(self, jour: date | None, balance: float) -> None:
        """
        Annonce une nouvelle journée FTMO et **repose le plancher journalier** sur la BALANCE
        de début de journée : `balance - 5 % du capital initial` (règle FTMO réelle, 2026-08-10).

        ────────────────────────────────────────────────────────────────────────────────
        La garde de progression stricte — tâche 4.12, dette ouverte le 2026-08-16
        ────────────────────────────────────────────────────────────────────────────────
        `jour` est le `jour_ftmo` de la journée qui s'ouvre. Il est **lu**, jamais dérivé d'un
        horodatage ici : l'appelant reste seul propriétaire de cette notion (B11).

        **Contre quoi.** Cette méthode croyait l'appelant sur parole. Appelée à chaque **barre**
        au lieu de chaque **frontière de journée** — une ligne au mauvais niveau d'indentation —
        le plancher journalier se reposerait à chaque barre et **suivrait la balance** : il ne
        mordrait pratiquement plus jamais, presque tous les busts journaliers disparaîtraient, et
        **`P(funded)` serait gonflé sans le moindre symptôme**. Rien ne planterait.

        C'est le trou que la tâche 5.3 avait fermé côté `SuiviAIMD` et que ce module portait
        encore. L'asymétrie était **connue et inscrite comme dette**, avec la phase 6 pour
        échéance : c'est là que `P(funded)` cesse d'être un chiffre de preuve pour devenir un
        résultat qu'on lit.

        ⚠️ **`jour=None` est un choix EXPLICITE, jamais un oubli** — le paramètre n'a pas de
        défaut. Il dit « je n'ai aucune notion de journée à faire vérifier », et c'est le cas de
        `methode_b` : elle rejoue des trades déjà résolus, sans barres ni dates, en regroupant sur
        un drapeau. Lui faire fabriquer des dates serait inventer de la donnée pour satisfaire une
        garde. La désactivation s'écrit, comme celle de l'airbag (H9).

        C'est bien la balance, pas l'équity : un flottant porté au passage de journée ne déplace
        pas l'ancre FTMO — il comptera en revanche dans la violation, qui se teste sur l'équity
        (F3). Dans le simulateur réel la nuance est morte (aucun trade ne peut enjamber la
        frontière de 17:00 NY, la contiguïté C9 l'interdit), mais la règle doit être juste même
        sur les cas qu'on fabrique à la main.

        À appeler à chaque frontière de `jour_ftmo` (17:00 NY, minuit serveur), y compris sur les
        journées sans aucun trade : le plancher se recalcule, que l'on trade ou non.
        """
        if self.termine:
            raise RuntimeError(f"Challenge déjà terminé ({self.issue.value}) : plus de journée.")
        if jour is not None and self._jour is not None and jour <= self._jour:
            raise RuntimeError(
                f"Journée FTMO annoncée : {jour}, alors que {self._jour} est déjà ouverte. Une "
                "frontière doit STRICTEMENT avancer. Deux appels pour la même journée signalent "
                "un appel placé dans la boucle des barres au lieu du bloc de frontière — le "
                "plancher journalier suivrait alors la balance et ne mordrait plus jamais, sans "
                "que rien d'autre ne le signale."
            )
        self._jour = jour
        self._balance_debut_journee = balance
        self._journee_deja_comptee = False

    def marquer_journee_tradee(self) -> None:
        """
        Compte la journée en cours dans les `jours_minimum`. Idempotent : dix trades dans la même
        journée FTMO ne font qu'**une** journée de trading.
        """
        if self._balance_debut_journee is None:
            raise RuntimeError("Aucune journée FTMO en cours : appeler `nouvelle_journee` d'abord.")
        if not self._journee_deja_comptee:
            self._journee_deja_comptee = True
            self.jours_tradees += 1

    def observer(self, equity: float, balance: float) -> Issue:
        """
        Le cœur du first-passage. **L'ordre des tests EST la métrique.**

        Les planchers sont évalués **avant** la cible : sur une observation qui franchit les deux,
        le compte est mort. Toute autre convention fabrique des challenges réussis par une équity
        qui n'a existé qu'après avoir déjà fermé le compte.
        """
        if self.termine:
            raise RuntimeError(
                f"Challenge déjà terminé ({self.issue.value}) : observer un compte fermé produit "
                "un P(funded) faux et silencieux."
            )
        if self._balance_debut_journee is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `nouvelle_journee` avant d'observer. "
                "Le plancher journalier se recalcule à 17:00 NY (B11), et ce module ne devine "
                "pas les frontières — il exige qu'on les lui annonce."
            )

        # 1. LES PLANCHERS D'ABORD — le compte peut être mort avant d'être gagnant.
        plancher_j = self.regles.plancher_journalier(self._balance_debut_journee)
        plancher_t = self.regles.plancher_total(self.phase)
        if equity <= max(plancher_j, plancher_t):
            # Lequel a mordu : le plus haut des deux, croisé en premier par une équity qui
            # descend. À égalité stricte, le journalier — c'est celui qui se referme le plus vite.
            self.motif_bust = (
                MotifBust.PLANCHER_JOURNALIER
                if plancher_j >= plancher_t
                else MotifBust.PLANCHER_TOTAL
            )
            self.issue = Issue.BUST
            self.phase = Phase.TERMINE
            return self.issue

        # 2. LA CIBLE ENSUITE — sur la balance par défaut, jamais sur un profit flottant.
        valeur = equity if self.regles.cible_sur_equity else balance
        atteinte_maintenant = valeur >= self.regles.cible(self.phase)
        if atteinte_maintenant:
            self._cible_deja_atteinte = True

        # Les deux lectures de la règle des 4 jours. En mode simultané (défaut, pessimiste) la
        # cible doit être satisfaite À L'INSTANT où le compte de journées est complet : l'avoir
        # touchée au 2ᵉ jour puis rendue ne vaut rien. En mode « acquis », l'avoir touchée une
        # fois suffit et on attend seulement les journées.
        cible_ok = (
            atteinte_maintenant
            if self.regles.cible_et_jours_simultanes
            else self._cible_deja_atteinte
        )
        if cible_ok and self.jours_tradees >= self.regles.jours_minimum:
            self._valider_phase()

        return self.issue

    def _valider_phase(self) -> None:
        """Passage de phase, ou compte financé."""
        if self.phase is Phase.UN:
            self.phase = Phase.DEUX
            self.jours_tradees = 0
            self._cible_deja_atteinte = False
            # Compte NEUF : la journée en cours ne vaut plus rien, le simulateur doit la
            # redéclarer avec le capital de départ de la phase 2.
            #
            # ⚠️ **L'ancre de journée est remise à None elle aussi** (4.12) : le simulateur doit
            # redéclarer la journée EN COURS, donc avec le même `jour_ftmo`, ce que la garde de
            # progression stricte refuserait. Un compte neuf repart sans passé, ancre de journée
            # comprise. La garde se réarme dès la déclaration suivante.
            self._jour = None
            self._balance_debut_journee = None
            self._journee_deja_comptee = False
        else:
            self.issue = Issue.FUNDED
            self.phase = Phase.TERMINE

    def resume(self) -> str:
        """Une ligne lisible, à imprimer après un run."""
        motif = f" ({self.motif_bust.value})" if self.motif_bust is not None else ""
        return (
            f"{self.issue.value}{motif} · phase {self.phase.value} · "
            f"{self.jours_tradees} journée(s) tradée(s)"
        )
