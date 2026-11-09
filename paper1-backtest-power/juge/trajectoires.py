"""
Simulation full-path et agrégation — tâche 4.6, décisions F0, F8, F13.

Un challenge unique ne dit rien : il est gagné ou perdu, c'est un tirage. **P(funded) se mesure
en rejouant beaucoup de challenges** et en comptant. Ce module fait ce comptage ; le choix de
*comment* on génère les départs — fenêtres glissantes (méthode A) ou bootstrap par blocs
(méthode B) — est la tâche 4.7.

────────────────────────────────────────────────────────────────────────────────────────
On COMPTE, on ne moyenne jamais
────────────────────────────────────────────────────────────────────────────────────────
`P(funded)` est une proportion de chemins, pas la moyenne d'un indicateur. La nuance a l'air
scolastique et ne l'est pas : moyenner des `P(funded)` calculés sur des sous-ensembles de
tailles différentes donne un nombre qui ne correspond à aucune expérience.

────────────────────────────────────────────────────────────────────────────────────────
Les chemins CENSURÉS sont une troisième catégorie, jamais un échec ni un déchet
────────────────────────────────────────────────────────────────────────────────────────
Un chemin qui atteint la fin des données sans être ni financé ni busté est **censuré**
(`EN_COURS`). Ce n'est pas un échec : c'est une question sans réponse. L'historique s'est
arrêté avant le challenge.

**Convention retenue — décision F19, tranchée par Hugo le 2026-08-09 :**

> `p_funded` = financés / **chemins qui ont conclu**. Les non conclus sont écartés du calcul,
> et leur nombre est rapporté à côté, systématiquement.

Ce qui a tranché : **un challenge FTMO n'a aucune limite de temps.** Un chemin qui n'a pas
conclu n'a donc rien perdu — il n'est simplement pas fini. Le mettre au dénominateur reviendrait
à répondre « perdu » à une question dont on n'a pas la réponse, et à infliger une pénalité
mécanique d'un facteur `(1 - c)` sans contrepartie.

*(Jusqu'au 2026-08-09, les censurés étaient au dénominateur. La justification écrite ici —
« les exclure surestime P(funded), c'est la version qui flatte » — était **fausse** : exclure
une censure à droite administrative est le traitement standard, et ne flatte que si les censurés
avaient une moindre chance de réussite. Conservée comme `p_funded_borne_basse`.)*

⚠️ **LE PIÈGE DE CETTE CONVENTION, et pourquoi `p_censure` se lit avec le chiffre, jamais après.**

Un système **très lent** — sizing minuscule, cadence rare — busterait peu et financerait
rarement : la plupart de ses chemins ne concluraient pas. S'il en résout dix et en finance huit,
il affiche fièrement `p_funded = 80 %` sur 10 chemins... et 90 % de censure. Le chiffre est
juste, la lecture est absurde.

C'est exactement ce qui peut arriver pendant la calibration de l'AIMD (H5-H8), où l'on compare
des configurations de sizing : la plus prudente gagnerait sur `p_funded` sans rien prouver.
D'où l'avertissement automatique de F13 au-delà de 10 % de censure — et d'où le fait que le
vrai combat est de **faire baisser la censure**, en n'admettant que des départs qui ont assez de
piste devant eux (tâche 4.7), plutôt que de débattre du dénominateur.

Second biais à connaître, plus subtil : les chemins résolus **sur-représentent les issues
rapides**. Un bust journalier peut tomber au 3ᵉ jour, un financement demande au minimum
8 journées de trading plus le temps d'atteindre +10 % puis +5 %. À censure élevée, l'ensemble
résolu est donc enrichi en busts, et `p_funded` **sous-estime**. Le diagnostic est instrumenté :
`duree_mediane_funded` et `duree_mediane_bust` sont calculées séparément — si la première est
bien plus grande que la seconde, ce biais est actif.

────────────────────────────────────────────────────────────────────────────────────────
La durée médiane se calcule par ISSUE, jamais toutes issues confondues
────────────────────────────────────────────────────────────────────────────────────────
« Le challenge dure en médiane 40 jours » n'a aucun sens si la moitié des chemins sont des
busts survenus au 3ᵉ jour. Un bust rapide et un financement lent se mélangent en une valeur qui
ne décrit aucun des deux. On rapporte donc **deux médianes séparées**, et les censurés n'en ont
aucune — leur durée est justement inconnue.

────────────────────────────────────────────────────────────────────────────────────────
Ce que ce module ne fait PAS
────────────────────────────────────────────────────────────────────────────────────────
Il ne choisit pas les départs (4.7), n'ordonnance pas les trades (4.9), et **n'applique aucun
coût** (F7). Tout P(funded) produit avant la tâche 10.1 est hors spread, hors slippage, hors
commission.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from aimd import AIMDConfig
from regles_ftmo import Issue, ReglesFTMO
from simulateur import Decision, ResultatSimulation, simuler

#: Au-delà de cette part de chemins censurés, le résultat est annoncé comme peu concluant (F13).
SEUIL_CENSURE_ALARMANT: float = 0.10


@dataclass(frozen=True)
class Distribution:
    """
    L'agrégat sur N chemins. **Des comptages et des proportions, aucune moyenne de probabilité.**

    Se lit toujours en entier : `p_funded` seule, sans `p_censure` à côté, peut décrire aussi
    bien un système solide qu'une série trop courte pour conclure.
    """

    n_chemins: int
    n_funded: int
    n_bust: int
    n_censures: int
    #: Durées en journées FTMO, calculées séparément par issue — jamais mélangées.
    duree_mediane_funded: float | None
    duree_mediane_bust: float | None
    #: Combien de busts par plancher. Lequel mord alimente l'**airbag** (H16), qui mesure la
    #: marge jusqu'au plancher le PLUS PROCHE des deux — chez FTMO les deux tuent. *(Cette ligne
    #: citait H2 jusqu'au 2026-08-15 : le déclencheur de l'AIMD n'est plus la distance à la
    #: barrière mais le résultat net de la journée.)*
    motifs_bust: dict[str, int] = field(default_factory=dict)

    # ── diagnostic AIMD agrégé (tâche 5.6) ──────────────────────────────────────────────
    # **Des TOTAUX sur tous les chemins**, pas des moyennes : on compte, on ne moyenne jamais
    # (F8). Tous à zéro quand le sizing est constant. Sans eux, la phase 6 ne distingue pas une
    # config qui protège d'une config inerte (6.4) — et deux valeurs d'axe sont déjà suspectes :
    # le seuil **k = 4** (il faudrait 8 journées perdantes consécutives pour la seconde coupe,
    # or la plus longue série du dev en fait 8) et l'**airbag désactivé**.
    n_montees: int = 0
    n_coupes: int = 0
    n_declenchements_coupe: int = 0
    n_declenchements_airbag: int = 0
    #: Décompte des quatre issues de journée, cumulé sur tous les chemins. Répond à « combien de
    #: journées vides ? » (I13), qui pourrait être le cas majoritaire avec le plancher I13.
    bilans_journee: dict[str, int] = field(default_factory=dict)

    @property
    def n_resolus(self) -> int:
        """Les chemins qui ont conclu, dans un sens ou dans l'autre. Le dénominateur du verdict."""
        return self.n_funded + self.n_bust

    @property
    def p_funded(self) -> float:
        """
        **Le chiffre du verdict** — la proportion de financés **parmi les chemins qui ont
        conclu**. Décision F19 (2026-08-09).

        Les censurés ne sont pas au dénominateur : on ne sait pas ce qu'ils seraient devenus,
        et les compter comme des pertes répondrait « perdu » à une question sans réponse.
        Ils ne disparaissent pas pour autant — `p_censure` les rapporte, toujours.
        """
        return self.n_funded / self.n_resolus if self.n_resolus else 0.0

    @property
    def p_bust(self) -> float:
        """
        Complément de `p_funded` sur les résolus : les deux somment à 1.

        Rapportée quand même (H7 : jamais une seule métrique), mais l'information indépendante
        est désormais portée par `p_censure`, pas par celle-ci.
        """
        return self.n_bust / self.n_resolus if self.n_resolus else 0.0

    @property
    def p_censure(self) -> float:
        """
        Part des chemins qui n'ont pas conclu, **sur l'ensemble des chemins**.

        ⚠️ **À lire systématiquement avec `p_funded`, jamais après.** C'est elle qui empêche le
        piège décrit dans l'en-tête du module : un système très lent peut afficher un `p_funded`
        magnifique sur une poignée de chemins résolus.
        """
        return self.n_censures / self.n_chemins if self.n_chemins else 0.0

    @property
    def p_funded_borne_basse(self) -> float:
        """
        Ce que vaudrait P(funded) si **tous** les censurés finissaient par échouer.

        Borne basse, conservée pour l'encadrement : `p_funded_borne_basse = p_funded x (1 - c)`.
        C'était le chiffre du verdict jusqu'au 2026-08-09 — abandonné parce que, sans limite de
        temps sur un challenge FTMO, il affirme un échec qu'on n'a pas observé.
        """
        return self.n_funded / self.n_chemins if self.n_chemins else 0.0

    def avertissements(self) -> list[str]:
        """Ce qui doit être dit à voix haute avec le chiffre, jamais en note de bas de page."""
        messages: list[str] = []
        if self.n_chemins == 0:
            return ["Aucun chemin simulé : il n'y a rien à interpréter."]
        if self.n_resolus == 0:
            return [
                f"AUCUN chemin n'a conclu sur {self.n_chemins} : P(funded) n'a pas de "
                "dénominateur et vaut 0 par convention. Ce n'est pas un résultat."
            ]
        if self.p_censure > SEUIL_CENSURE_ALARMANT:
            messages.append(
                f"CENSURE ÉLEVÉE : {100 * self.p_censure:.0f} % des chemins n'ont pas conclu "
                f"(seuil {100 * SEUIL_CENSURE_ALARMANT:.0f} %, F13). P(funded) ne porte donc que "
                f"sur {self.n_resolus} chemins sur {self.n_chemins}, et il est calculé sur un "
                "sous-ensemble qui SUR-REPRÉSENTE les issues RAPIDES. ⚠️ Le sens du biais dépend "
                "de la configuration, il ne se devine pas : sans airbag les busts sont rapides "
                "(52 j contre 45 pour un funded) et P(funded) est tiré vers le BAS ; avec airbag "
                "ils deviennent lents (356 j) et il est tiré vers le HAUT — mesuré le 2026-08-19. "
                f"Encadrement : {100 * self.p_funded_borne_basse:.1f} % si tous les censurés "
                f"échouaient, {100 * self.p_funded:.1f} % s'ils suivaient les conclus."
            )
        if self.n_resolus < 30:
            messages.append(
                f"ÉCHANTILLON MINCE : {self.n_resolus} chemins résolus. Une proportion sur si peu "
                "de tirages n'a pas d'intervalle de confiance exploitable."
            )
        if self.n_funded == 0:
            messages.append("AUCUN chemin financé : P(funded) = 0, ce n'est pas un arrondi.")
        return messages

    def resume(self) -> str:
        """Le bloc à imprimer. Les trois proportions ensemble, toujours."""
        med_f = f"{self.duree_mediane_funded:.0f} j" if self.duree_mediane_funded else "—"
        med_b = f"{self.duree_mediane_bust:.0f} j" if self.duree_mediane_bust else "—"
        motifs = " · ".join(f"{k} {v}" for k, v in sorted(self.motifs_bust.items())) or "—"
        lignes = [
            f"  chemins            : {self.n_chemins}  dont {self.n_resolus} conclus",
            f"  P(funded)          : {100 * self.p_funded:5.1f} %   ({self.n_funded}/{self.n_resolus} conclus)",
            f"  P(bust)            : {100 * self.p_bust:5.1f} %   ({self.n_bust}/{self.n_resolus} conclus)",
            f"  non conclus        : {100 * self.p_censure:5.1f} %   ({self.n_censures}/{self.n_chemins})",
            f"  borne basse        : {100 * self.p_funded_borne_basse:5.1f} %   (si tous les non conclus échouaient)",
            f"  durée médiane      : funded {med_f} · bust {med_b}",
            f"  motifs de bust     : {motifs}",
            "  ⚠️ hors coûts de transaction (F7) — spread, slippage et commission non branchés",
        ]
        return "\n".join(lignes + [f"  ⚠️ {m}" for m in self.avertissements()])


def agreger(resultats: list[ResultatSimulation]) -> Distribution:
    """
    Transforme une liste de chemins joués en distribution.

    Fonction **pure** : aucune donnée de marché, aucun tirage. C'est ce qui permet de la prouver
    sur des issues fabriquées à la main, où la bonne réponse se calcule de tête.
    """
    funded = [r for r in resultats if r.issue is Issue.FUNDED]
    bust = [r for r in resultats if r.issue is Issue.BUST]
    censures = [r for r in resultats if r.issue is Issue.EN_COURS]

    if len(funded) + len(bust) + len(censures) != len(resultats):
        raise RuntimeError(
            "Une issue inconnue traîne dans les résultats : les trois catégories doivent couvrir "
            "exactement l'ensemble, sinon les proportions ne somment plus à 1."
        )

    motifs: dict[str, int] = {}
    for r in bust:
        cle = r.motif_bust or "inconnu"
        motifs[cle] = motifs.get(cle, 0) + 1

    bilans: dict[str, int] = {}
    for r in resultats:
        for nom, n in r.bilans_journee.items():
            bilans[nom] = bilans.get(nom, 0) + n

    return Distribution(
        n_chemins=len(resultats),
        n_funded=len(funded),
        n_bust=len(bust),
        n_censures=len(censures),
        duree_mediane_funded=statistics.median([r.n_journees for r in funded]) if funded else None,
        duree_mediane_bust=statistics.median([r.n_journees for r in bust]) if bust else None,
        motifs_bust=motifs,
        n_montees=sum(r.n_montees for r in resultats),
        n_coupes=sum(r.n_coupes for r in resultats),
        n_declenchements_coupe=sum(r.n_declenchements_coupe for r in resultats),
        n_declenchements_airbag=sum(r.n_declenchements_airbag for r in resultats),
        bilans_journee=bilans,
    )


def jouer_chemin(
    decisions: list[Decision],
    barres: pd.DataFrame,
    depart: int,
    risque_par_trade_pct: float | None = None,
    aimd: AIMDConfig | None = None,
    regles: ReglesFTMO | None = None,
    budget_journees: int | None = None,
    unites_minimum: float = 0.0,
) -> ResultatSimulation:
    """
    Joue **un** challenge complet en partant de la décision d'indice `depart`.

    Le sizing est **à source unique** et obligatoire — `simuler` le vérifie (H9) : constant OU
    adaptatif, jamais les deux ni aucun. ⚠️ **Chaque chemin repart à `m₀`** (H4 ②), parce que
    `simuler` construit un `SuiviAIMD` neuf à chaque appel : un départ différent est un challenge
    différent, il n'hérite d'aucune confiance. La garantie est **structurelle**, pas déclarative —
    aucun objet d'état n'est partagé, il n'y a rien à oublier de réinitialiser.

    Les barres antérieures à cette décision sont écartées : un challenge qui démarre le 12 mars
    ne doit pas voir février. Les laisser fausserait la toute première journée FTMO — son
    plancher se poserait sur une barre qui précède l'ouverture du compte.

    Les décisions sont supposées **triées par date d'entrée**. On ne les retrie pas ici : un tri
    silencieux masquerait une séquence mal construite en amont, et la tâche 4.9 doit pouvoir
    compter sur l'ordre réel.

    ────────────────────────────────────────────────────────────────────────────────
    `budget_journees` — le même temps pour tous les chemins (décision de Hugo, 2026-08-21)
    ────────────────────────────────────────────────────────────────────────────────
    Sans budget, la fenêtre d'un chemin va de son départ à la **fin des données** : un chemin qui
    part en 2021 a quatre ans devant lui, un qui part en 2024 quelques mois. Deux configurations
    comparées sur ces chemins ne sont donc **pas comparées sur le même temps**.

    Le défaut est silencieux et il n'est pas symétrique. **Mesuré le 2026-08-19** : à 2 % de
    risque, un bust médian tombe en 52 journées **sans** airbag mais en **356 avec**, tandis que
    le funded médian reste à 45 dans les deux cas. La troncature supprime donc surtout les issues
    **lentes** — les busts quand l'airbag est actif — et `P(funded)` calculé sur les seuls chemins
    conclus **monte** pour cette raison seule : 0,293 sans airbag contre 0,386 avec, alors que le
    **nombre de financés est identique** (36 contre 34). L'airbag n'avait produit aucun financement
    de plus ; il avait transformé 33 busts en « on ne sait pas ».

    Avec un budget identique pour tous, `EN_COURS` cesse d'être un accident de date de départ pour
    devenir une **issue mesurée** : « toujours en vie au bout du budget ». Elle se lit alors comme
    les deux autres, et un système qui flotte sans conclure se voit — ce qui est aussi un échec,
    puisqu'un challenge immobilise des frais et du temps.

    ⚠️ **Il ne suffit pas de tronquer : il faut aussi ne partir que de points de départ qui ont le
    budget entier.** C'est le travail de `methode_a`, qui écarte les départs trop tardifs — sinon
    on rétablirait le défaut qu'on vient de corriger.
    """
    if not decisions:
        raise ValueError("Aucune décision : il n'y a pas de challenge à jouer.")
    if not 0 <= depart < len(decisions):
        raise ValueError(
            f"Départ {depart} hors de la séquence (0 à {len(decisions) - 1}). Un départ hors "
            "bornes produirait un chemin vide, compté comme censuré — un faux silencieux."
        )
    dts = [d.dt_entree for d in decisions]
    if dts != sorted(dts):
        raise ValueError(
            "Les décisions ne sont pas triées par date d'entrée. Elles ne sont pas retriées ici "
            "volontairement : l'ordre réel est une donnée du problème (H11, F14)."
        )

    restantes = decisions[depart:]
    debut = restantes[0].dt_entree
    # Recherche binaire (F12) : cette fonction est appelée une fois par chemin, donc des
    # centaines à des milliers de fois par campagne. Un masque booléen relirait toute la série.
    dt_barres = np.asarray(pd.Series(barres["DateTime"]).values)
    i0 = int(np.searchsorted(dt_barres, np.datetime64(debut), side="left"))
    fenetre = barres.iloc[i0:]
    if len(fenetre) == 0:
        raise ValueError(f"Aucune barre au-delà du {debut} : la fenêtre de simulation est vide.")

    if budget_journees is not None:
        if budget_journees < 1:
            raise ValueError(f"budget_journees doit valoir au moins 1, reçu {budget_journees}.")
        jours = pd.to_datetime(pd.Series(fenetre["jour_ftmo"])).dt.date
        distinctes = pd.unique(jours)
        if len(distinctes) < budget_journees:
            raise ValueError(
                f"Départ {depart} : seulement {len(distinctes)} journées FTMO disponibles pour un "
                f"budget de {budget_journees}. Tronquer ici donnerait à ce chemin MOINS de temps "
                "qu'aux autres, ce qui est exactement le défaut que le budget corrige. C'est à "
                "l'appelant d'écarter ce départ."
            )
        fenetre = fenetre.loc[jours.to_numpy() <= distinctes[budget_journees - 1]]
        # ⚠️ **Tronquer la fenêtre oblige à tronquer les décisions avec elle.** Sans ça, le
        # simulateur reçoit des décisions dont la barre d'entrée n'existe plus dans le panel et
        # lève — c'est arrivé au premier jet, et c'est la garde « la barre d'entrée n'existe pas »
        # qui l'a dit. Un chemin ne peut trader que ce qui tombe dans SON budget.
        fin_fenetre = pd.Timestamp(pd.Series(fenetre["DateTime"]).iloc[-1]).to_pydatetime()
        restantes = [d for d in restantes if d.dt_entree <= fin_fenetre]

    return simuler(
        restantes,
        fenetre,
        risque_par_trade_pct=risque_par_trade_pct,
        aimd=aimd,
        regles=regles,
        unites_minimum=unites_minimum,
    )
