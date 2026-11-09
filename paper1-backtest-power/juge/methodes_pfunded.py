"""
Les deux méthodes de P(funded) — tâche 4.7, décisions F8, F9, F10, F11, F13, H11.

**A** rejoue l'historique réel en décalant le départ. **B** recompose des séquences qui n'ont
pas eu lieu, en tirant des blocs de trades. Elles ne mesurent pas la même chose et **ne se
moyennent jamais** (F8) : le chiffre retenu est **le plus bas des deux**.

────────────────────────────────────────────────────────────────────────────────────────
Ce que chacune capture, et ce qu'elle rate
────────────────────────────────────────────────────────────────────────────────────────
**A — fenêtres glissantes.** Ordre réel préservé, mécanique complète : barres, flottant,
résolution intra-barre, arbitrage F5. C'est la mesure fidèle. Sa faiblesse est le **nombre** :
les chemins se chevauchent massivement (celui qui part au trade 10 et celui du trade 11
partagent presque tout), donc mille départs ne font pas mille observations indépendantes. C'est
exactement le problème d'unicité de la phase 3 (D1, D4), transposé aux trajectoires — et il est
mesuré ici, pas supposé.

**B — bootstrap par blocs.** Répond à « et si les mêmes trades étaient arrivés dans un autre
ordre ? ». Donne autant de chemins qu'on veut, mais **plus grossière par construction** : elle
travaille sur des rendements de trades déjà résolus, donc sans équity flottante intra-trade.
Un compte ne peut y sauter qu'à la clôture d'un trade, jamais en cours de position.

⚠️ **F11 — B est INUTILISABLE avec un sizing adaptatif**, puisqu'elle réordonne les trades
alors que celui-ci dépend entièrement de l'ordre réel des résultats et de la frontière des
journées. **A est seule.** Depuis la tâche 5.6 ce n'est plus une phrase mais une **garde qui
lève** : `methode_b` porte un paramètre `aimd` qui n'existe **que pour ça**.

────────────────────────────────────────────────────────────────────────────────────────
Le biais structurel de B, à connaître avant de lire l'écart A/B (F13)
────────────────────────────────────────────────────────────────────────────────────────
Un chemin de B tire des blocs **jusqu'à conclusion**, dans la limite de `len(trades) x 4`. Il
dispose donc de bien plus de matière qu'un chemin de A, qui s'arrête quand l'historique
s'arrête. B conclut donc plus souvent — et **gonfle mécaniquement son P(funded)**. Ce n'est pas
un bug, c'est une des causes de l'écart A/B, et F10 impose de le lire avant de l'interpréter.

────────────────────────────────────────────────────────────────────────────────────────
La séquence de trades est précalculée une fois (H11)
────────────────────────────────────────────────────────────────────────────────────────
Le rendement d'un trade ne dépend pas de la taille de la position : il s'exprime en **multiples
de risque** (+1 au take-profit, -1 au stop, une fraction au timeout). Le sizing ne fait que
multiplier. `resoudre_trades` produit donc la séquence une fois, et les deux méthodes s'en
servent — c'est aussi ce qui permettra de comparer des configurations de sizing sur exactement
les mêmes trades (H8).

⚠️ Cette fonction **duplique la résolution intra-barre** du simulateur, ce que D3 met en garde
de faire. La contrepartie exigée par D3 est appliquée : la preuve contient une **assertion de
non-régression** qui compare les deux implémentations trade par trade sur les mêmes données.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from aimd import AIMDConfig
from regles_ftmo import Issue, ReglesFTMO, SuiviChallenge
from simulateur import LONG, Decision, ResultatSimulation, horizon_global
from trajectoires import Distribution, agreger, jouer_chemin

#: Multiplicateur du budget de résolution de la méthode B (F13).
BUDGET_B: int = 4


@dataclass(frozen=True)
class TradeRealise:
    """
    Un trade résolu, **indépendant du sizing** (H11).

    `rendement` est en multiples de risque : +1 si la cible est touchée, -1 au stop, une
    fraction si le trade meurt de vieillesse — ou, quand la décision porte sa propre sortie
    (`dt_sortie`, C1), le rendement à l'**Open** de la barre de sortie, sans objectif qui le borne. Le convertir en dollars, c'est le multiplier par
    le risque — et c'est tout ce que le sizing change.

    `nouvelle_journee` dit si ce trade ouvre une journée FTMO. **Ce drapeau est ce que le
    bootstrap doit absolument préserver** : deux pertes le même jour partagent le même plancher
    journalier, deux pertes sur deux jours non. C'est la différence entre un compte mort et un
    compte vivant, à trades identiques.
    """

    rendement: float
    nouvelle_journee: bool
    #: Indice de la barre d'entrée dans le panel, et nombre de barres que le trade a vécues.
    #: Ces deux champs ne servent pas au bootstrap — ils portent l'**occupation du compte**,
    #: dont l'ordonnanceur (4.9) a besoin. Ils valent 0 sur les séquences fabriquées à la main.
    barre_entree: int = 0
    duree_barres: int = 0

    @property
    def barre_sortie(self) -> int:
        """Dernière barre pendant laquelle le compte est occupé par ce trade, incluse."""
        return self.barre_entree + max(self.duree_barres, 1) - 1


def resoudre_trades(decisions: list[Decision], barres: pd.DataFrame) -> list[TradeRealise]:
    """
    Résout chaque décision **sans compte** : ni plancher, ni balance, ni occupation.

    C'est délibéré et c'est H11 : la séquence de trades est une propriété du marché et de la
    règle de décision, pas du capital. Un bust n'interrompt donc rien ici — on veut la séquence
    complète, y compris ce qui se serait passé après.
    """
    if not decisions:
        return []
    dt_barres = np.asarray(pd.Series(barres["DateTime"]).values)
    ouv = np.asarray(barres["Open"], dtype=float)
    haut = np.asarray(barres["High"], dtype=float)
    bas = np.asarray(barres["Low"], dtype=float)
    clo = np.asarray(barres["Close"], dtype=float)
    jours = np.asarray(pd.to_datetime(pd.Series(barres["jour_ftmo"])).dt.date)
    # C1 : l'horizon global n'est lu que si une décision en a besoin — même règle que `simuler`.
    horizon: int | None = horizon_global() if any(d.dt_sortie is None for d in decisions) else None

    sortie: list[TradeRealise] = []
    jour_precedent = None
    for d in decisions:
        i0 = int(np.searchsorted(dt_barres, np.datetime64(d.dt_entree), side="left"))
        if i0 >= len(dt_barres) or dt_barres[i0] != np.datetime64(d.dt_entree):
            raise ValueError(f"La barre d'entrée {d.dt_entree} n'existe pas dans le panel.")
        # C1 — les barres DÉTENUES : jusqu'à l'horizon, ou jusqu'à la barre de sortie EXCLUE
        # (on la quitte à son Open, ce qui s'y passe ensuite ne nous concerne plus — Cb2).
        if d.dt_sortie is not None:
            j = int(np.searchsorted(dt_barres, np.datetime64(d.dt_sortie), side="left"))
            if j >= len(dt_barres) or dt_barres[j] != np.datetime64(d.dt_sortie) or j <= i0:
                raise ValueError(
                    f"La barre de sortie {d.dt_sortie} n'existe pas dans le panel, ou n'est pas "
                    f"après l'entrée {d.dt_entree}."
                )
            n_detenues = j - i0
        else:
            assert horizon is not None  # garanti par la lecture ci-dessus
            n_detenues = horizon

        rendement = 0.0
        duree = 0
        resolu = False
        cible = d.cible
        for k in range(n_detenues):
            i = i0 + k
            if i >= len(dt_barres):
                break
            duree = k + 1
            pire = bas[i] if d.direction == LONG else haut[i]
            meilleur = haut[i] if d.direction == LONG else bas[i]
            # Côté défavorable d'abord (F4) : sur une barre qui touche les deux, le stop gagne.
            if (pire <= d.stop) if d.direction == LONG else (pire >= d.stop):
                rendement = -1.0
                resolu = True
                break
            # C1 : sans objectif, rien ne fait gagner avant la sortie.
            if cible is not None and (
                (meilleur >= cible) if d.direction == LONG else (meilleur <= cible)
            ):
                rendement = 1.0
                resolu = True
                break
            rendement = d.direction * (clo[i] - d.prix_entree) / d.distance_stop
        # C1 : sortie propre non résolue par le stop -> le rendement est celui de l'OPEN de la
        # barre de sortie, pas du Close de la dernière barre détenue (Cb2).
        if d.dt_sortie is not None and not resolu and duree == n_detenues:
            i_sortie = i0 + n_detenues
            rendement = d.direction * (ouv[i_sortie] - d.prix_entree) / d.distance_stop

        sortie.append(
            TradeRealise(
                rendement=rendement,
                nouvelle_journee=jours[i0] != jour_precedent,
                barre_entree=i0,
                duree_barres=duree,
            )
        )
        jour_precedent = jours[i0]
    return sortie


def plus_longue_serie_perdante(trades: list[TradeRealise]) -> int:
    """
    La plus longue suite de trades perdants **réellement observée**.

    C'est le plancher de la taille de bloc (F9) : un bloc plus court coupe les mauvaises passes
    en deux, et le bootstrap ne peut alors plus jamais reproduire la pire série vécue — il
    **sous-estime le risque de bust**, silencieusement.
    """
    record = courante = 0
    for t in trades:
        courante = courante + 1 if t.rendement < 0 else 0
        record = max(record, courante)
    return record


# ── Méthode A ────────────────────────────────────────────────────────────────────────────
def methode_a(
    decisions: list[Decision],
    barres: pd.DataFrame,
    risque_par_trade_pct: float | None = None,
    aimd: AIMDConfig | None = None,
    pas: int = 1,
    regles: ReglesFTMO | None = None,
    budget_journees: int | None = None,
    unites_minimum: float = 0.0,
) -> Distribution:
    """
    Fenêtres de départ glissantes sur l'historique **réel** (F8).

    Un chemin par départ, l'ordre des trades est celui qui s'est produit. `pas` espace les
    départs : `pas=1` prend tous les départs possibles, `pas=10` un sur dix. Ça ne change pas
    l'indépendance — les chemins se chevauchent de toute façon — seulement le coût de calcul.

    **C'est la SEULE méthode utilisable avec un sizing adaptatif** (F11) : elle préserve
    l'ordre réel des résultats, dont l'AIMD dépend entièrement. La méthode B, qui rebat les
    trades par blocs, lève si on lui passe une config — voir la garde de `methode_b`.

    ────────────────────────────────────────────────────────────────────────────────
    `budget_journees` — le même temps pour tous (décision de Hugo, 2026-08-21)
    ────────────────────────────────────────────────────────────────────────────────
    Quand un budget est donné, **les départs qui ne disposent pas du budget entier sont écartés**.
    Sans ce filtre, les derniers départs auraient moins de temps que les premiers et l'on
    rétablirait le défaut que le budget corrige — un `EN_COURS` qui ne dirait rien de la
    configuration, seulement de la date de départ.

    Les départs écartés sont les **mêmes pour toutes les configurations** — ils ne dépendent que
    des données —, donc `n_chemins` reste identique d'une configuration à l'autre et la
    comparaison appariée de **H8** est préservée. `departs_utilisables` permet de le dire à voix
    haute plutôt que de laisser deviner l'écart.
    """
    return agreger(
        resultats_methode_a(
            decisions,
            barres,
            risque_par_trade_pct,
            aimd,
            pas,
            regles,
            budget_journees,
            unites_minimum,
        )
    )


def resultats_methode_a(
    decisions: list[Decision],
    barres: pd.DataFrame,
    risque_par_trade_pct: float | None = None,
    aimd: AIMDConfig | None = None,
    pas: int = 1,
    regles: ReglesFTMO | None = None,
    budget_journees: int | None = None,
    unites_minimum: float = 0.0,
) -> list[ResultatSimulation]:
    """
    Les chemins joués de la méthode A, **avant** agrégation — généralisation du 2026-09-26 (9.2).

    `methode_a` en est l'agrégat. Exposer les résultats bruts permet de **fusionner plusieurs
    séquences de décisions** (les chemins CPCV) en une seule `Distribution` exacte — médianes
    comprises — au lieu de moyenner des distributions déjà agrégées, ce qui est faux dès que les
    séquences n'ont pas le même nombre de départs.
    """
    if pas < 1:
        raise ValueError(f"pas doit valoir au moins 1, reçu {pas}")
    if not decisions:
        raise ValueError("Aucune décision : la méthode A n'a aucun départ à jouer.")

    departs = range(0, len(decisions), pas)
    if budget_journees is not None:
        utilisables = departs_utilisables(decisions, barres, budget_journees)
        departs = [d for d in departs if d < utilisables]
        if not departs:
            raise ValueError(
                f"Aucun départ ne dispose d'un budget de {budget_journees} journées FTMO. La "
                "série est trop courte pour ce budget, ou le budget trop long pour la série."
            )
    resultats: list[ResultatSimulation] = [
        jouer_chemin(
            decisions,
            barres,
            depart,
            risque_par_trade_pct,
            aimd,
            regles,
            budget_journees,
            unites_minimum,
        )
        for depart in departs
    ]
    return resultats


def departs_utilisables(
    decisions: list[Decision], barres: pd.DataFrame, budget_journees: int
) -> int:
    """
    Combien de départs disposent du budget **entier**, en partant du début.

    Les départs sont chronologiques, donc la propriété est **monotone** : si le départ `k` a le
    budget, tous ceux d'avant l'ont aussi. Le résultat est donc un **seuil**, pas un ensemble —
    ce qui rend le filtre trivial à appliquer et impossible à appliquer de travers.

    ⚠️ Ne dépend **que des données**, jamais de la configuration de sizing : c'est ce qui garantit
    que toutes les configurations jouent exactement les mêmes départs (H8).
    """
    if budget_journees < 1:
        raise ValueError(f"budget_journees doit valoir au moins 1, reçu {budget_journees}.")
    jours = pd.to_datetime(pd.Series(barres["jour_ftmo"])).dt.date.to_numpy()
    dt_barres = np.asarray(pd.Series(barres["DateTime"]).values)
    distinctes = pd.unique(jours)
    if len(distinctes) < budget_journees:
        return 0
    # La dernière journée qui peut encore ouvrir un budget complet.
    derniere = distinctes[len(distinctes) - budget_journees]
    for k, d in enumerate(decisions):
        i = int(np.searchsorted(dt_barres, np.datetime64(d.dt_entree), side="left"))
        if i >= len(jours) or jours[i] > derniere:
            return k
    return len(decisions)


def chemins_effectifs(distribution: Distribution, n_journees_totales: int) -> float:
    """
    Combien de chemins **indépendants** valent les chemins de la méthode A.

    Les fenêtres glissantes se recouvrent : deux départs voisins partagent presque tous leurs
    trades. Compter les départs comme des observations indépendantes rétrécit artificiellement
    les intervalles de confiance et fabrique de la certitude — c'est exactement l'erreur que D4
    interdit sur les labels, transposée aux trajectoires.

    Estimation, du même esprit que l'unicité de la phase 3 : le nombre de challenges
    **non chevauchants** qui tiennent dans l'historique, soit la durée totale divisée par la
    durée médiane d'un challenge résolu. Jamais plus que le nombre de chemins joués.

    ⚠️ **Estimateur ANALYTIQUE, pas la mesure de référence — ne pas s'en servir pour trancher**
    (noté à l'audit du 2026-09-06). Depuis la tâche 6.6, l'effectif qui a le droit d'être lu est
    **mesuré** en rejouant le null sous plusieurs seeds (`stabilite.Dispersion.n_chemins_effectif`),
    pas déduit d'une durée médiane. Les deux concordent grossièrement (~22 ici contre ~24 mesuré),
    mais un seul est la référence : avoir deux chiffres d'effectif qui peuvent diverger est le
    piège que D3 met en garde de créer. Cette fonction reste pour l'inspection rapide et sa preuve
    de non-régression ; un verdict passe par `stabilite`.
    """
    duree = distribution.duree_mediane_funded or distribution.duree_mediane_bust
    if not duree or duree <= 0:
        return 0.0
    return min(float(distribution.n_chemins), n_journees_totales / duree)


# ── Méthode B ────────────────────────────────────────────────────────────────────────────
def _jouer_sequence(
    trades: list[TradeRealise], risque_par_trade_pct: float, regles: ReglesFTMO
) -> ResultatSimulation:
    """
    Rejoue une séquence de trades **déjà résolus** sur un compte.

    Plus grossier que le simulateur, et de façon assumée : sans barres, il n'y a pas d'équity
    flottante, donc le compte ne peut sauter qu'à la clôture d'un trade. C'est le prix du
    bootstrap, et c'est une des raisons pour lesquelles A reste la mesure de référence.
    """
    suivi = SuiviChallenge(regles)
    balance = regles.capital_initial
    n_journees = n_trades = 0
    ouverte = False

    for t in trades:
        if t.nouvelle_journee or not ouverte:
            # `jour=None` : la méthode B rejoue des trades DÉJÀ RÉSOLUS, sans barres ni
            # dates — elle regroupe sur un drapeau. Lui faire fabriquer des dates serait
            # inventer de la donnée pour satisfaire une garde. La désactivation s'écrit (4.12).
            suivi.nouvelle_journee(None, balance)
            n_journees += 1
            ouverte = True
        suivi.marquer_journee_tradee()
        balance += t.rendement * regles.capital_initial * risque_par_trade_pct
        n_trades += 1

        phase_avant = suivi.phase
        if suivi.observer(equity=balance, balance=balance) is not Issue.EN_COURS:
            break
        if phase_avant != suivi.phase:  # compte neuf en phase 2 (F17)
            balance = regles.capital_de_depart(suivi.phase)
            ouverte = False

    return ResultatSimulation(
        issue=suivi.issue,
        balance_finale=balance,
        n_trades=n_trades,
        n_barres=0,
        n_decisions_ignorees=0,
        motif_bust=suivi.motif_bust.value if suivi.motif_bust else None,
        jours_tradees=suivi.jours_tradees,
        n_journees=n_journees,
    )


def methode_b(
    trades: list[TradeRealise],
    risque_par_trade_pct: float,
    taille_bloc: int,
    n_chemins: int,
    seed: int,
    aimd: AIMDConfig | None = None,
    regles: ReglesFTMO | None = None,
) -> Distribution:
    """
    Bootstrap par blocs (F8), avec budget borné (F13).

    ⚠️ **F11 — le paramètre `aimd` n'existe ici QUE POUR LEVER** (critère de recette ③ de 5.6).
    B rebat les trades par blocs, alors qu'un sizing adaptatif dépend entièrement de **l'ordre
    réel** des résultats et de la frontière des journées : lui passer une config produirait un
    `P(funded)` parfaitement lisible et dénué de sens. **Une convention écrite dans une docstring
    finit par être contournée ; une garde qui lève, non** — d'où un paramètre plutôt qu'une
    phrase. Pour un sizing adaptatif, c'est la méthode **A**, et elle seule.

    `taille_bloc` est **obligatoire et validée contre F9** : elle doit couvrir au moins la plus
    longue série perdante observée. `seed` est obligatoire aussi — la comparaison de
    configurations de sizing doit être appariée (H8), et rien n'est stocké (A8) : tout se
    recalcule à l'identique.

    Chaque chemin tire des blocs jusqu'à conclusion, dans la limite de `len(trades) x 4` trades.
    Un chemin qui épuise son budget est **censuré, jamais relancé**.
    """
    if aimd is not None:
        raise NotImplementedError(
            "F11 : la méthode B ne peut PAS porter un sizing adaptatif. Elle rebat les trades "
            "par blocs, alors que l'AIMD dépend de l'ordre réel des résultats et de la frontière "
            "des journées FTMO — le P(funded) produit serait lisible et sans aucun sens. Utiliser "
            "la méthode A, qui préserve l'ordre réel."
        )
    regles = regles if regles is not None else ReglesFTMO()
    if not trades:
        raise ValueError("Aucun trade : le bootstrap n'a rien à tirer.")
    if n_chemins < 1:
        raise ValueError(f"n_chemins doit valoir au moins 1, reçu {n_chemins}")
    if taille_bloc < 1:
        raise ValueError(f"taille_bloc doit valoir au moins 1, reçu {taille_bloc}")

    minimum = plus_longue_serie_perdante(trades)
    if taille_bloc < minimum:
        raise ValueError(
            f"taille_bloc={taille_bloc} est plus courte que la plus longue série perdante "
            f"observée ({minimum} trades). F9 l'interdit : un bloc trop court coupe les mauvaises "
            "passes en deux, donc le bootstrap ne peut plus reproduire la pire série vécue et "
            "SOUS-ESTIME le risque de bust — sans que rien ne le signale."
        )

    if taille_bloc > len(trades):
        raise ValueError(
            f"taille_bloc={taille_bloc} dépasse le nombre de trades disponibles "
            f"({len(trades)}) : chaque tirage rendrait la séquence entière, et le bootstrap "
            "ne tirerait plus rien au hasard."
        )

    alea = random.Random(seed)
    budget = len(trades) * BUDGET_B
    dernier_depart = len(trades) - taille_bloc

    resultats: list[ResultatSimulation] = []
    for _ in range(n_chemins):
        # On tire d'emblée le budget complet, puis on joue UNE fois. C'est équivalent à tirer
        # bloc par bloc jusqu'à conclusion — `_jouer_sequence` s'arrête d'elle-même sur une
        # issue terminale — mais en temps linéaire au lieu de quadratique.
        sequence: list[TradeRealise] = []
        while len(sequence) < budget:
            i = alea.randint(0, dernier_depart)
            sequence.extend(trades[i : i + taille_bloc])
        resultats.append(_jouer_sequence(sequence[:budget], risque_par_trade_pct, regles))
    return agreger(resultats)


# ── La comparaison ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ComparaisonAB:
    """Les deux méthodes côte à côte. **Jamais moyennées** (F8)."""

    a: Distribution
    b: Distribution

    @property
    def conservative(self) -> float:
        """**Le chiffre retenu** : le plus bas des deux (F8). Une moyenne n'aurait aucun sens."""
        return min(self.a.p_funded, self.b.p_funded)

    @property
    def ecart(self) -> float:
        return abs(self.a.p_funded - self.b.p_funded)

    def lecture(self) -> list[str]:
        """
        F10 — **un grand écart A/B se lit avant de s'interpréter.**

        Trois causes possibles, et elles ne se confondent pas : une vraie fragilité au
        clustering, une série trop courte pour que A conclue, ou le biais structurel de budget
        de B. L'ordre de lecture est imposé : la non-conclusion de A d'abord.
        """
        messages: list[str] = []
        if self.a.p_censure > 0.10:
            messages.append(
                f"LIRE D'ABORD : {100 * self.a.p_censure:.0f} % des chemins de A n'ont pas "
                f"conclu ({self.a.n_resolus} conclus sur {self.a.n_chemins}). Un écart A/B "
                "s'explique alors surtout par le manque de recul de A, pas par une fragilité."
            )
        if self.ecart > 0.10:
            messages.append(
                f"ÉCART A/B de {100 * self.ecart:.0f} points. Rappel F13 : B tire des blocs "
                "jusqu'à conclusion dans la limite de son budget, elle dispose donc de plus de "
                "matière que A et conclut plus souvent — son P(funded) est mécaniquement gonflé. "
                "Ce n'est pas un bug, c'est une cause connue de l'écart."
            )
        if self.b.p_funded < self.a.p_funded:
            messages.append(
                "B est SOUS A, ce qui va contre son biais de budget : le clustering des pertes "
                "coûte donc réellement quelque chose. À prendre au sérieux."
            )
        return messages

    def resume(self) -> str:
        lignes = [
            "  ── Méthode A (fenêtres glissantes, ordre réel) ──",
            self.a.resume(),
            "",
            "  ── Méthode B (bootstrap par blocs) ──",
            self.b.resume(),
            "",
            f"  P(funded) RETENU (le plus bas des deux) : {100 * self.conservative:.1f} %",
        ]
        return "\n".join(lignes + [f"  ⚠️ {m}" for m in self.lecture()])
