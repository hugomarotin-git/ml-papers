"""
La calibration du sizing AIMD sur le twin-null — tâche 6.4, décisions H5 à H8, H14, H16.

**On calibre sur le null, jamais sur le modèle** (H5). Le null n'a aucun signal directionnel :
optimiser seuil, pas, facteur et airbag dessus **ne peut pas fabriquer un faux edge**. Ça ne règle
que la mécanique de survie — et c'est pour ça que **cette calibration ne consomme aucun essai
DSR** (H6) : l'AIMD est commun au null et au modèle, il ne touche pas à la DIFFÉRENCE entre les
deux, qui est l'edge.

Corollaire pratique : **la taille de la grille ne coûte que du temps machine.** Mesuré sur le dev
au 2026-08-19 — 2,6 s par configuration à 255 chemins, donc les 243 configurations en ~11 min.

────────────────────────────────────────────────────────────────────────────────────────
La comparaison est APPARIÉE, et c'est structurel (H8)
────────────────────────────────────────────────────────────────────────────────────────
`calibrer` reçoit **une** séquence de décisions et la joue pour **toutes** les configurations.
Ce n'est pas une convention : il n'existe aucun moyen de lui donner deux séquences différentes.

C'est ce qui rend une PARTIE de la calibration interprétable malgré le bruit : la dispersion de
**l'ÉCART** entre deux configurations jouées sur les mêmes trades est plus faible que celle de
chaque mesure isolée.

⚠️ **6.6 l'a MESURÉ le 2026-08-23, sur 12 réplications du null, et le résultat est sévère :**

- le bruit annoncé jusque-là — « ±1,8 point à 400 chemins » — était l'écart-type **binomial**,
  qui suppose les chemins indépendants. **Ils se chevauchent** (`methode_a` le dit) : un chemin
  couvre 250 journées et les départs sont espacés de 3. Le bruit réel de la borne basse est de
  **8 points en médiane**, soit une sous-estimation d'un facteur **~4**, et les 257 chemins se
  comportent comme **24** ;
- **le classement fin est du hasard** : sur 12 réplications, **10 vainqueurs différents** pour le
  top-1 et **62 configurations sur 162** passées dans le top-10, recouvrement moyen **0,13** ;
- **seul l'axe `risque_base` se départage** (0,5 % contre 2 % : 23,8 points d'écart apparié,
  54 paires sur 54 lisibles). `hausse_additive`, `seuil_journees_perdantes` et `m_max` ont un
  effet moyen **plus petit que sa propre variation d'une configuration à l'autre** : pas d'effet
  propre. `airbag` est **inerte** (+0,23 point, 3 paires lisibles sur 81).

**Ce module produit donc un tableau, jamais un vainqueur** — et `stabilite` (6.6) est le seul
endroit d'où un écart peut sortir avec le droit d'être lu.

────────────────────────────────────────────────────────────────────────────────────────
Deux colonnes, jamais une (H7)
────────────────────────────────────────────────────────────────────────────────────────
`P(funded)` **et** `P(bust)`, toujours ensemble. La raison est mesurée, pas prudentielle : sur un
système **sans edge**, `P(funded)` monte **mécaniquement** avec le risque — 13,6 % à 0,5 % contre
27,5 % à 2 %, le jeu hardi du gambler's ruin à espérance négative. Choisir sur `P(funded)` seul
sélectionnerait donc **le risque le plus élevé**, systématiquement, en croyant optimiser la survie.

────────────────────────────────────────────────────────────────────────────────────────
⚠️ Un point de grille peut être NON MESURABLE, et ce n'est pas un mauvais score
────────────────────────────────────────────────────────────────────────────────────────
Mesuré le 2026-08-19 sur le dev : à `risque_base = 0,5 %`, **aucun chemin ne conclut**
(`p_censure = 1,000`). `P(funded)` vaut alors `0,0` **par convention**, et le juge le dit à voix
haute dans ses `avertissements()` : *« P(funded) n'a pas de dénominateur… ce n'est pas un
résultat »*.

Un classement naïf trierait cette configuration **en dernier**, comme si elle ne finançait jamais,
alors qu'on n'en sait **rien**. Chaque point porte donc ses avertissements, et `classement()`
**refuse** de trier ce qui n'est pas interprétable. La censure se lit **à côté** du chiffre,
jamais après (F13, F19).

*(Rien n'est stocké : tout se recalcule à seed identique — A8, G5.)*
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from aimd import AIMDConfig
from methodes_pfunded import methode_a
from regles_ftmo import ReglesFTMO
from simulateur import Decision
from trajectoires import SEUIL_CENSURE_ALARMANT, Distribution

#: **Les cinq axes de la grille**, arrêtés le 2026-08-14. Aucune valeur de la configuration de référence n'est reprise en
#: dur (A8) : les siennes deviennent le **centre** de chaque axe, et la phase 6 tranche.
#:
#: Ce qui est FIXE hors grille : `facteur_coupe` 0,5 · `m_initial` 1 · `m_min` 0,25 · le reset du
#: compteur après une coupe (H14, tranché le 2026-08-15) · le reset au passage de phase (H4 ③).
AXES: dict[str, tuple] = {
    # Le paramètre DOMINANT, et absent de la configuration de référence. Les trois valeurs sont celles sur
    # lesquelles le comportement de P(funded) a déjà été mesuré le 2026-08-09 (13,6 % à 0,5 %,
    # 27,5 % à 2 % sur un système sans edge).
    # ⚠️ **`0,005` conservé bien qu'il ne conclue PAS à 1 trade/jour** — mesuré le 2026-08-22 :
    # 1 financement et 192 chemins encore en vol au bout d'un an. Le retirer serait prématuré :
    # le verdict dépend de la **cadence**, et c'est précisément ce que la table de 6.5 balaie.
    # Un système deux fois plus rapide conclut deux fois plus vite.
    #
    # ✅ **Balayé le 2026-09-05 (table 6.5)** : la censure médiane à 0,5 % vaut 94,5 % à
    # 0,5 trade/jour, 75,7 % à 1/jour et **encore 45,5 % à 2/jour** — le plafond I7. « Deux fois
    # plus vite » se confirme, mais ne suffit pas : à 0,5 % de risque, aucune cadence du
    # périmètre ne rend la ligne interprétable (seuil d'alarme : 10 %). La valeur reste dans la
    # grille — l'écarter est une décision à prendre, pas un fait acquis — mais toute lecture de
    # sa ligne passe par la censure rapportée à côté (F13).
    "risque_base": (0.005, 0.01, 0.02),
    # Centre 0,25 (configuration de référence). ⚠️ Sa vitesse ne se transporte pas : 4 journées pour remonter de 0,25 à
    # 1,25 valent ~6 trades chez nous contre ~40 chez lui — un facteur 6 sur la vitesse relative.
    "hausse_additive": (0.10, 0.25, 0.50),
    # L'axe le plus discutable, et le seul dont la fréquence de déclenchement soit mesurée :
    # 10,3 % des journées à k=2, 4,0 % à k=3, 1,6 % à k=4.
    "seuil_journees_perdantes": (2, 3, 4),
    # Notre seul ajout à la configuration de référence (H16). `None` = désactivé, explicitement (H9).
    #
    # ⚠️ **`N = 3` retiré de la grille le 2026-08-22, retrait CONFIRMÉ le 2026-09-04 sous 12
    # réplications (décisions de Hugo) — mais la justification a changé en route.** Le chiffre
    # d'origine (« 45 financements contre 50 », soit 1,95 pt de borne basse sur UNE seed) était
    # sous son propre bruit — 6.6 a mesuré ~2,6 pts de seuil pour une seed. La re-mesure du
    # 2026-09-04 (12 réplications, 2 % de risque, 9 contextes hausse x k) donne un effet négatif
    # dans 9 contextes sur 9, et surtout : **`N = 3` est NON-INTERPRÉTABLE à cette cadence** —
    # son plafond de 833 $ par perte ralentit tant les morts qu'un challenge sur trois ne conclut
    # pas dans le budget G10 (censure 30,5 %, 0 point lisible sur 108). Les deux valeurs
    # conservées encadrent la question — « pas d'airbag » et « le plus léger ».
    #
    # ⚠️ **Et l'axe entier est CONSERVÉ malgré un résultat nul sur le null, délibérément.** Le
    # twin-null n'a aucun edge : y survivre plus longtemps ne peut **rien** rapporter, puisque
    # l'espérance est négative. Or l'argument de H16 est « rester en vie assez longtemps pour que
    # l'edge s'exprime ». **Le null ne peut donc pas montrer la valeur de l'airbag** — il peut
    # seulement dire ce qu'il coûte, et à `N = 2` il ne coûte rien (borne basse 0,269 contre
    # 0,259). La question se rouvre en **phase 9**, quand un modèle existera.
    "airbag_n": (None, 2),
    # Centre 1,25 (configuration de référence). Contraint par 5.5 : `risque_base x m_max` doit rester STRICTEMENT sous
    # le plancher journalier — ici au pire 0,02 x 1,50 = 3 %, donc toutes les combinaisons passent.
    "m_max": (1.10, 1.25, 1.50),
}


#: **Le budget de temps donné à chaque chemin**, en journées FTMO — décision de Hugo, 2026-08-21.
#: Le dev compte 1 020 journées FTMO sur quatre ans, soit ~255 par an : 250 vaut donc **un an**.
#:
#: Sans budget uniforme, un chemin parti en 2021 dispose de quatre ans et un chemin parti en 2024
#: de quelques mois — et deux configurations ne sont pas comparées sur le même temps. Avec lui,
#: `EN_COURS` devient une **issue mesurée** : « toujours en vie au bout d'un an ». C'en est une :
#: un challenge qui dure un an sans conclure immobilise des frais et du temps.
BUDGET_UN_AN: int = 250

#: **Plancher de TAILLE, provisoire — décision de Hugo du 2026-08-22, lecture A.**
#:
#: En dessous du lot minimum du broker, l'ordre ne peut pas être passé. Un compte réel ne se met
#: pas en pause pour autant : il passe le **minimum**, donc **il sur-risque** par rapport à ce que
#: le sizing demandait — et il redevient donc mortel.
#:
#: Sans ce plancher, l'airbag crée une **asymptote** : son plafond vaut `marge / N`, chaque perte
#: retire une fraction de ce qui reste, et le plancher FTMO n'est jamais atteint. **Mesuré le
#: 2026-08-21** : zéro bust avec airbag contre 133 sans, à nombre de financés identique — et
#: `P(funded)` affichait 1,000 faute de bust au dénominateur.
#:
#: ⚠️ **La valeur est PROVISOIRE et le restera jusqu'à la tâche 10.1**, qui branchera la vraie
#: granularité (`pas_de_lot`) depuis la spec de contrat du broker. Repère mesuré sur le dev : le
#: stop médian fait 63,7 points, donc 0,01 lot à 0,01 $/point risque ~0,64 $.
UNITES_MINIMUM_PROVISOIRE: float = 0.01


@dataclass(frozen=True)
class PointDeGrille:
    """
    Une configuration et ce qu'elle donne sur le null. **Jamais réduite à un seul chiffre.**

    Les avertissements du juge voyagent **avec** le point : les séparer du chiffre est exactement
    ce que F13 interdit.
    """

    config: AIMDConfig
    p_funded: float
    p_bust: float
    p_censure: float
    n_chemins: int
    n_resolus: int
    n_funded: int
    #: `financés / chemins` — **dénominateur fixe**, donc la seule grandeur que la censure ne
    #: peut pas gonfler. C'est elle qui a démasqué l'asymptote de H16 en restant à 0,197 pendant
    #: que `p_funded` montait de 0,309 à 0,566. ⚠️ **Lue dans `Distribution`, jamais recalculée
    #: ici** : un second endroit où s'écrit `n_funded / n_chemins` est un endroit où la formule
    #: peut diverger en silence (F6, C3).
    p_funded_borne_basse: float
    #: Durées médianes **par issue**, jamais confondues. ⚠️ **À lire à côté de `P(funded)`** : une
    #: configuration dont le bust médian dépasse largement le funded médian voit ses morts
    #: tronquées par le budget, donc son `P(funded)` tiré vers le haut. C'est ce qui faisait
    #: paraître l'airbag meilleur qu'il n'est.
    duree_mediane_funded: float | None
    duree_mediane_bust: float | None
    # ── diagnostic : une config qui protège se distingue-t-elle d'une config inerte ? (6.4) ──
    n_montees: int
    n_coupes: int
    n_declenchements_coupe: int
    n_declenchements_airbag: int
    avertissements: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interpretable(self) -> bool:
        """
        Faux dès qu'aucun chemin n'a conclu, ou que la censure dépasse le seuil de F13.

        ⚠️ **Non interprétable n'est PAS un mauvais score.** À `risque_base = 0,5 %`, aucun chemin
        ne conclut sur le dev : `P(funded)` vaut 0 **par convention**, pas par mesure. Trier ce
        point avec les autres reviendrait à le condamner pour une question qu'on n'a jamais posée.
        """
        return self.n_resolus > 0 and self.p_censure <= SEUIL_CENSURE_ALARMANT

    def resume(self) -> str:
        c = self.config
        drapeau = "" if self.interpretable else "   [NON INTERPRÉTABLE]"
        return (
            f"r={c.risque_base:<6g} h={c.hausse_additive:<5g} k={c.seuil_journees_perdantes} "
            f"airbag={c.airbag_n!s:<4} m_max={c.m_max:<5g} | "
            f"P(funded) {self.p_funded:.3f}  P(bust) {self.p_bust:.3f}  "
            f"censure {self.p_censure:.3f} | durée F/B "
            f"{self.duree_mediane_funded or 0:>4.0f}/{self.duree_mediane_bust or 0:>4.0f} j | "
            f"airbag {self.n_declenchements_airbag:>5}{drapeau}"
        )


def grille_complete(regles: ReglesFTMO | None = None) -> list[AIMDConfig]:
    """
    Le produit cartésien des cinq axes — **162 configurations** depuis le retrait de `N = 3`.

    ⚠️ Le plan annonçait « ~81 », soit `3^4` : c'est un **compte à quatre axes** alors que la
    grille en a **cinq**. À trois valeurs chacun, il y en a `3^5 = 243`. La correction est sans
    conséquence — **H6 dit que cette calibration ne consomme aucun essai DSR**, donc la taille ne
    coûte que du temps machine, et il a été mesuré : ~11 minutes à 255 chemins.

    Chaque configuration est **validée à la construction** (5.5), y compris contre le barème :
    `risque_base x m_max` doit rester strictement sous le plancher journalier.
    """
    regles = regles if regles is not None else ReglesFTMO()
    configs: list[AIMDConfig] = []
    for valeurs in itertools.product(*AXES.values()):
        params = dict(zip(AXES.keys(), valeurs, strict=True))
        config = AIMDConfig(
            facteur_coupe=0.5,
            m_initial=1.0,
            m_min=0.25,
            reset_au_passage_de_phase=True,
            **params,  # pyright: ignore[reportArgumentType]
        )
        config.valider_contre_regles(regles)
        configs.append(config)
    return configs


def _point(config: AIMDConfig, d: Distribution) -> PointDeGrille:
    return PointDeGrille(
        config=config,
        p_funded=d.p_funded,
        p_bust=d.p_bust,
        p_censure=d.p_censure,
        n_chemins=d.n_chemins,
        n_resolus=d.n_resolus,
        n_funded=d.n_funded,
        p_funded_borne_basse=d.p_funded_borne_basse,
        duree_mediane_funded=d.duree_mediane_funded,
        duree_mediane_bust=d.duree_mediane_bust,
        n_montees=d.n_montees,
        n_coupes=d.n_coupes,
        n_declenchements_coupe=d.n_declenchements_coupe,
        n_declenchements_airbag=d.n_declenchements_airbag,
        avertissements=tuple(d.avertissements()),
    )


def calibrer(
    decisions: list[Decision],
    barres: pd.DataFrame,
    configs: Sequence[AIMDConfig],
    pas: int = 1,
    regles: ReglesFTMO | None = None,
    budget_journees: int | None = BUDGET_UN_AN,
    unites_minimum: float = UNITES_MINIMUM_PROVISOIRE,
) -> list[PointDeGrille]:
    """
    Joue **la même** séquence de décisions sous chaque configuration (H8).

    L'appariement est **structurel** : il n'y a qu'un seul argument `decisions`, donc il est
    impossible de comparer deux configurations sur deux séquences différentes. C'est ce qui rend
    lisible un écart plus petit que le bruit de chaque mesure isolée.

    La méthode **A** est la seule utilisable ici : elle préserve l'ordre réel des résultats, dont
    l'AIMD dépend entièrement. La méthode B rebat les trades par blocs et **lève** si on lui passe
    une configuration (F11).

    ⚠️ Ce module **ne classe pas** — voir `classement`, et surtout `stabilite` (6.6) : la
    dispersion des écarts appariés a été mesurée le 2026-08-23, et elle interdit de lire le
    classement fin. Un écart ne se lit qu'en passant par `stabilite.ecart_apparie`.
    """
    if not configs:
        raise ValueError("Aucune configuration à calibrer.")
    return [
        _point(
            c,
            methode_a(
                decisions,
                barres,
                aimd=c,
                pas=pas,
                regles=regles,
                budget_journees=budget_journees,
                unites_minimum=unites_minimum,
            ),
        )
        for c in configs
    ]


def classement(points: Sequence[PointDeGrille]) -> list[PointDeGrille]:
    """
    Trie les points **interprétables** par `P(funded)` décroissant, puis `P(bust)` croissant.

    ⚠️ **Les points non interprétables sont ÉCARTÉS, jamais classés en dernier.** Un point dont
    aucun chemin n'a conclu affiche `P(funded) = 0` par convention : le classer reviendrait à le
    condamner pour une question qu'on n'a jamais posée.

    ⚠️ **Ce tri n'est PAS un verdict, et 6.6 a mesuré à quel point.** Sur 12 réplications du
    null, le premier de ce classement change **10 fois sur 12**, et 62 configurations sur 162
    ont figuré dans son top-10. Deux points voisins ne sont pas « presque à égalité » : ils sont
    **dans un ordre tiré au sort**.

    Il ordonne pour donner une vue, il ne tranche pas. Le seul écart lisible est celui que
    `stabilite.ecart_apparie` déclare `lisible`, sur plusieurs réplications.
    """
    retenus = [p for p in points if p.interpretable]
    return sorted(retenus, key=lambda p: (-p.p_funded, p.p_bust))
