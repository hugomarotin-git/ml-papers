"""
Le sizing AIMD v2 — tâches 5.3 (le noyau), 5.4 (l'airbag) et 5.5 (la validation).
Décisions H1, H2, H4, H9, H10, H14, H15, H16.

**Ce module ne connaît ni barres, ni trades, ni marché.** Il reçoit des bilans de journée et
produit un risque. C'est exactement la séparation 4.1 / 4.2 transposée au sizing : ce qui
ne dépend pas des données se prouve sans données.

Il ne fait **pas** partie du juge : il ne dit pas si c'est gagné, il dit combien on risque.

────────────────────────────────────────────────────────────────────────────────────────
Le déclencheur : le RÉSULTAT NET DE LA JOURNÉE FTMO (H2, réécrite le 2026-08-15)
────────────────────────────────────────────────────────────────────────────────────────
La v1 déclenchait sur la **distance à la barrière**. Elle est morte d'une mesure, pas d'un avis :
depuis que F2 est aligné sur le vrai barème FTMO (2 500 $ de perte autorisés par jour, montant
**fixe**), cette distance vaut 5 % du capital **par définition** à chaque ouverture de journée.
Mesurée sur le singe : plate sur **92,4 % des premiers trades du jour**, soit **~57 % de tous
les trades**. Un déclencheur muet sur la majorité des trades ne déclenche rien.

Le résultat de la journée, lui, a toujours quelque chose à dire. Trois précisions font partie de
la décision (H2 ①②③), parce que **sans elles le code peut être faux en silence** :

1. le bilan est la variation de **BALANCE**, donc du **réalisé seul**. Une position encore
   ouverte à la frontière ne compte pas : son résultat tombera dans la journée où elle se ferme.
   Prendre l'équity ferait entrer du flottant non réalisé dans une décision de sizing ;
2. la frontière est celle de la **journée FTMO** (minuit serveur, B4), jamais la journée
   calendaire. C'est le piège n°1 du projet, et il ne plante pas ;
3. la mise à jour se fait **à la clôture**. `m` n'influence donc que les positions ouvertes
   **après** — le look-ahead intra-journée devient impossible **par construction**, pas par
   convention.

────────────────────────────────────────────────────────────────────────────────────────
Quatre issues, pas trois (H15) — et « nulle » n'est PAS « sans trade »
────────────────────────────────────────────────────────────────────────────────────────
Les deux valent zéro et n'ont pas le même effet :

| bilan | fait monter `m` ? | casse la série de pertes ? |
|---|---|---|
| gagnante | oui (additif) | oui |
| perdante | non — elle allonge la série | non |
| **nulle** (des trades ont conclu, net à zéro) | non | **oui** |
| **sans trade** (rien n'a conclu) | non | **non** — totalement transparente |

Pourquoi la journée sans trade est transparente : monter `m` sans avoir tradé, c'est **gagner de
la confiance sans preuve** ; et effacer une série de pertes sur une journée vide, c'est remettre
le compteur de danger à zéro **sans que rien n'ait montré que le mauvais régime est fini**. Avec
le plancher de confiance (I13), ce n'est pas un cas limite : ce sera peut-être le cas majoritaire.
**Divergence assumée avec la configuration de référence**, où une journée vide fait monter `m` et casse la série.

⚠️ **La frontière entre « nulle » et « sans trade » est le nombre de trades CLÔTURÉS dans la
journée, jamais le nombre de trades ouverts.** C'est une précision que H15 ne tranche pas
explicitement, et elle est forcée par H2 ① : le résultat d'une position tombe dans la journée où
elle **se ferme**. Une journée dont l'unique trade est encore ouvert à la frontière n'a donc
produit **aucun résultat** — elle ne prouve rien, elle est transparente. Compter les *ouvertures*
ferait casser une série de pertes par un trade qui n'a rien conclu.

*(Sur les vraies données ce cas n'existe pas : aucun trade issu du label ne peut enjamber la
frontière de 17:00 NY, la contiguïté C9 l'interdit. Mais la règle doit être juste aussi sur les
cas qu'on fabrique à la main — et elle le redeviendra le jour où le périmètre changera.)*

⚠️ **Conséquence connue, à revoir en 10.1** : le coût d'un trade est prélevé à son **ouverture**
(`simulateur.cout_par_trade`). Une journée qui ouvrirait une position sans la clôturer verrait
donc sa balance bouger du montant du coût tout en étant classée « sans trade ». C'est le bon
arbitrage — **un coût n'est pas un résultat** — mais ça se saura quand les coûts seront branchés.
Aujourd'hui `cout_par_trade` vaut zéro, et le cas est de toute façon impossible (C9).

────────────────────────────────────────────────────────────────────────────────────────
Le zéro exact d'une journée nulle se teste sur des flottants (pièges n°6 et n°7)
────────────────────────────────────────────────────────────────────────────────────────
H15 signale le piège sans fixer la règle, parce que c'est un choix de code : une journée dont les
trades s'annulent **exactement** ne donne presque jamais `0.0` en flottant. `balance_cloture -
balance_ouverture` ressort à ±1e-11, et un `< 0` naïf classerait cette journée **perdante** — donc
allongerait une série de pertes qui n'existe pas, donc couperait `m` sur du bruit numérique.

La tolérance retenue n'est **pas** une epsilon flottante, elle est **tirée du domaine** :
`TOLERANCE_BILAN_NUL` vaut un **demi-cent**. Un compte est libellé en dollars et cents ; une
variation qui n'arrive pas à déplacer un cent n'a pas déplacé la balance. Le choix se justifie
sans parler de flottants — et il couvre le piège avec neuf ordres de grandeur de marge (l'erreur
relative d'un `float` sur une balance de 50 000 est de l'ordre de 5e-12, cf. check 3).

Une tolérance n'est **jamais** un paramètre de configuration : ce serait un réglage capable de
changer la classification d'une journée, donc le sizing, donc `P(funded)`.

────────────────────────────────────────────────────────────────────────────────────────
L'hystérésis (H14) — une journée perdante isolée ne coupe JAMAIS
────────────────────────────────────────────────────────────────────────────────────────
Mesuré sur le singe (dev) : la journée FTMO contient **1,62 trade** et **42,8 %** des journées
sont perdantes. Une journée perdante isolée est donc **l'événement le plus banal du système** —
couper dessus, c'est réagir au bruit. La coupe demande `k` journées perdantes **consécutives**,
`k ≥ 2`, et `k` part en grille (2 / 3 / 4) en phase 6 (H5), jamais écrit en dur (A8).

**Après une coupe, le compteur repart à 0** — lecture (a), tranchée par Hugo le 2026-08-15. La
variante « le compteur continue » ferait dégénérer un seuil à 2 en « coupe à chaque journée
perdante dès la 2ᵉ », soit exactement la réaction au bruit qu'on vient de supprimer. Et la mesure
ne lui donne aucune prémisse : sur le singe, la distribution des séries perdantes est
**indiscernable de tirages indépendants** (60,1 / 24,3 / 9,5 / 6,1 % mesuré, 57,2 / 24,5 / 10,5 /
7,8 % attendu sous indépendance) — une longue série ne dit **rien** du lendemain.

⚠️ **Conséquence à connaître avant la phase 6** : avec ce reset, atteindre le plancher de `m`
depuis `m₀` demande **2k journées perdantes consécutives** (4 à k=2, 6 à k=3, **8 à k=4**), or la
plus longue série observée sur tout le dev fait **8 journées**. À k=4, la seconde coupe est donc
un événement unique dans l'historique : **cette valeur d'axe est quasi inerte**. À constater au
diagnostic de 5.6, pas à supposer. La propriété elle-même est gravée en preuve (check 14).

────────────────────────────────────────────────────────────────────────────────────────
Le départ et les resets (H4, amendée le 2026-08-15)
────────────────────────────────────────────────────────────────────────────────────────
① Départ à **`m₀`** sans aucun ajustement préalable : le tout premier trade d'un compte sort à
`risque_base x m₀`. Le slow start de la v1 a disparu — il imposait à chaque chemin un handicap de
départ **qui n'apprenait rien**.

② **Aucune mémoire ne franchit la frontière d'un challenge**, et la garantie est **structurelle**,
pas déclarative : il n'existe aucun état de module ici, seulement des attributs d'instance. C'est
le simulateur qui construit un `SuiviAIMD` par simulation (fait en 5.6), donc deux chemins
ne peuvent pas partager d'objet. Une convention écrite n'aurait pas suffi : le jour où deux
chemins partagent un état, `P(funded)` cesse d'être une probabilité pour devenir un artefact
d'ordonnancement — et le chiffre reste parfaitement plausible.

③ Au **passage en phase 2**, le compte est neuf (F17) : `passage_de_phase` remet l'état si
`reset_au_passage_de_phase` est vrai. ⚠️ **Et dans les deux cas la journée en cours est
ABANDONNÉE, jamais close** — voir l'avertissement porté par la méthode : la balance vient d'être
remise au capital de départ, donc la classer produirait une journée gagnante ou perdante
**fabriquée par un saut comptable**.

────────────────────────────────────────────────────────────────────────────────────────
Ce qui n'est PAS dans cette tâche
────────────────────────────────────────────────────────────────────────────────────────
- ~~L'airbag de survie (H16) → tâche 5.4~~ — ✅ **FAIT le 2026-08-16** : `risque_effectif` est
  désormais le **seul** accès public au risque, `_risque_nominal` est privée. Lire le nominal
  contournerait le filet en silence.
- ~~La validation de `AIMDConfig` → tâche 5.5~~ — ✅ **FAITE le 2026-08-17** : chaque valeur
  absurde est refusée **pour sa propre raison**, et le produit `risque_base x m_max` doit rester
  sous le plancher journalier, borne **lue** dans `ReglesFTMO`. Avec l'autre moitié de H9, déjà
  là depuis 5.3 : **aucun défaut sur aucun champ**.
- **Le branchement dans le simulateur** → tâche **5.6**, avec ses cinq critères de recette.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from regles_ftmo import ReglesFTMO

#: **Un demi-cent.** En deçà, la variation de balance d'une journée ne déplace pas un cent : elle
#: est nulle au sens du compte. Seuil tiré du domaine (un compte est libellé en dollars et cents),
#: et non d'une epsilon flottante — mais il couvre le piège des pièges n°6/n°7 avec neuf ordres de
#: grandeur de marge, ce que le check 3 mesure au lieu de l'affirmer.
#:
#: ⚠️ **Ce n'est délibérément PAS un paramètre de configuration.** Un réglage ici changerait la
#: classification des journées, donc le sizing, donc `P(funded)` — et le seul symptôme serait un
#: chiffre légèrement faux.
TOLERANCE_BILAN_NUL: float = 0.005


class BilanJournee(Enum):
    """
    Le résultat net d'une journée FTMO — **quatre issues, pas trois** (H15).

    `NULLE` et `SANS_TRADE` valent toutes deux zéro et n'ont **pas** le même effet : la première
    casse une série de pertes (des trades ont conclu, le régime s'est exprimé), la seconde est
    totalement transparente (rien n'a conclu, donc rien n'a été prouvé). Les confondre, c'est
    remettre le compteur de danger à zéro sur une journée vide.
    """

    GAGNANTE = "gagnante"
    PERDANTE = "perdante"
    NULLE = "nulle"
    SANS_TRADE = "sans_trade"


def classer_journee(variation_balance: float, n_trades_clotures: int) -> BilanJournee:
    """
    Classe une journée FTMO à partir de son **réalisé** — fonction pure, prouvable seule.

    `variation_balance` est `balance de clôture - balance d'ouverture` de la journée FTMO (H2 ①) :
    du réalisé, jamais de l'équity. `n_trades_clotures` est le nombre de trades qui se sont
    **fermés** dans la journée — pas ceux qui s'y sont ouverts (voir l'en-tête du module).

    L'ordre des tests est la règle : **l'absence de clôture prime sur la variation**. Une journée
    où rien n'a conclu est transparente, même si sa balance a bougé (cas du coût prélevé à
    l'ouverture, cf. l'en-tête) — un coût n'est pas un résultat.

    ⚠️ **Le refus du non-fini n'est pas une précaution de principe.** Un `NaN` échoue les DEUX
    comparaisons d'encadrement (`>= tol` et `<= -tol`) et tomberait donc dans la branche par
    défaut : la journée serait classée **nulle**, elle casserait la série de pertes, et rien ne
    signalerait jamais qu'une balance était corrompue en amont. C'est la signature exacte du faux
    silencieux — un chiffre plausible produit par une donnée qui ne l'est pas. Le trou a été trouvé
    en relisant la fonction après coup, pas par les preuves : elles ne l'exerçaient pas.
    """
    if not math.isfinite(variation_balance):
        raise ValueError(
            f"variation_balance doit être un nombre fini, reçu {variation_balance!r}. Un NaN "
            "échoue les deux comparaisons et serait classé « journée nulle » en silence, ce qui "
            "casserait la série de pertes sur une balance corrompue."
        )
    if n_trades_clotures < 0:
        raise ValueError(
            f"n_trades_clotures ne peut pas être négatif, reçu {n_trades_clotures}. Un compteur "
            "négatif signale un site d'appel qui décrémente là où il devrait incrémenter."
        )
    if n_trades_clotures == 0:
        return BilanJournee.SANS_TRADE
    if variation_balance >= TOLERANCE_BILAN_NUL:
        return BilanJournee.GAGNANTE
    if variation_balance <= -TOLERANCE_BILAN_NUL:
        return BilanJournee.PERDANTE
    return BilanJournee.NULLE


@dataclass(frozen=True)
class AIMDConfig:
    """
    Les paramètres du sizing adaptatif. **Aucun défaut, pour aucun champ** (H9).

    Ce n'est pas une coquetterie : un risque hérité d'un périmètre abandonné survit sans que rien
    ne le signale, et `risque_base` est le **paramètre dominant** du résultat. Tout doit être dit.

    Gelée (`frozen`) pour la même raison que `ReglesFTMO` : une simulation ne doit pas pouvoir
    modifier ses propres réglages en cours de route — c'est le genre de mutation qui produit un
    résultat flatteur et introuvable. Le corollaire est utile : une config peut être partagée sans
    risque entre chemins, tandis que l'**état** (`SuiviAIMD`) est créé par simulation (H4 ②).

    ⚠️ **La validation arrive en tâche 5.5**, où chaque valeur absurde sera refusée **pour sa
    propre raison**, plus la garde de plafond de risque (`risque_base x m_max` sous le plancher
    journalier, borne **lue** depuis `ReglesFTMO`). D'ici là, cette classe ne vérifie rien : une
    config incohérente (`m_initial` hors de `[m_min, m_max]`, par exemple) sera acceptée. Ne pas
    la valider à moitié maintenant est délibéré — une validation partielle donne la confiance
    d'une validation complète.

    Ce qui part en grille en phase 6 (H5, calibré sur le twin-null) : `risque_base`,
    `hausse_additive`, `seuil_journees_perdantes`, `m_max`, et le `N` de l'airbag (5.4).
    Fixés hors grille : `facteur_coupe` = 0,5 · `m_initial` = 1 · `m_min` = 0,25.
    """

    #: Fraction du capital INITIAL risquée sur un trade **avant** le multiplicateur. Dominant.
    risque_base: float
    #: Le pas de la montée **additive** après une journée gagnante (H1).
    hausse_additive: float
    #: `k` — nombre de journées perdantes **consécutives** avant de couper (H14). `k >= 2`.
    seuil_journees_perdantes: int
    #: Le facteur de la coupe **multiplicative** (H1). Repris de la configuration de référence : 0,5.
    facteur_coupe: float
    #: `m₀` — le multiplicateur au premier trade d'un compte (H4 ①). Plus de slow start.
    m_initial: float
    #: Plancher du multiplicateur : borne l'**adaptation**. Ne remplace pas l'airbag (H16), qui
    #: borne l'**exposition absolue**.
    m_min: float
    #: Plafond du multiplicateur. Contraint en 5.5 par `risque_base x m_max < plancher journalier`.
    m_max: float
    #: H4 ③ — au passage en phase 2, le compte est neuf (F17) : faut-il remettre `m` et la série
    #: de journées perdantes à leur valeur de départ ? Lecture à **mesurer** en phase 6, pas à
    #: choisir au jugé — d'où un champ explicite, comme F15/F16/F17 côté juge.
    reset_au_passage_de_phase: bool
    #: **`N` de l'airbag** (H16) — « combien de trades perdants au MINIMUM pour mourir ».
    #: `None` **désactive** l'airbag, et cette désactivation doit être **écrite** : le champ n'a
    #: aucun défaut (H9), donc `airbag_n=None` est un choix posé, jamais un oubli. Un filet absent
    #: par distraction ne se voit nulle part. Part en grille phase 6 : désactivé / 2 / 3.
    airbag_n: int | None

    def __post_init__(self) -> None:
        """
        Chaque valeur absurde refusée **pour sa propre raison** (H9, tâche 5.5) — jamais par un
        contrôle générique. Une erreur de config se lit dans son message, pas dans une trace.

        ────────────────────────────────────────────────────────────────────────────────
        ⚠️ La FORME du test d'intervalle décide s'il attrape les `NaN` — piège n°8
        ────────────────────────────────────────────────────────────────────────────────
        Ces deux lignes se lisent pareil et **ne font pas la même chose** ::

            if not 0 < x < 1:        # NaN REFUSÉ  : `0 < nan` est False, donc `not` déclenche
            if x <= 0 or x >= 1:     # NaN ACCEPTÉ : les DEUX comparaisons sont False

        La seconde laisserait un `NaN` entrer dans `risque_base`, donc dans le nombre d'unités,
        donc dans la balance. À partir de là **aucun bust n'est plus jamais détecté** (toute
        comparaison avec `NaN` est fausse) et `P(funded)` s'effondre sans raison lisible.

        Toutes les bornes ci-dessous sont donc écrites en **encadrement**, jamais en disjonction.
        C'est vérifié champ par champ (check 43), pas supposé.
        """
        if not 0 < self.risque_base < 1:
            raise ValueError(
                f"risque_base est une FRACTION du capital initial, dans ]0, 1[ — reçu "
                f"{self.risque_base!r}. À 0 le sizing ne risque rien et tout le mécanisme est "
                "mort sans le dire ; à 1 un seul trade joue le compte entier."
            )
        if not self.hausse_additive > 0:
            raise ValueError(
                f"hausse_additive doit être > 0 — reçu {self.hausse_additive!r}. À 0, `m` ne peut "
                "plus que descendre : l'AIMD dégénère en décroissance multiplicative pure, ce que "
                "H1 n'est pas. Rien ne planterait, le sizing continuerait de fonctionner."
            )
        if not isinstance(self.seuil_journees_perdantes, int) or isinstance(
            self.seuil_journees_perdantes, bool
        ):
            raise TypeError(
                f"seuil_journees_perdantes compte des JOURNÉES : c'est un entier, reçu "
                f"{self.seuil_journees_perdantes!r}. Un flottant comme 2,5 ne planterait pas — la "
                "comparaison marcherait et la coupe tomberait à la 3ᵉ journée, pas à la 2ᵉ. La "
                "grille de la phase 6 étant construite par programme, la valeur peut arriver en "
                "flottant sans qu'on l'ait voulu."
            )
        if self.seuil_journees_perdantes < 2:
            raise ValueError(
                f"seuil_journees_perdantes (k) vaut au moins 2 — reçu "
                f"{self.seuil_journees_perdantes}. H14 est explicite : « une journée perdante "
                "isolée ne coupe JAMAIS ». À k=1 l'hystérésis n'existe plus et le sizing coupe "
                "sur l'événement le plus banal du système — 42,8 % des journées du dev sont "
                "perdantes, et une perdante isolée arrive 60 % du temps."
            )
        if not 0 < self.facteur_coupe < 1:
            raise ValueError(
                f"facteur_coupe est dans ]0, 1[ — reçu {self.facteur_coupe!r}. À 1 la coupe ne "
                "coupe rien ; au-dessus de 1 elle **augmente** `m` — le frein devient un "
                "accélérateur, et il se déclenche précisément dans les mauvaises passes."
            )
        if not self.m_min > 0:
            raise ValueError(
                f"m_min doit être > 0 — reçu {self.m_min!r}. À 0 le risque tombe à zéro "
                "définitivement dès la première coupe ; en négatif il devient négatif, donc le "
                "nombre d'unités aussi : la position partirait dans le sens INVERSE de la "
                "décision, sans que rien ne le signale."
            )
        if not self.m_min <= self.m_max:
            raise ValueError(
                f"m_min ({self.m_min!r}) doit être <= m_max ({self.m_max!r}). Bornes croisées, le "
                "`min` de la hausse et le `max` de la coupe se contredisent et le résultat "
                "dépend de l'ordre des opérations. ⚠️ L'ÉGALITÉ est autorisée exprès : "
                "`m_min == m_max == m_initial` est la config DÉGÉNÉRÉE qui doit reproduire le "
                "sizing constant au centime près — c'est un critère de recette de la tâche 5.6."
            )
        if not self.m_min <= self.m_initial <= self.m_max:
            raise ValueError(
                f"m_initial ({self.m_initial!r}) doit être dans [m_min, m_max] = "
                f"[{self.m_min!r}, {self.m_max!r}]. Hors bornes, les premiers trades sortiraient à "
                "un risque que les bornes déclarent impossible, et `m` se recalerait en silence au "
                "premier ajustement — un début de chemin faussé, invisible ensuite."
            )
        if self.airbag_n is not None:
            if not isinstance(self.airbag_n, int) or isinstance(self.airbag_n, bool):
                raise TypeError(
                    f"airbag_n compte des TRADES : c'est un entier, ou `None` pour désactiver — "
                    f"reçu {self.airbag_n!r}."
                )
            if self.airbag_n < 1:
                raise ValueError(
                    f"airbag_n vaut au moins 1 — reçu {self.airbag_n}. À 0 le calcul du plafond "
                    "divise par zéro ; en négatif le plafond devient négatif, donc le risque "
                    "négatif, donc la position INVERSÉE, sans que rien ne plante. Pour désactiver "
                    "l'airbag il faut écrire `airbag_n=None` : la désactivation doit être "
                    "explicite (H9), jamais obtenue par une valeur qui a l'air neutre."
                )

    def valider_contre_regles(self, regles: ReglesFTMO) -> None:
        """
        **Le seul contrôle qui a besoin du barème** : le risque maximal atteignable doit rester
        sous le plancher journalier. Appelé par `SuiviAIMD` à la construction, donc impossible à
        oublier — un couple (config, barème) incohérent ne peut pas produire de suivi.

        `risque_base x m_max` est la plus grosse perte qu'un trade puisse encaisser, en fraction
        du capital initial. Le plancher journalier autorise `plancher_journalier_pct` de ce même
        capital — un montant **fixe** depuis F2. Si le produit l'atteint, **un seul trade perdant
        tue la journée**, et un sizing adaptatif n'a plus rien à adapter : il garantit la mort.

        **Strictement inférieur**, pas « inférieur ou égal » : à égalité exacte, la perte amène
        l'équity **sur** le plancher, et `observer` buste au plancher (`<=`, classe F18). Une
        config à exactement 5 % serait donc mortelle tout en paraissant à la limite du légal.

        La borne est **LUE** dans `ReglesFTMO`, jamais recopiée (piège n°5) : un `0.05` en dur ici
        survivrait à un changement de barème et laisserait passer une config mortelle.

        ⚠️ **Indépendante de l'airbag** (H16) : l'airbag borne l'exposition quand le compte est
        déjà bas ; cette garde interdit une config qui serait mortelle **même sur un compte
        intact**. Les confondre laisserait croire qu'un airbag actif dispense de la vérifier.
        """
        pire_risque = self.risque_base * self.m_max
        if not pire_risque < regles.plancher_journalier_pct:
            raise ValueError(
                f"risque_base x m_max = {pire_risque!r} atteint le plancher journalier "
                f"({regles.plancher_journalier_pct!r} du capital initial) : un seul trade perdant "
                "tuerait la journée, et le sizing adaptatif garantirait la mort au lieu de "
                "l'éviter. Baisser risque_base ou m_max."
            )


class SuiviAIMD:
    """
    La machine à états du sizing — **un objet par simulation** (H4 ②).

    Elle ne connaît ni barres, ni prix, ni trades : seulement des frontières de journée FTMO
    annoncées, et des clôtures de trade signalées. C'est toujours l'appelant qui **calcule** la
    notion de `jour_ftmo` (B11) — ce module ne la dérive jamais d'un horodatage, il la reçoit.

    Emploi — c'est exactement ce que fait `simuler` depuis la tâche 5.6 ::

        aimd = SuiviAIMD(config, regles)
        aimd.frontiere_de_journee(lundi, balance=50_000.0)   # ouvre la 1ʳᵉ journée, ne classe rien
        aimd.marquer_cloture_de_trade()                      # à chaque trade qui se FERME
        aimd.frontiere_de_journee(mardi, balance=50_400.0)   # classe lundi, ouvre mardi
        risque = aimd.risque_effectif(equity, suivi.plancher_effectif())

    **Un seul appel par frontière, et une seule balance.** Ce n'est pas de l'élégance : la balance
    de clôture d'une journée **EST** la balance d'ouverture de la suivante, donc les passer
    séparément ouvrirait la porte à un décalage d'un jour — une erreur qui ne plante pas et qui
    fausse toutes les classifications.

    ────────────────────────────────────────────────────────────────────────────────────
    Pourquoi la journée est passée en argument — décision de Hugo, 2026-08-16
    ────────────────────────────────────────────────────────────────────────────────────
    `SuiviChallenge.nouvelle_journee(balance)` ne prend **pas** de jour : l'appelant annonce une
    frontière et le module le croit. Ici on diverge délibérément, et voici contre quoi.

    Le pire bug possible du branchement (5.6) est **une ligne au mauvais niveau d'indentation** :
    prévenir l'AIMD à chaque **barre** au lieu de chaque **frontière de journée**. Ce qui se
    passerait alors est parfaitement muet — une barre sans clôture donne « sans trade », donc
    transparente ; une barre où un trade se ferme donne « gagnante » ou « perdante ». Autrement
    dit **l'AIMD réagirait trade par trade**, c'est-à-dire exactement le déclencheur que **H2 a
    explicitement écarté** (*« le résultat du trade isolé — à 1,62 trade/jour, c'est du bruit
    pur »*). Rien ne planterait, `m` bougerait, le diagnostic se remplirait, et le `P(funded)`
    sortirait un chiffre crédible qui ne décrirait plus la mécanique décidée.

    En recevant le jour, le module **refuse qu'on lui annonce deux fois la même journée** : appelé
    à chaque barre, il lève dès la deuxième. Le bug passe d'« attrapé par une preuve » à
    « impossible ». La mutation de site d'appel reste écrite en 5.7 par-dessus — **double
    protection, décision de Hugo** : la mutation vérifie que la ligne est au bon endroit dans
    *cette* version du code, la garde tient quelle que soit la version.

    ⚠️ **`SuiviChallenge` porte le même trou et n'a pas la garde** : appelé à chaque barre, son
    plancher journalier se recalculerait à chaque barre et suivrait la balance — presque plus
    aucun bust journalier, donc `P(funded)` gonflé, sans symptôme. L'asymétrie est **assumée et
    connue** : la corriger rouvrirait un module prouvé et stable depuis la phase 4, ce qui est une
    tâche à part entière. Ne pas en conclure que le juge est protégé de la même façon.

    ⚠️ **La dernière journée d'une simulation n'est jamais classée** : aucune frontière ne la
    suit. C'est un no-op par construction — `m` n'influence que les trades **suivants** (H2 ③),
    et il n'y en a pas. À savoir en lisant le diagnostic de 5.6 : la somme de `bilans` vaut le
    nombre de journées **closes**, soit une de moins que les journées vécues.
    """

    def __init__(self, config: AIMDConfig, regles: ReglesFTMO) -> None:
        self.config: AIMDConfig = config
        #: Le barème du challenge. **Requis, sans défaut** — contrairement à `SuiviChallenge`, qui
        #: accepte `None` et retombe sur `ReglesFTMO()`. La divergence est voulue : l'airbag lit
        #: `capital_initial` ici pour convertir sa marge en dollars vers une fraction de risque
        #: (voir `risque_effectif`). Un barème implicite sur un compte 10k plafonnerait le risque
        #: comme s'il valait 50k — cinq fois trop haut, et parfaitement silencieux. C'est le
        #: piège n°5 : **on importe, on ne recopie pas**, et on ne devine surtout pas.
        self.regles: ReglesFTMO = regles
        # Tâche 5.5 — la garde qui a besoin des DEUX : `risque_base x m_max` sous le plancher
        # journalier. Ici et pas dans `AIMDConfig.__post_init__`, parce que la config seule ne
        # connaît pas le barème. Appelée à la construction, donc un couple (config, barème)
        # incohérent ne peut produire aucun suivi.
        config.valider_contre_regles(regles)
        #: Le multiplicateur courant. Part de `m₀` (H4 ①).
        self.m: float = config.m_initial
        #: La série en cours de journées perdantes **consécutives** (H14).
        self.journees_perdantes_consecutives: int = 0

        # ── diagnostic (remonté dans `Distribution` en 5.6) ──────────────────────────────
        # Sans lui, la phase 6 ne distingue pas une config qui protège d'une config inerte.
        #: Montées **effectives** : `m` a strictement augmenté. Une gagnante au plafond n'y entre
        #: pas — sinon une config bloquée à `m_max` paraîtrait active.
        self.n_montees: int = 0
        #: Coupes **effectives** : `m` a strictement diminué.
        self.n_coupes: int = 0
        #: Déclenchements du seuil `k`, **effet ou non**. Toujours `>= n_coupes` : au plancher, la
        #: coupe se déclenche sans faire bouger `m`. C'est ce compteur — et lui seul — qui dira si
        #: `k = 4` est vraiment inerte (H14).
        self.n_declenchements_coupe: int = 0
        #: Combien de journées de chaque issue. Répond à « combien de journées vides ? » (I13).
        self.bilans: dict[BilanJournee, int] = {b: 0 for b in BilanJournee}
        #: Combien de fois l'airbag a **mordu** — pas combien de fois il a été consulté (H16).
        #: ⚠️ **Sans ce compteur, on ne peut pas distinguer « airbag utile » de « airbag
        #: décoration »**, et l'axe « désactivé / 2 / 3 » de la grille phase 6 ne séparerait rien.
        #: H16 l'exige explicitement : *« à vérifier, pas à supposer »*.
        self.n_declenchements_airbag: int = 0

        # ── journée en cours ─────────────────────────────────────────────────────────────
        self._jour: date | None = None
        self._balance_ouverture: float | None = None
        self._n_clotures_journee: int = 0

    # ── lecture ─────────────────────────────────────────────────────────────────────────
    @property
    def journee_ouverte(self) -> bool:
        """Vrai dès qu'une frontière de journée a été annoncée."""
        return self._balance_ouverture is not None

    @property
    def jour_courant(self) -> date:
        """Le `jour_ftmo` de la journée en cours, tel qu'il a été **annoncé** (jamais dérivé)."""
        if self._jour is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `frontiere_de_journee` d'abord."
            )
        return self._jour

    @property
    def balance_ouverture(self) -> float:
        """
        La balance sur laquelle la journée en cours a été ouverte — l'ancre du bilan (H2 ①).

        En lecture seule : la déplacer en cours de journée changerait le résultat de la journée
        après coup, ce qui est la définition du look-ahead que H2 ③ ferme.
        """
        if self._balance_ouverture is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `frontiere_de_journee` d'abord."
            )
        return self._balance_ouverture

    @property
    def _risque_nominal(self) -> float:
        """
        `risque_base x m` — **la seule chose que l'AIMD ajuste** (H10). Ni la distance du stop, ni
        le choix des trades, ni la direction, ni la cadence.

        ⚠️ **PRIVÉE depuis la tâche 5.4, et c'est le point.** Ce n'est pas le risque à appliquer :
        l'airbag (H16) le plafonne ensuite. Un site d'appel qui lirait le nominal contournerait le
        filet **en silence**, et le compte mourrait sur un trade que l'airbag était censé retenir.
        `risque_effectif` est le **seul** accès public au risque.
        """
        return self.config.risque_base * self.m

    def risque_effectif(self, equity: float, plancher: float) -> float:
        """
        **Le seul risque à appliquer** : le nominal (H10), plafonné par l'airbag de survie (H16).

        `plancher` doit être **le plancher le plus proche des deux** — journalier ou total —,
        c'est-à-dire exactement ce que retourne `SuiviChallenge.plancher_effectif()`. Chez FTMO
        **les deux tuent**, donc ne regarder que le total ferait mesurer une marge qui n'existe
        pas. Il est **lu** là-bas, jamais recalculé ici (piège n°5).

        ────────────────────────────────────────────────────────────────────────────────
        Le trou que le déclencheur « résultat de la journée » ne voit pas
        ────────────────────────────────────────────────────────────────────────────────
        Compte à 46 000, barrière à 45 000 : il reste **1 000 $**. Deux bonnes journées viennent
        de remonter `m`, donc le sizing nominal veut risquer 1 000 $ — **un seul trade perdant et
        le challenge est fini**. Le résultat des journées récentes ne dit **rien** de cette
        proximité ; pire, il vient de dire « tout va bien ». L'airbag borne l'**exposition
        absolue**, là où le plancher de `m` ne borne que l'**adaptation** : deux choses
        différentes, et H16 insiste pour qu'on ne les confonde pas.

        ────────────────────────────────────────────────────────────────────────────────
        ⚠️ La formule de H16 est un raccourci — ses deux termes n'ont pas la même unité
        ────────────────────────────────────────────────────────────────────────────────
        H16 écrit `min(risque_base x m, marge / N)`. Le premier terme est une **fraction** du
        capital initial, le second des **dollars** : les comparer directement n'a pas de sens, et
        le faire quand même donnerait un plafond absurde de plusieurs centaines (donc jamais
        mordant, donc un airbag décoratif que rien ne signalerait).

        La conversion est forcée par le sizing du simulateur, où la perte au stop vaut
        `capital_initial x risque`. La contrainte « au moins `N` trades perdants pour mourir »
        s'écrit donc `capital_initial x risque x N <= marge`, soit ::

            risque <= marge / (N x capital_initial)

        `capital_initial` est **lu** dans `ReglesFTMO`, jamais recopié — sur un compte 10k, un
        50 000 en dur plafonnerait cinq fois trop haut, en silence.

        ────────────────────────────────────────────────────────────────────────────────
        Deux propriétés à ne pas perdre
        ────────────────────────────────────────────────────────────────────────────────
        - **C'est un plafond, jamais un plancher** : quand l'airbag ne mord pas, il ne remonte
          **rien**. Un `max` à la place du `min` transformerait le filet en accélérateur.
        - **Il ne double pas l'AIMD** : quand `m` a déjà coupé, le nominal est plus petit, donc
          l'airbag est **encore plus loin** de mordre. Il ne sert que dans le cas précis où l'AIMD
          ne voit pas le danger — compte bas, journées récentes gagnantes.

        ⚠️ **Un appel par trade** : le compteur de déclenchements est incrémenté ici. L'appeler
        deux fois pour la même position gonflerait le diagnostic de la phase 6.
        """
        for nom, valeur in (("equity", equity), ("plancher", plancher)):
            if not math.isfinite(valeur):
                raise ValueError(
                    f"{nom} doit être un nombre fini, reçu {valeur!r} — un non-fini traverserait "
                    "les comparaisons sans lever et donnerait un plafond de risque insensé."
                )
        marge = equity - plancher
        if marge <= 0:
            # Le compte est mort ou exactement à la barrière : `observer` l'aurait déjà tranché.
            # Ne PAS lever ici serait la pire issue possible — une marge négative donne un plafond
            # négatif, donc un risque négatif, donc un nombre d'unités négatif : la position
            # partirait dans le SENS INVERSE de la décision. Rien ne planterait, et le compte
            # gagnerait sur ses pires trades.
            raise RuntimeError(
                f"Marge nulle ou négative : equity {equity} contre plancher {plancher}. Le compte "
                "est déjà mort — dimensionner ici produirait un risque NÉGATIF, donc une position "
                "inversée, sans que rien ne le signale."
            )
        nominal = self._risque_nominal
        if self.config.airbag_n is None:
            return nominal  # H16 : désactivé, et la désactivation était explicite (H9)
        plafond = marge / (self.config.airbag_n * self.regles.capital_initial)
        if plafond < nominal:
            self.n_declenchements_airbag += 1
            return plafond
        return nominal

    # ── pilotage ────────────────────────────────────────────────────────────────────────
    def marquer_cloture_de_trade(self) -> None:
        """
        Signale qu'un trade s'est **fermé** dans la journée en cours.

        C'est ce compteur — et non la variation de balance — qui sépare une journée **nulle**
        d'une journée **sans trade** (H15). Les deux valent zéro : sans lui, elles seraient
        indistinguables et une journée vide casserait les séries de pertes.

        À appeler à la **fermeture**, jamais à l'ouverture : le résultat d'une position tombe dans
        la journée où elle se ferme (H2 ①).
        """
        if self._balance_ouverture is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `frontiere_de_journee` d'abord. Compter "
                "une clôture hors journée la ferait tomber dans la journée suivante, ou nulle "
                "part — et le bilan resterait plausible."
            )
        self._n_clotures_journee += 1

    def frontiere_de_journee(self, jour: date, balance: float) -> BilanJournee | None:
        """
        **Clôt la journée écoulée et ouvre la suivante**, en un seul appel et une seule balance.

        `jour` est le `jour_ftmo` de la journée qui **s'ouvre**, tel que `ftmo_cleaner` l'a
        calculé (B11) — il est **lu**, jamais dérivé d'un horodatage par ce module.

        Retourne le bilan de la journée qui vient de se clore, ou `None` au tout premier appel
        (il n'y a pas de journée précédente à classer).

        À appeler à chaque changement de `jour_ftmo` (17:00 NY, minuit serveur — B4, piège n°1),
        **y compris les journées sans aucun trade** : c'est ce qui permet de les compter comme
        telles au lieu de les faire disparaître.

        ⚠️ **`balance`, jamais `equity`** (H2 ①). Une équity contient le flottant d'une position
        ouverte : la passer ferait entrer un résultat non réalisé dans une décision de sizing, et
        le même résultat serait compté **deux fois** — une fois flottant, une fois réalisé.

        ⚠️ **Ne pas utiliser au passage de phase** : la balance y saute au capital de départ, et
        classer ce saut fabriquerait une journée. `passage_de_phase` est là pour ça.

        **La garde de progression stricte est la moitié « structurelle » de la double protection
        décidée par Hugo le 2026-08-16** (voir l'en-tête de la classe). Annoncer deux fois la même
        journée — ce que ferait un appel placé dans la boucle des barres au lieu du bloc de
        frontière — lève **immédiatement**, au lieu de transformer l'AIMD en mécanique trade par
        trade que H2 a écartée. Reculer dans le temps lève aussi : ce serait un panel mal trié,
        et le bilan porterait alors sur deux journées mélangées.
        """
        if self._jour is not None and jour <= self._jour:
            raise RuntimeError(
                f"Journée FTMO annoncée : {jour}, alors que {self._jour} est déjà ouverte. Une "
                "frontière doit STRICTEMENT avancer. Deux appels pour la même journée signalent "
                "un appel placé dans la boucle des barres au lieu du bloc de frontière — l'AIMD "
                "réagirait alors trade par trade, le déclencheur que H2 a explicitement écarté, "
                "et rien d'autre ne le signalerait."
            )
        bilan: BilanJournee | None = None
        if self._balance_ouverture is not None:
            bilan = classer_journee(balance - self._balance_ouverture, self._n_clotures_journee)
            self.bilans[bilan] += 1
            self._appliquer(bilan)
        self._jour = jour
        self._balance_ouverture = balance
        self._n_clotures_journee = 0
        return bilan

    def passage_de_phase(self, jour: date, balance: float) -> None:
        """
        Le compte est **neuf** (F17, H4 ③). Remet l'état si `reset_au_passage_de_phase`.

        `jour` doit être **exactement la journée en cours** : un passage de phase se produit sur
        une barre, à l'intérieur d'une journée FTMO déjà ouverte, jamais à une frontière. Exiger
        l'égalité plutôt que de ne rien demander transforme un enchaînement d'appels incohérent en
        erreur immédiate, au même titre que la garde de `frontiere_de_journee`.

        ⚠️ **La journée en cours est ABANDONNÉE, jamais close — dans les deux cas.** Au passage de
        phase, le simulateur vient de remettre la balance au capital de départ : la variation
        depuis l'ouverture de la journée n'est plus un résultat de trading mais un **saut
        comptable**. La classer fabriquerait une journée gagnante ou perdante de plusieurs
        milliers de dollars, qui ferait monter ou couper `m` sans qu'aucun trade n'ait eu lieu.
        Rien ne planterait, et `P(funded)` resterait crédible.

        Le **diagnostic** (`n_montees`, `n_coupes`, `bilans`) n'est PAS remis à zéro : il décrit
        le challenge entier, pas la phase. Le remettre à zéro effacerait l'histoire de la phase 1
        de la seule mesure qui dira, en phase 6, si une config fait réellement quelque chose.
        """
        if self._jour is None:
            raise RuntimeError(
                "Aucune journée FTMO en cours : appeler `frontiere_de_journee` d'abord. Un "
                "passage de phase se produit toujours à l'intérieur d'une journée ouverte."
            )
        if jour != self._jour:
            raise RuntimeError(
                f"Passage de phase annoncé sur {jour}, alors que {self._jour} est la journée "
                "ouverte. Un passage de phase se produit sur une barre, à l'intérieur d'une "
                "journée FTMO, jamais à une frontière — un écart signale un enchaînement d'appels "
                "incohérent côté simulateur."
            )
        if self.config.reset_au_passage_de_phase:
            self.m = self.config.m_initial
            self.journees_perdantes_consecutives = 0
        self._balance_ouverture = balance
        self._n_clotures_journee = 0

    # ── la mécanique AIMD ───────────────────────────────────────────────────────────────
    def _appliquer(self, bilan: BilanJournee) -> None:
        """
        Applique un bilan de journée. **Additive à la hausse, multiplicative à la baisse** (H1).

        Les bornes sont appliquées par `min`/`max` plutôt que par un test d'égalité : c'est ce qui
        rend l'accumulation flottante inoffensive. Répéter `m + 0.1` finit par dépasser un plafond
        de 1e-16 ; le `min` l'écrase, un `if m == m_max` ne l'aurait pas vu (pièges n°6 et n°7).
        """
        if bilan is BilanJournee.SANS_TRADE:
            # H15 — totalement transparente : ni montée, ni série cassée, ni série prolongée.
            return

        if bilan is BilanJournee.GAGNANTE:
            self.journees_perdantes_consecutives = 0
            nouveau = min(self.m + self.config.hausse_additive, self.config.m_max)
            if nouveau > self.m:
                self.n_montees += 1
            self.m = nouveau
            return

        if bilan is BilanJournee.NULLE:
            # Des trades ont conclu, le régime s'est exprimé : la série est cassée. Mais rien n'a
            # été gagné, donc `m` ne monte pas.
            self.journees_perdantes_consecutives = 0
            return

        # PERDANTE — la série s'allonge, et la coupe n'arrive qu'au seuil (H14).
        self.journees_perdantes_consecutives += 1
        if self.journees_perdantes_consecutives < self.config.seuil_journees_perdantes:
            return
        self.n_declenchements_coupe += 1
        nouveau = max(self.m * self.config.facteur_coupe, self.config.m_min)
        if nouveau < self.m:
            self.n_coupes += 1
        self.m = nouveau
        # H14, lecture (a) tranchée par Hugo le 2026-08-15 : il faut `k` NOUVELLES journées
        # perdantes pour couper à nouveau. Sans ce reset, un seuil à 2 dégénère en « coupe à
        # chaque journée perdante dès la 2ᵉ » — la réaction au bruit qu'on venait de supprimer.
        self.journees_perdantes_consecutives = 0
