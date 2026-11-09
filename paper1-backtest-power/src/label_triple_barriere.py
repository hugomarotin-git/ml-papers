"""
Le label triple-barrière — tâche 2.1.

Ce que le modèle apprendra à prédire. Pour chaque barre candidate `t`, une seule question :

    « Si j'ouvrais un trade à la barre suivante, qu'est-ce qui se passerait en premier ? »

C'est le premier module qui **décide** quelque chose. Tout ce qui suit — features, modèle,
juge — hérite de ses choix, et une erreur ici ne se verrait nulle part ailleurs : elle
produirait simplement un modèle qui apprend la mauvaise cible.

────────────────────────────────────────────────────────────────────────────────────────
Quatre issues, pas trois (décision C12)
────────────────────────────────────────────────────────────────────────────────────────
========  =====================================  ==========  ==========
issue     ce qui s'est passé                     long        short
========  =====================================  ==========  ==========
haut      barrière haute touchée en premier      gagne       perd
bas       barrière basse touchée en premier      perd        gagne
double    les deux dans la MÊME bougie           **perd**    **perd**
timeout   aucune dans l'horizon                  clôture     clôture
========  =====================================  ==========  ==========

Le cas `double` vaut 1,9 % des candidats et **ne rentre pas** dans un schéma à trois classes.
Une bougie horaire ne dit pas dans quel ordre ses extrêmes ont été atteints : on ne peut donc
pas savoir laquelle des deux barrières est venue en premier. Le classer en `timeout` serait
faux — un timeout se clôture au marché avec un petit P&L, alors qu'ici c'est une perte pleine.
L'affecter arbitrairement à `haut` ou `bas` injecterait du bruit pur, que le modèle tenterait
d'apprendre.

Mais ce cas a un sens de trading parfaitement clair : **quelle que soit la direction prise, on
est stoppé.** D'où la convention pessimiste — ça peut avoir gagné, on compte le stop au cas où.

────────────────────────────────────────────────────────────────────────────────────────
Les règles anti-look-ahead, et pourquoi chacune
────────────────────────────────────────────────────────────────────────────────────────
1. **Entrée à l'Open de t+1, jamais au Close de t** (C5). Entrer au Close[t] supposerait une
   exécution instantanée au prix même qui a servi à décider — un biais optimiste gratuit.
2. **ATR figé à t** (C4). Il ne voit que des barres ≤ t. Comme la décision se prend à Close[t],
   ça n'introduit aucune fuite.
3. **La barre d'entrée est vérifiée comme les autres.** Ce n'est pas du look-ahead : l'Open
   précède mécaniquement le High et le Low dans le déroulé réel de la bougie.

────────────────────────────────────────────────────────────────────────────────────────
Les 7 barres doivent SE SUIVRE — la règle qui a remplacé trois autres
────────────────────────────────────────────────────────────────────────────────────────
La barre candidate, l'entrée et tout l'horizon doivent être **contigus** : sept barres
consécutives, 60 minutes d'écart chacune.

Cette règle est née d'un bug trouvé le 2026-08-08. L'ancienne version vérifiait seulement que
l'entrée et la fin étaient dans la même journée FTMO — sans vérifier que le CANDIDAT y était
aussi. Un candidat à la clôture d'une séance recevait donc une entrée à l'ouverture de la
suivante : **on décidait vendredi à 16 h pour entrer dimanche à 17 h**, en traversant un
week-end entier. 1 419 labels dans ce cas, soit 5,6 % du total.

Trois conséquences, et la dernière est la pire :

- le label décrivait un **trade impossible** — le marché est fermé entre les deux ;
- les features auraient été **périmées** de 49 heures au moment d'entrer ;
- le dernier candidat du dev recevait un label calculé sur des barres de janvier 2025, donc
  **une fuite du hold-out**.

La contiguïté totale règle les trois d'un coup, et elle **remplace** l'ancienne règle de
journée : vérifié, aucun cas n'est contigu tout en franchissant une séance — une frontière de
séance comporte toujours une coupure.

⚠️ **Comptage des heures, à ne pas confondre.** Un horodatage marque le DÉBUT d'une barre. Pour
les 6 barres de vie du trade, l'écart entre le premier et le dernier horodatage vaut donc 5 h
(6 barres, 5 intervalles) — mais **le trade vit bien 6 heures**, de l'ouverture de la première
à la fin de la dernière. C'est cette seconde valeur qui a un sens ; la première n'est qu'une
conséquence de la convention d'horodatage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from atr import ATR_PERIODE, atr_causal
from ftmo_cleaner import DATA_DIR, load_clean
from intraday_filter import assert_aucune_barre_journaliere, filtrer_intraday

#: Largeur des barrières, en multiples d'ATR. RR 1:1 — le TP et le SL sont à la même distance.
K_ATR: float = 1.5
#: Nombre de barres de vie du trade, barre d'entrée comprise (décision C6).
HORIZON_BARRES: int = 6
#: ⚠️ `ATR_PERIODE` et `atr_causal` vivent désormais dans `atr` (déplacés le 2026-09-23, P12) :
#: l'ATR n'est le label de personne. Importés ici, plus définis.

#: Les quatre issues possibles.
ISSUE_HAUT, ISSUE_BAS, ISSUE_DOUBLE, ISSUE_TIMEOUT = "haut", "bas", "double", "timeout"

#: Pourquoi une barre ne produit pas de label. Aucune n'est une anomalie : ce sont les bornes.
RAISON_ATR = "atr_indisponible"
RAISON_CONTIGUITE = "barres_non_contigues"
RAISON_FIN = "fin_des_donnees"

DEFAULT_OUT: Path = DATA_DIR / "v2" / "nas100_h1_labelle.csv"


def _resoudre_une_entree(
    haut: np.ndarray,
    bas: np.ndarray,
    debut: int,
    fin: int,
    barriere_haute: float,
    barriere_basse: float,
) -> tuple[str, int]:
    """
    Déroule les barres `debut..fin` et renvoie l'issue et l'indice où elle est survenue.

    Une bougie ne dit pas dans quel ordre son High et son Low ont été atteints. Quand les deux
    barrières sont franchissables sur la même bougie, on ne tranche donc pas : l'issue est
    `double`, et les deux directions y perdent (convention pessimiste, décision C12).
    """
    for i in range(debut, fin + 1):
        touche_haut = haut[i] >= barriere_haute
        touche_bas = bas[i] <= barriere_basse
        if touche_haut and touche_bas:
            return ISSUE_DOUBLE, i
        if touche_haut:
            return ISSUE_HAUT, i
        if touche_bas:
            return ISSUE_BAS, i
    return ISSUE_TIMEOUT, fin


#: Ce que chaque issue rapporte à un long et à un short. `timeout` vaut 0 : la position se
#: clôture au marché, ni gagnante ni perdante au sens des barrières.
GAINS: dict[str, tuple[int, int]] = {
    ISSUE_HAUT: (1, -1),
    ISSUE_BAS: (-1, 1),
    ISSUE_DOUBLE: (-1, -1),
    ISSUE_TIMEOUT: (0, 0),
}


def construire_labels(
    df: pd.DataFrame,
    k_atr: float = K_ATR,
    horizon: int = HORIZON_BARRES,
    atr_periode: int = ATR_PERIODE,
) -> pd.DataFrame:
    """
    Ajoute le label à chaque barre candidate, et la raison pour celles qui n'en ont pas.

    Une barre ne produit pas de label dans quatre cas, tous structurels :

    - l'ATR n'est pas encore disponible (début de série) ;
    - **les 7 barres ne se suivent pas** (décision C9) — c'est le cas le plus fréquent, il
      concerne la fin de chaque séance et les trous internes ;
    - il ne reste plus assez de barres.

    ⚠️ Il n'y a **plus** de borne sur la durée réelle. Elle existait (décision C13), et la
    contiguïté l'a rendue **inatteignable** : les 6 barres de vie du trade couvrent exactement
    6 heures, jamais plus. Une garde qu'aucun test ne peut exercer donne une fausse confiance — c'est le test
    de mutation qui l'a signalé, en montrant qu'on pouvait la supprimer sans qu'aucune preuve ne
    bronche. Retirée le 2026-08-08.

    Rien de tout cela n'est une anomalie. La colonne `raison_sans_label` permet de le vérifier
    plutôt que de le supposer.
    """
    # ⚠️ **`k_atr` n'était validé nulle part — corrigé le 2026-08-18** (piège n°8, audit du 18/08).
    # `k_atr = NaN` donne des barrières NaN. Or toute comparaison avec NaN est fausse : AUCUNE
    # barrière ne serait jamais franchie, et **100 % des barres sortiraient en `timeout`**. Le
    # fichier de labels ressemblerait alors à « ce marché n'a pas de structure » — la conclusion
    # la plus coûteuse possible, tirée d'un bug. À 0, les deux barrières collent au prix d'entrée
    # et tout devient `double` ; en négatif, la barrière haute passe SOUS la basse.
    if not math.isfinite(k_atr) or not k_atr > 0:
        raise ValueError(
            f"k_atr doit être un nombre fini > 0, reçu {k_atr!r}. Un NaN produirait 100 % de "
            "timeout (aucune comparaison ne serait jamais vraie), un 0 produirait 100 % de "
            "double, et un négatif inverserait les deux barrières."
        )
    manquantes = {"DateTime", "Open", "High", "Low", "Close", "jour_ftmo", "gap_kind"} - set(
        df.columns
    )
    if manquantes:
        raise ValueError(
            f"Colonnes absentes : {sorted(manquantes)}. Ce module attend un panel produit par "
            "`ftmo_cleaner.clean_ftmo_h1` — `gap_kind` sert à vérifier la contiguïté des barres."
        )
    assert_aucune_barre_journaliere(df)

    df = df.sort_values("DateTime").reset_index(drop=True)
    haut = np.asarray(df["High"], dtype=float)
    bas = np.asarray(df["Low"], dtype=float)
    ouverture = np.asarray(df["Open"], dtype=float)
    horodatage = np.asarray(pd.Series(df["DateTime"]), dtype="datetime64[ns]")
    contigu = np.asarray(pd.Series(df["gap_kind"]) == "contigu", dtype=bool)
    atr = atr_causal(df, atr_periode)

    n = len(df)

    # `gap_kind` est écrit par le nettoyeur AVANT tout retrait de lignes (filtre intraday) et
    # n'est jamais recalculé : si une barre voisine a disparu depuis, la colonne ment. Les
    # horodatages du panel REÇU, eux, ne peuvent pas mentir — on refuse donc tout « contigu »
    # que l'écart réel contredit. C'est la direction dangereuse : la relire telle quelle ferait
    # enjamber une coupure à un label, exactement la classe de bug C9.
    ecart_ok = np.zeros(n, dtype=bool)
    if n > 1:
        ecart_ok[1:] = (horodatage[1:] - horodatage[:-1]) == np.timedelta64(60, "m")
    if bool((contigu & ~ecart_ok).any()):
        raise AssertionError(
            "gap_kind annonce « contigu » sur une barre dont l'écart réel n'est pas de 60 min : "
            "colonne périmée — des lignes ont été retirées après le nettoyage. La contiguïté C9 "
            "ne peut plus être lue dessus, il faut la recalculer sur le panel reçu."
        )

    issue = np.full(n, "", dtype=object)
    raison = np.full(n, "", dtype=object)
    nat = np.datetime64("NaT", "ns")
    entree_dt = np.full(n, nat, dtype="datetime64[ns]")
    fin_dt = np.full(n, nat, dtype="datetime64[ns]")
    duree = np.full(n, np.nan)
    prix_entree = np.full(n, np.nan)
    b_haute = np.full(n, np.nan)
    b_basse = np.full(n, np.nan)

    for t in range(n):
        debut = t + 1
        fin = t + horizon
        if fin >= n:
            raison[t] = RAISON_FIN
            continue
        if np.isnan(atr[t]) or atr[t] <= 0:
            raison[t] = RAISON_ATR
            continue
        if not contigu[debut : fin + 1].all():
            raison[t] = RAISON_CONTIGUITE
            continue
        entree = ouverture[debut]
        distance = k_atr * atr[t]
        hausse, baisse = entree + distance, entree - distance
        res, idx = _resoudre_une_entree(haut, bas, debut, fin, hausse, baisse)

        issue[t] = res
        entree_dt[t] = horodatage[debut]
        fin_dt[t] = horodatage[idx]
        duree[t] = idx - debut + 1
        prix_entree[t] = entree
        b_haute[t] = hausse
        b_basse[t] = baisse

    out = df.copy()
    out["atr"] = atr
    out["issue"] = [i if i else None for i in issue]
    out["raison_sans_label"] = [None if i else r for i, r in zip(issue, raison, strict=True)]
    out["label_long"] = [GAINS[i][0] if i else np.nan for i in issue]
    out["label_short"] = [GAINS[i][1] if i else np.nan for i in issue]
    out["prix_entree"] = prix_entree
    out["barriere_haute"] = b_haute
    out["barriere_basse"] = b_basse
    out["label_entree_dt"] = entree_dt
    out["label_fin_dt"] = fin_dt
    out["label_duree_barres"] = duree
    return out


#: Colonnes de date produites par ce module, à restaurer après un aller-retour CSV.
COLONNES_DATES: tuple[str, ...] = ("label_entree_dt", "label_fin_dt")


def load_labelle(path: Path) -> pd.DataFrame:
    """
    Relit un fichier labellisé en **restaurant tous ses types de date**.

    Même raison que `ftmo_cleaner.load_clean` : un CSV ne transporte pas les types, et
    `label_entree_dt` revient en texte. Une comparaison de dates sur du texte ne plante pas —
    elle renvoie simplement un résultat faux. Passer par cette fonction évite d'y penser.
    """
    df = load_clean(path)
    for col in COLONNES_DATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def resume(df: pd.DataFrame) -> str:
    """Répartition des issues et des raisons — à lire après chaque run."""
    n = len(df)
    avec = np.asarray(pd.Series(df["issue"]).notna())
    lignes = [
        f"  barres              : {n:,}",
        f"  labellisées         : {int(avec.sum()):,} ({100 * avec.mean():.1f} %)",
    ]
    comptes: dict[str, int] = {
        str(k): int(v) for k, v in pd.Series(df["issue"]).value_counts().items()
    }
    for nom in (ISSUE_HAUT, ISSUE_BAS, ISSUE_DOUBLE, ISSUE_TIMEOUT):
        c = comptes.get(nom, 0)
        lignes.append(
            f"    {nom:<9} : {c:>7,}  ({100 * c / max(int(avec.sum()), 1):.1f} % des labels)"
        )
    lignes.append("  sans label, par raison :")
    for nom, c in pd.Series(df["raison_sans_label"]).value_counts().items():
        lignes.append(f"    {nom!s:<28} : {int(c):>7,}")
    return "\n".join(lignes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Construit le label triple-barrière.")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "v2" / "nas100_h1_clean.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df, _ = filtrer_intraday(load_clean(args.input))
    labelle = construire_labels(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labelle.to_csv(args.out, index=False)

    print(f"\nLabel triple-barrière — {args.input.name} -> {args.out.name}")
    print(f"  k = {K_ATR} x ATR({ATR_PERIODE}), horizon {HORIZON_BARRES} barres contiguës")
    print(resume(labelle))
    print()


if __name__ == "__main__":
    main()
