"""
L'ATR causal — **déplacé** ici le 2026-09-23 (bloc 1 de la session de nettoyage, règle P12).

Il vivait dans `label_triple_barriere`. Or il ne lui appartient pas : un ATR n'est le label de
personne — c'est une grandeur de la donnée. Le déplacement a été fait quand un second label (le
swing, mort en C6 le 2026-09-25) en a eu besoin : tant que l'ATR vivait chez l'un des deux, l'autre
dépendait d'un module condamné à mourir, et la suppression du perdant aurait cassé le gagnant. Le
swing est parti ; l'ATR reste ici, à sa place.

**Déplacement, pas duplication.** La fonction est celle qui a été prouvée en phase 2, ligne pour
ligne ; seule son adresse change. Sa preuve de causalité l'a suivie (`sanity_check_atr`), et la
mutation « ATR centré » aussi.

Lancer :  aucun main — ce module n'est qu'une fonction et une constante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Période de l'ATR (C4).
ATR_PERIODE: int = 14


def atr_causal(df: pd.DataFrame, periode: int = ATR_PERIODE) -> np.ndarray:
    """
    ATR de Wilder au sens du True Range, **figé à t** : la valeur à l'indice `i` n'utilise que
    des barres d'indice <= `i`.

    Le True Range inclut les trous de prix (`|High - Close_precedent|`), conformément à la
    définition d'origine. Mesuré sur le dev : les inclure ne change l'ATR médian que de
    **+0,7 %** — la question a été tranchée par la mesure, pas par principe.

    Moyenne mobile simple sur `periode` barres : les **`periode` premières valeurs sont NaN** —
    pas `periode - 1`. La première barre n'a pas de clôture précédente, donc son True Range est
    NaN, et la première fenêtre qui le contient l'est aussi. ⚠️ Le docstring de la phase 2
    disait `periode - 1` depuis août ; personne ne l'avait vérifié parce que rien n'en dépendait
    (le label saute les NaN). Attrapé le 2026-09-23 par la preuve écrite pour le déménagement.
    """
    haut = np.asarray(df["High"], dtype=float)
    bas = np.asarray(df["Low"], dtype=float)
    cloture = np.asarray(df["Close"], dtype=float)
    cloture_prec = np.concatenate(([np.nan], cloture[:-1]))

    true_range = np.maximum(
        haut - bas, np.maximum(np.abs(haut - cloture_prec), np.abs(bas - cloture_prec))
    )
    return np.asarray(pd.Series(true_range).rolling(periode).mean())
