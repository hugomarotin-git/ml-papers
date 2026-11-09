"""
Preuve de l'ATR causal — déplacée avec sa fonction le 2026-09-23 (P12).

`check_1` est celle de la phase 2, mot pour mot : elle vivait dans `sanity_check_label` et a suivi
`atr_causal` dans son nouveau module. Les deux autres épinglent ce que le docstring affirme —
le True Range inclut les trous de prix, et les `periode - 1` premières valeurs sont NaN.
"""

from __future__ import annotations

import sys
from typing import NoReturn

import numpy as np
import pandas as pd

from atr import ATR_PERIODE, atr_causal
from ftmo_cleaner import DATA_DIR, load_clean
from periodes import restreindre_au_dev


def _echec(msg: str) -> NoReturn:
    raise AssertionError(f"ÉCHEC — {msg}")


def check_1_aucun_look_ahead(df: pd.DataFrame) -> None:
    """
    **La vérification centrale.** L'ATR à l'instant `t` ne doit dépendre d'aucune barre future.

    Testée pour de vrai : on **tronque** la série juste après `t` et on recalcule. Si la valeur
    change, c'est qu'une barre postérieure y participait. Un simple examen du code ne prouverait
    rien — c'est le genre de fuite qui se glisse dans un décalage d'indice.
    """
    complet = atr_causal(df)
    for t in (500, 5_000, 12_000, 20_000):
        tronque = atr_causal(pd.DataFrame(df.iloc[: t + 1]))
        if not np.isclose(complet[t], tronque[t], equal_nan=True):
            _echec(
                f"ATR à t={t} : {complet[t]} sur la série complète, {tronque[t]} tronquée -> look-ahead"
            )
    print("  1. aucun look-ahead : l'ATR est identique sur série tronquée, à 4 instants testés  ✓")


def check_2_les_premieres_valeurs_sont_NaN_et_le_reste_positif(df: pd.DataFrame) -> None:
    """
    Ce que le docstring promet : `periode` NaN en tête (TR[0] n'a pas de clôture précédente),
    puis rien que du fini et positif. ⚠️ En l'écrivant, le docstring de phase 2 disait
    `periode - 1` : faux d'un cran depuis août, sans conséquence, et jamais vérifié.
    """
    a = atr_causal(df)
    if not bool(np.isnan(a[:ATR_PERIODE]).all()):
        _echec(f"les {ATR_PERIODE} premières valeurs devraient être NaN")
    if not np.isfinite(a[ATR_PERIODE]):
        _echec(f"la valeur {ATR_PERIODE} devrait être la première finie")
    reste = a[ATR_PERIODE:]
    if len(reste) == 0:
        _echec("anti-vacuité : aucune valeur après l'échauffement")
    if not bool(np.isfinite(reste).all()) or not bool((reste > 0).all()):
        _echec("un ATR non fini ou non positif après l'échauffement")
    print(
        f"  2. {ATR_PERIODE} NaN en tête (pas {ATR_PERIODE - 1}), puis {len(reste):,} valeurs finies et > 0  ✓"
    )


def check_3_le_true_range_INCLUT_les_trous_de_prix() -> None:
    """
    ⚠️ Fixture hostile : des bougies minuscules (fourchette 1) et UN saut de 10 entre deux
    clôtures. Un « range » qui ne lirait que `High - Low` ne verrait jamais le saut — et
    sous-estimerait l'ATR précisément le jour où le marché a bougé.

    Le fixture est assez long pour qu'une fenêtre entière tienne AVANT le saut : c'est
    l'anti-vacuité (ATR = 1,0 exactement), sans laquelle « le saut est compté » ne se
    distinguerait pas d'un ATR faux partout.
    """
    n, saut = 40, 25
    prix = np.where(np.arange(n) < saut, 100.0, 110.0)
    df = pd.DataFrame({"High": prix + 0.5, "Low": prix - 0.5, "Close": prix})
    a = atr_causal(df, periode=ATR_PERIODE)
    # La fenêtre qui se termine sur le saut : 13 fois 1 et une fois 10,5.
    attendu = (13 * 1.0 + 10.5) / 14
    if not np.isclose(a[saut], attendu):
        _echec(f"ATR[{saut}] = {a[saut]:.4f}, attendu {attendu:.4f} : le saut n'est pas compté")
    if not np.isclose(a[saut - 1], 1.0):
        _echec(f"avant le saut, l'ATR devrait valoir 1,0 (fourchette pure), vu {a[saut - 1]}")
    print(
        f"  3. le True Range inclut le trou de prix : ATR[{saut}] = {a[saut]:.4f}, 1,0 juste avant  ✓"
    )


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DE L'ATR CAUSAL — déplacé le 2026-09-23 (P12)")
    print("=" * 78 + "\n")
    df = restreindre_au_dev(load_clean(DATA_DIR / "v2" / "nas100_h1_intraday.csv"))
    check_1_aucun_look_ahead(df)
    check_2_les_premieres_valeurs_sont_NaN_et_le_reste_positif(df)
    check_3_le_true_range_INCLUT_les_trous_de_prix()
    print("\nLES 3 PREUVES PASSENT.\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
