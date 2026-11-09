"""
Le découpage dev / hold-out — tâche 1.7.

**Une seule définition, importée partout.** Toute fonction qui a besoin de savoir « est-ce du
dev ? » passe par ici. C'est le point qui empêche la dérive : dans le build précédent, la même
constante était recopiée dans cinq fichiers, et il a suffi d'en oublier un pour que deux modules
travaillent sur des périodes différentes sans que rien ne le signale.

────────────────────────────────────────────────────────────────────────────────────────
Le découpage se fait sur la JOURNÉE FTMO, jamais sur la date de la barre
────────────────────────────────────────────────────────────────────────────────────────
Une séance FTMO court de 18:00 NY à 16:00 NY le lendemain : elle **traverse minuit**. Découper
sur la date calendaire d'une barre couperait donc une séance en deux, et la moitié du soir
tomberait du mauvais côté du sceau.

Vérifié sur les données : il n'existe aujourd'hui aucune barre le 31 décembre après 17 h, sur
**aucune** des cinq années — la frontière est donc propre. Mais elle l'est **par chance de
calendrier**, pas par construction. Découper sur `jour_ftmo` la rend propre par construction, ce
qui est la seule forme sur laquelle on peut compter.

────────────────────────────────────────────────────────────────────────────────────────
Le hold-out
────────────────────────────────────────────────────────────────────────────────────────
Il est **scellé** : une copie figée et son empreinte vivent dans `data/_HOLDOUT_SCELLE/`. Ce
module ne sert pas à y accéder, mais à garantir qu'on ne l'atteint pas par accident depuis le
travail courant — d'où `assert_dev_seulement`, à poser en tête de toute étape qui apprend
quelque chose.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

#: Première journée FTMO réellement intraday (décision B3 — détectée par signature, pas supposée).
DEV_DEBUT: date = date(2021, 1, 22)
#: Dernière journée FTMO du dev.
DEV_FIN: date = date(2024, 12, 31)
#: Première journée FTMO du hold-out. Scellé le 2026-08-04, à ne consommer qu'une fois.
HOLDOUT_DEBUT: date = date(2025, 1, 1)

#: Nom de la colonne qui porte la journée FTMO, produite par `ftmo_cleaner.add_ftmo_trading_day`.
COLONNE_JOUR: str = "jour_ftmo"


def _jours(df: pd.DataFrame) -> np.ndarray:
    """Les journées FTMO du panel, en `datetime.date`, avec un message clair si absentes."""
    if COLONNE_JOUR not in df.columns:
        raise ValueError(
            f"Colonne « {COLONNE_JOUR} » absente. Le découpage se fait sur la journée FTMO, "
            "jamais sur la date d'une barre — passer les données par `ftmo_cleaner.clean_ftmo_h1` "
            "(ou `load_clean`) d'abord."
        )
    return np.asarray(pd.to_datetime(pd.Series(df[COLONNE_JOUR])).dt.date)


def masque_dev(df: pd.DataFrame) -> np.ndarray:
    """Vrai pour les barres appartenant à une journée FTMO du dev."""
    j = _jours(df)
    return (j >= DEV_DEBUT) & (j <= DEV_FIN)


def masque_holdout(df: pd.DataFrame) -> np.ndarray:
    """Vrai pour les barres appartenant à une journée FTMO du hold-out."""
    return _jours(df) >= HOLDOUT_DEBUT


def restreindre_au_dev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ne garde que le dev. **À appeler au plus tôt** dans toute chaîne qui apprend quelque chose.

    Restreindre tôt vaut mieux que vérifier tard : une étape intermédiaire qui aurait vu le
    hold-out l'aurait déjà influencée, même si une garde le détecte ensuite.
    """
    return pd.DataFrame(df[masque_dev(df)]).reset_index(drop=True)


def assert_dev_seulement(df: pd.DataFrame, etape: str = "cette étape") -> None:
    """
    Refuse un panel qui contient du hold-out.

    À poser en tête de tout ce qui apprend, mesure ou sélectionne. Le coût est d'une ligne ;
    ce qu'elle empêche est irréversible — un hold-out regardé une fois de trop n'est plus un
    hold-out, et on n'a pas de rechange.
    """
    fuite = masque_holdout(df)
    if bool(fuite.any()):
        j = _jours(df)[fuite]
        raise AssertionError(
            f"{int(fuite.sum())} barre(s) de HOLD-OUT dans {etape} "
            f"(du {min(j)} au {max(j)}). Le hold-out ne se touche qu'une fois, à la toute fin. "
            "Restreindre avec `restreindre_au_dev` avant cette étape."
        )


def assert_pas_de_bougie_journaliere(df: pd.DataFrame) -> None:
    """
    Refuse un panel qui commence avant le premier jour réellement intraday.

    Doublon volontaire avec `intraday_filter` : cette garde-ci attrape le cas où quelqu'un
    court-circuiterait le filtre. Deux verrous indépendants sur la même porte, parce que ces
    barres corrompent l'ATR sans rien casser d'autre.
    """
    j = _jours(df)
    trop_tot = j < DEV_DEBUT
    if bool(trop_tot.any()):
        raise AssertionError(
            f"{int(trop_tot.sum())} barre(s) antérieure(s) au {DEV_DEBUT} (jusqu'au {max(j[trop_tot])}) : "
            "ce sont des bougies journalières déguisées en H1. Passer par `intraday_filter`."
        )


def resume_decoupage(df: pd.DataFrame) -> str:
    """Ce que contient un panel, tranche par tranche — à imprimer avant tout run sérieux."""
    j = _jours(df)
    dev, hold = masque_dev(df), masque_holdout(df)
    avant = j < DEV_DEBUT
    lignes = [
        f"  avant le {DEV_DEBUT} : {int(avant.sum()):>7,} barres  (bougies journalières, à écarter)",
        f"  dev   {DEV_DEBUT} → {DEV_FIN} : {int(dev.sum()):>7,} barres, "
        f"{len(set(j[dev])):,} journées FTMO",
        f"  hold-out ≥ {HOLDOUT_DEBUT}         : {int(hold.sum()):>7,} barres, "
        f"{len(set(j[hold])):,} journées FTMO  [SCELLÉ]",
    ]
    return "\n".join(lignes)
