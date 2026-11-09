"""
Écarter les barres journalières déguisées en H1 — tâche 1.4.

C'est le module qui **retire** des lignes, contrairement au nettoyeur qui n'annote jamais.
Les deux responsabilités sont séparées volontairement : nettoyer, c'est rendre lisible ;
filtrer, c'est décider d'un périmètre.

────────────────────────────────────────────────────────────────────────────────────────
Le problème
────────────────────────────────────────────────────────────────────────────────────────
Le serveur FTMO n'a pas stocké d'intraday sur US100 dès le début. Sur toute la première
partie de l'historique, il ne renvoie qu'**une seule barre par jour** — une bougie
journalière, servie dans un fichier H1 sans que rien ne la distingue. Amplitude d'une
journée entière, horodatée à une heure fixe.

Les laisser passer aurait un effet qu'on ne verrait pas : elles corrompent l'**ATR**. Mesuré
sur la transition réelle, l'ATR(14) monte à **200 points** juste après, contre ~33 points en
régime normal — six fois trop large. Toutes les barrières calculées en multiple d'ATR sur
cette zone seraient absurdes, et rien ne planterait pour le signaler.

────────────────────────────────────────────────────────────────────────────────────────
Pourquoi une DÉTECTION et pas un filtre sur une date
────────────────────────────────────────────────────────────────────────────────────────
La tentation était d'écrire `DateTime >= 2021-01-03`. **Cette date est fausse**, et c'est
précisément la mesure qui l'a montré : les barres journalières continuent jusqu'au
**2021-01-21**, la première journée pleinement intraday étant le **2021-01-22**. Un filtre
sur la date d'origine aurait laissé passer 13 barres journalières en plein dans le dev.

Une date en dur se contente d'être crue. Une détection par signature se vérifie, et elle
**refuse de se laisser oublier** : si la nature des données change après un
rafraîchissement, `assert_signature_journaliere` échoue au lieu de laisser filer.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ftmo_cleaner import DATA_DIR, load_clean

#: En deçà de ce nombre de barres, une journée FTMO n'est pas une vraie séance intraday.
#: Mesuré : les séances réelles ont **au minimum 11 barres** (23 en régime normal), les
#: journalières déguisées en ont **1 ou 2**. Le seuil est posé loin des deux bords.
MIN_BARRES_SEANCE: int = 6

#: Une barre journalière couvre 24 h de mouvement : son amplitude doit dépasser largement
#: celle d'une barre horaire. Garde-fou de la détection, pas critère principal.
RATIO_AMPLITUDE_MIN: float = 2.0

DEFAULT_IN: Path = DATA_DIR / "v2" / "nas100_h1_clean.csv"
DEFAULT_OUT: Path = DATA_DIR / "v2" / "nas100_h1_intraday.csv"


@dataclass(frozen=True)
class RapportFiltre:
    """Ce qui a été retiré, et sur quelle base — pour que la décision reste auditable."""

    n_avant: int
    n_apres: int
    n_barres_retirees: int
    n_seances_retirees: int
    premiere_seance_intraday: object
    heure_dominante_retiree: int
    part_heure_dominante: float
    amplitude_mediane_retiree: float
    amplitude_mediane_gardee: float

    def __str__(self) -> str:
        return "\n".join(
            [
                f"  barres            : {self.n_avant:,} -> {self.n_apres:,} "
                f"({self.n_barres_retirees:,} retirées)",
                f"  séances retirées  : {self.n_seances_retirees:,}",
                f"  1re séance gardée : {self.premiere_seance_intraday}",
                f"  signature         : {self.part_heure_dominante:.0%} des barres retirées à "
                f"{self.heure_dominante_retiree} h, amplitude médiane "
                f"{self.amplitude_mediane_retiree:.0f} pts contre "
                f"{self.amplitude_mediane_gardee:.0f} pts pour les barres gardées",
            ]
        )


def compter_barres_par_seance(df: pd.DataFrame) -> pd.Series:
    """
    Nombre de barres de chaque journée FTMO, réaligné sur l'index des barres.

    Vérifie la présence des colonnes attendues et le dit clairement : sans ça, un panel qui
    n'est pas passé par le nettoyeur produit un `KeyError` nu, qui n'indique ni ce qui manque
    ni quoi faire.
    """
    manquantes = {"jour_ftmo", "DateTime"} - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Colonnes absentes : {sorted(manquantes)}. Ce module attend un panel produit par "
            "`ftmo_cleaner.clean_ftmo_h1` (ou relu par `load_clean`)."
        )
    tailles = df.groupby("jour_ftmo", observed=True)["DateTime"].transform("size")
    return pd.Series(tailles, index=df.index)


def detecter_barres_journalieres(
    df: pd.DataFrame, min_barres: int = MIN_BARRES_SEANCE
) -> pd.Series:
    """
    Masque des barres appartenant à une séance trop courte pour être de l'intraday.

    Le critère est le **nombre de barres dans la journée FTMO**, pas la date. C'est la
    signature directe du problème : une bougie journalière est seule dans sa journée.
    """
    return compter_barres_par_seance(df) < min_barres


def assert_signature_journaliere(
    df: pd.DataFrame, masque: pd.Series
) -> tuple[int, float, float, float]:
    """
    Vérifie que ce qu'on s'apprête à retirer **ressemble vraiment** à des bougies journalières.

    Deux marqueurs indépendants du critère de comptage, donc capables de le contredire :

    1. **Une heure dominante.** Les bougies journalières sont toutes horodatées pareil.
       Si les barres retirées étaient dispersées sur la journée, ce ne seraient pas des
       bougies journalières mais des séances réelles tronquées — et on serait en train de
       supprimer de la vraie donnée.
    2. **Une amplitude de journée.** Elles couvrent 24 h de mouvement, donc bien plus qu'une
       barre horaire.

    Lève une `AssertionError` si l'un des deux ne tient pas. C'est le point qui empêche ce
    filtre de devenir silencieusement destructeur si les données changent.
    """
    retirees = df[masque]
    gardees = df[~masque]
    if len(retirees) == 0:
        return (-1, 0.0, 0.0, float(np.median(np.asarray(gardees["High"] - gardees["Low"]))))

    heures: np.ndarray = np.asarray(pd.Series(retirees["DateTime"]).dt.hour, dtype=int)
    valeurs, comptes = np.unique(heures, return_counts=True)
    heure_dominante = int(valeurs[int(np.argmax(comptes))])
    part = float(comptes.max() / comptes.sum())

    ampl_retirees = float(np.median(np.asarray(retirees["High"] - retirees["Low"], dtype=float)))
    ampl_gardees = float(np.median(np.asarray(gardees["High"] - gardees["Low"], dtype=float)))

    if part < 0.90:
        raise AssertionError(
            f"Les barres à retirer ne sont PAS concentrées sur une heure "
            f"({part:.0%} à {heure_dominante} h) — ce ne sont pas des bougies journalières. "
            "Refus de filtrer : vérifier les données avant d'aller plus loin."
        )
    # ⚠️ **Une amplitude non finie DÉSACTIVERAIT ce garde-fou en silence** — corrigé le
    # 2026-08-18 (piège n°8, audit du 18/08). `nan > 0` vaut False : la condition entière
    # serait fausse, donc l'assertion suivante ne lèverait jamais. Or c'est précisément ce
    # garde-fou qui empêche le filtre de devenir destructeur si les données changent — il se
    # serait tu au moment exact où il aurait dû parler. Un OHLC troué le produirait.
    if not (math.isfinite(ampl_retirees) and math.isfinite(ampl_gardees)):
        raise AssertionError(
            f"Amplitude non finie (retirées {ampl_retirees!r}, gardées {ampl_gardees!r}) : des "
            "OHLC manquants rendent la comparaison d'amplitude impossible, et un `NaN` la ferait "
            "passer sans lever. Refus de filtrer : vérifier les données avant d'aller plus loin."
        )
    if ampl_gardees > 0 and ampl_retirees / ampl_gardees < RATIO_AMPLITUDE_MIN:
        raise AssertionError(
            f"Les barres à retirer n'ont pas une amplitude de journée "
            f"({ampl_retirees:.0f} pts contre {ampl_gardees:.0f} pts gardés, "
            f"ratio {ampl_retirees / ampl_gardees:.1f} < {RATIO_AMPLITUDE_MIN}). Refus de filtrer."
        )
    return heure_dominante, part, ampl_retirees, ampl_gardees


def filtrer_intraday(
    df: pd.DataFrame, min_barres: int = MIN_BARRES_SEANCE
) -> tuple[pd.DataFrame, RapportFiltre]:
    """
    Retire les séances non intraday et rend un rapport de ce qui a été fait.

    L'ordre compte : on **vérifie la signature avant de couper**. Un filtre qui coupe d'abord
    et regarde ensuite ne protège de rien.
    """
    masque = detecter_barres_journalieres(df, min_barres)
    heure, part, ampl_r, ampl_g = assert_signature_journaliere(df, masque)

    garde = pd.DataFrame(df[~masque]).reset_index(drop=True)
    rapport = RapportFiltre(
        n_avant=len(df),
        n_apres=len(garde),
        n_barres_retirees=int(masque.sum()),
        n_seances_retirees=int(df.loc[masque, "jour_ftmo"].nunique()),
        premiere_seance_intraday=(pd.Series(garde["jour_ftmo"]).iloc[0] if len(garde) else None),
        heure_dominante_retiree=heure,
        part_heure_dominante=part,
        amplitude_mediane_retiree=ampl_r,
        amplitude_mediane_gardee=ampl_g,
    )
    return garde, rapport


def assert_aucune_barre_journaliere(df: pd.DataFrame, min_barres: int = MIN_BARRES_SEANCE) -> None:
    """
    Garde à poser en tête de toute étape qui suit (label, features, juge).

    Coûte une ligne, et rend impossible de calculer un ATR sur des bougies journalières sans
    s'en apercevoir.
    """
    masque = detecter_barres_journalieres(df, min_barres)
    if bool(masque.any()):
        seances = sorted(set(df.loc[masque, "jour_ftmo"]))
        raise AssertionError(
            f"{int(masque.sum())} barre(s) appartenant à une séance de moins de {min_barres} "
            f"barres, sur {len(seances)} séance(s), de {seances[0]} à {seances[-1]}.\n"
            "Deux causes possibles, à ne pas confondre :\n"
            "  - des bougies journalières déguisées -> passer par `filtrer_intraday` ;\n"
            "  - un DÉCOUPAGE qui a tronqué une séance en plein milieu (un `head(n)` ou un "
            "`iloc[:n]` sur un nombre arbitraire de lignes) -> découper sur des journées FTMO "
            "entières, pas sur un compte de lignes."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Écarte les barres journalières déguisées en H1.")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = load_clean(args.input)
    garde, rapport = filtrer_intraday(df)
    assert_aucune_barre_journaliere(garde)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    garde.to_csv(args.out, index=False)
    print(f"\nFiltre intraday — {args.input.name} -> {args.out.name}")
    print(rapport)
    print()


if __name__ == "__main__":
    main()
