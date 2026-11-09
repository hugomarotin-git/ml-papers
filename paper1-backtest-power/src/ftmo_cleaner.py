"""
Nettoyage des données FTMO / MT5 — tâche 1.2.

Prend l'export brut de `ftmo_download_h1.mq5` (heure serveur FTMO) et produit un fichier
exploitable en heure de New York, ANNOTÉ mais jamais amputé.

╔════════════════════════════════════════════════════════════════════════════════════════╗
║ CE MODULE N'ENLÈVE JAMAIS UNE LIGNE.                                                    ║
║                                                                                          ║
║ Il annote, il ne juge pas. Aucun filtrage de session (décision B5 : on garde les 24 h,   ║
║ les heures de trading se décideront par la mesure, pas d'avance), aucune suppression des ║
║ barres suspectes (c'est la tâche 1.4 qui traite les barres D1 déguisées d'avant 2021,   ║
║ séparément, parce que c'est une décision de périmètre et pas du nettoyage).              ║
║                                                                                          ║
║ Le fichier brut n'est JAMAIS modifié (décision B6) : tout est rejouable depuis lui.      ║
╚════════════════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────────────
La conversion de fuseau : DEUX régimes, détectés par signature — jamais par une date en dur
────────────────────────────────────────────────────────────────────────────────────────
Le serveur FTMO suit le changement d'heure AMÉRICAIN : serveur et New York basculent aux mêmes
dates, l'écart est donc constant à l'intérieur d'un régime. Mais il y a eu DEUX régimes
(mesuré le 2026-08-09, corrige la décision B4 qui affirmait « -7 constant sur 16 ans ») :

- **jusqu'au 2021-03-27** : serveur à NY+6 (GMT+1/+2 suivant le DST US) ;
- **depuis le 2021-03-28** (passage à l'heure d'été EUROPÉENNE : le serveur a changé de
  base ce week-end-là) : serveur à NY+7 (GMT+2/+3 suivant le DST US).

Appliquer -7 partout mettait les 9 premières semaines de l'ère intraday (46 séances, 4,5 % du
dev) en avance d'une heure : sessions affichées 17:00 → 15:00 NY, ouverture du cash à 8 h 30.
Toute feature horaire aurait été bruitée sur cette fenêtre — toujours dans le même sens.

**La bascule se détecte par SIGNATURE, pas par une date en dur (corollaire B12).** L'horloge
extérieure est l'heure d'ouverture des sessions : une session (re)démarre toujours à 18:00 NY,
et le brut la stampe 00:00 serveur dans l'ancien régime, 01:00 dans le nouveau — sans une seule
exception sur toute la série. Le décalage détecté est écrit DANS la donnée, barre par barre
(colonne `decalage_serveur_ny`), et deux gardes lèvent plutôt que de deviner : une ouverture à
une heure non signée, ou plus d'une bascule (ou une bascule à l'envers).

`jour_ftmo`, lui, ne dépend d'aucun régime : la journée FTMO est bornée par MINUIT SERVEUR
(B11), donc `jour_ftmo` = la date en heure serveur, quelle que soit l'heure NY correspondante.
C'est pour ça que la correction du 2026-08-10 n'a déplacé AUCUNE frontière de journée.

La preuve empirique est dans `sanity_check_ftmo_cleaner.py` (tâche 1.3) : le repère extérieur
du cash NASDAQ (ouverture 9 h 30 NY), fenêtre par fenêtre — y compris sur T1 2021.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# Constantes de cadrage
# ──────────────────────────────────────────────────────────────────────────

#: Décalage heure serveur FTMO -> heure de New York du régime MODERNE (depuis le 2021-03-28).
SERVER_TO_NY_HOURS: int = -7

#: Décalage de l'ANCIEN régime (jusqu'au 2021-03-27 inclus) : le serveur était une heure plus
#: bas. Découvert le 2026-08-09 — corrige B4, qui affirmait un -7 constant sur toute la série.
SERVER_TO_NY_HOURS_ANCIEN: int = -6

#: La signature qui classe un régime : l'heure SERVEUR à laquelle une session (re)démarre.
#: Une session démarre toujours à 18:00 NY — stampée 00:00 serveur => serveur = NY+6 ;
#: stampée 01:00 => serveur = NY+7. Toute autre heure d'ouverture ne signe RIEN : elle
#: hérite du régime précédent (reprise en cours de session après un trou de feed).
DECALAGE_PAR_HEURE_OUVERTURE: dict[int, int] = {0: SERVER_TO_NY_HOURS_ANCIEN, 1: SERVER_TO_NY_HOURS}

#: Au-delà de cet écart entre deux barres, la reprise est une (ré)ouverture de session —
#: le rollover quotidien fait 120 min pile (une barre manquante), on coupe donc au-dessus.
SEUIL_OUVERTURE_MINUTES: float = 120.0

#: Durée nominale d'une barre H1.
BAR_MINUTES: int = 60

#: Format de date de l'export MQL5 : "2017.12.29 00:00:00".
MQL5_DATETIME_FORMAT: str = "%Y.%m.%d %H:%M:%S"

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
#: Les données vivent hors du dépôt, dans `trading/data/` (volumineuses et régénérables).
DATA_DIR: Path = PROJECT_DIR.parent / "data"

DEFAULT_RAW: Path = DATA_DIR / "ftmo_h1" / "RAW_ftmo_h1_nas100.csv"
DEFAULT_OUT: Path = DATA_DIR / "v2" / "nas100_h1_clean.csv"


# ──────────────────────────────────────────────────────────────────────────
# Étape 1 — lecture du brut
# ──────────────────────────────────────────────────────────────────────────


def load_raw(path: Path) -> pd.DataFrame:
    """
    Lit l'export MQL5 tel quel, sans rien transformer.

    Les horodatages restent en HEURE SERVEUR à ce stade : la conversion est une étape
    séparée et explicite, pour qu'on ne puisse jamais se demander dans quel fuseau est un
    DataFrame intermédiaire.
    """
    df = pd.read_csv(path, parse_dates=["DateTime"], date_format=MQL5_DATETIME_FORMAT)
    attendues = {"DateTime", "Open", "High", "Low", "Close", "TickVolume", "Spread"}
    manquantes = attendues - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes de l'export brut : {sorted(manquantes)}")
    return df.rename(columns={"TickVolume": "Volume"})


# ──────────────────────────────────────────────────────────────────────────
# Étape 2 — conversion de fuseau
# ──────────────────────────────────────────────────────────────────────────


def detecter_decalages_serveur_ny(df: pd.DataFrame) -> pd.Series:
    """
    Le décalage serveur -> NY de **chaque barre**, détecté par SIGNATURE (corollaire B12).

    Attend des horodatages en HEURE SERVEUR, triés. Le principe : toute reprise après une
    coupure de plus d'une barre ouvre un « segment », et l'heure serveur de cette reprise
    signe son régime — une session (re)démarre toujours à 18:00 NY, stampée 00:00 serveur
    dans l'ancien régime (NY+6), 01:00 dans le moderne (NY+7). Mesuré sur tout le brut :
    2 209 segments signés, une seule transition (-6 -> -7, au passage à l'heure d'été
    européenne du 2021-03-28), 6 reprises en cours de session qui héritent de leur régime.

    Deux gardes lèvent plutôt que de deviner :

    - le TOUT PREMIER segment doit être signé — sinon rien à hériter ;
    - plus d'une bascule, ou une bascule à l'envers (-7 -> -6), signale un export qui ne
      ressemble plus à ce qui a été vérifié : à regarder, pas à convertir en silence.
    """
    dt = pd.Series(df["DateTime"])
    if not dt.is_monotonic_increasing:
        raise ValueError("Horodatages non triés : la détection de régime lit des reprises.")

    ecart_min = dt.diff().dt.total_seconds() / 60.0
    ouverture = ecart_min.isna() | (ecart_min > float(BAR_MINUTES))
    heures = dt.dt.hour.where(ouverture)
    decalage_signe = heures.map(DECALAGE_PAR_HEURE_OUVERTURE)

    if pd.isna(decalage_signe.iloc[0]):
        raise ValueError(
            f"Le premier segment du fichier ouvre à {int(heures.iloc[0]):02d}:00 serveur — "
            f"une heure qui ne signe aucun régime connu ({sorted(DECALAGE_PAR_HEURE_OUVERTURE)}). "
            "Rien à hériter : export à examiner avant toute conversion."
        )

    signes = decalage_signe.dropna()
    bascules = signes[signes != signes.shift()].iloc[1:]
    if len(bascules) > 1 or bool((bascules == SERVER_TO_NY_HOURS_ANCIEN).any()):
        raise ValueError(
            f"{len(bascules)} bascule(s) de régime détectée(s) "
            f"({[int(b) for b in bascules]}) : le seul historique vérifié est UNE bascule "
            "-6 -> -7 (2021-03-28). Un export qui en montre d'autres a changé — à examiner, "
            "pas à convertir en silence."
        )

    # Les reprises non signées (trou de feed en cours de session) héritent du régime courant ;
    # les barres d'un segment héritent de leur ouverture.
    return decalage_signe.ffill().astype(int).rename("decalage_serveur_ny")


def convert_server_to_ny(df: pd.DataFrame, hours: int | None = None) -> pd.DataFrame:
    """
    Heure serveur FTMO -> heure de New York.

    Par défaut (`hours=None`), le décalage est **détecté par signature, barre par barre**
    (`detecter_decalages_serveur_ny`) et écrit dans la colonne `decalage_serveur_ny` — la
    donnée porte son propre régime, personne n'a à le deviner en aval.

    `hours` force un décalage constant, dans un seul but : permettre à la preuve (tâche 1.3)
    de montrer qu'un décalage constant unique est incompatible avec le repère du cash NASDAQ
    sur au moins une fenêtre. En usage normal, ne jamais s'en servir.
    """
    out = df.sort_values("DateTime").reset_index(drop=True)
    if hours is not None:
        out["DateTime"] = out["DateTime"] + pd.Timedelta(hours=hours)
        return out
    decalages = detecter_decalages_serveur_ny(out)
    out["decalage_serveur_ny"] = decalages
    out["DateTime"] = out["DateTime"] + pd.to_timedelta(decalages, unit="h")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Étape 3 — mesure et qualification des discontinuités
# ──────────────────────────────────────────────────────────────────────────


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute `minutes_since_prev`, `gap_flag` et `gap_kind`.

    `gap_kind` distingue trois situations que l'ancien pipeline confondait sous un seul
    booléen — il marquait « écart != 60 min », ce qui mettait les week-ends normaux et les
    vrais trous dans le même sac :

    - ``contigu``  : la barre suit la précédente (60 min). Le cas normal.
    - ``weekend``  : coupure de fin de semaine (barre précédente un vendredi, barre courante
      un dimanche ou un lundi). Attendue, ce n'est PAS une anomalie.
    - ``trou``     : tout le reste — férié, panne de feed, ou zone où le serveur ne stockait
      pas d'intraday. C'est **la seule catégorie qui mérite un regard**.

    Qualifier par les jours de la semaine plutôt que par un seuil en minutes rend la règle
    robuste : la durée réelle d'un week-end varie (fériés accolés, heure d'été), un seuil
    numérique se serait trompé aux bords.
    """
    out = df.copy()
    delta = out["DateTime"].diff()
    out["minutes_since_prev"] = delta.dt.total_seconds() / 60.0

    jour_prec = out["DateTime"].shift(1).dt.dayofweek
    jour_cour = out["DateTime"].dt.dayofweek

    contigu = out["minutes_since_prev"] == BAR_MINUTES
    weekend = (jour_prec == 4) & jour_cour.isin([6, 0]) & ~contigu

    out["gap_flag"] = ~contigu
    out["gap_kind"] = "trou"
    out.loc[contigu, "gap_kind"] = "contigu"
    out.loc[weekend, "gap_kind"] = "weekend"
    out.loc[out["minutes_since_prev"].isna(), "gap_kind"] = "premiere_barre"
    return out


# ──────────────────────────────────────────────────────────────────────────
# Étape 4 — spread manquant
# ──────────────────────────────────────────────────────────────────────────


def flag_missing_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les barres où `Spread == 0` comme **donnée manquante**, pas comme spread nul.

    Un spread nul n'existe pas sur un CFD : c'est un trou dans l'export. Mesuré sur le
    NASDAQ (décision B10), ça concerne ~12,8 % des barres.

    Sans ce drapeau, le risque est concret et silencieux : au moment de brancher les coûts,
    on appliquerait un coût nul sur un huitième de l'échantillon, et on sous-estimerait les
    coûts précisément là où la donnée manque.

    Ce module se contente de SIGNALER. Comment combler (report, médiane, exclusion) est une
    décision de la couche coûts, pas du nettoyage.
    """
    out = df.copy()
    out["spread_missing"] = out["Spread"] <= 0
    return out


# ──────────────────────────────────────────────────────────────────────────
# Étape 5 — la journée FTMO (≠ la journée calendaire)
# ──────────────────────────────────────────────────────────────────────────


def add_ftmo_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute `jour_ftmo` — **la seule notion de « journée » qui ait un sens sur ce projet.**

    Une session FTMO ne coïncide PAS avec une journée calendaire de New York : elle court de
    **18:00 NY à 16:00 NY le lendemain**, avec une coupure d'une heure au rollover. Mesuré sur
    les données : 1 046 des 1 084 trous d'une heure tombent à 18:00 NY, c'est-à-dire juste
    après le rollover — ce n'est pas une anomalie, c'est la respiration quotidienne du serveur.

    La frontière est **MINUIT SERVEUR** — l'instant où :

    - la session s'interrompt puis repart ;
    - **le plancher de perte journalier de FTMO se remet à zéro.**

    `jour_ftmo` est donc la **date en heure serveur**, et c'est ce qui la rend invariante au
    régime de conversion : minuit serveur = 17:00 NY dans le régime moderne (NY+7), 18:00 NY
    dans l'ancien (NY+6, jusqu'au 2021-03-27) — mais la date serveur, elle, ne bouge pas.
    C'est pour ça que la correction B4 du 2026-08-10 n'a déplacé aucune frontière de journée.

    Grouper par date calendaire NY couperait la session en plein milieu et se désalignerait du
    plancher journalier. Cette colonne donne le bon regroupement à tout ce qui suit : le
    label (« l'horizon tient-il dans la journée ? ») comme le juge (« quand le plancher
    journalier se recalcule-t-il ? »).
    """
    out = df.copy()
    if "decalage_serveur_ny" in out.columns:
        decalages = pd.Series(out["decalage_serveur_ny"])
        heure_serveur = out["DateTime"] - pd.to_timedelta(decalages, unit="h")
        out["jour_ftmo"] = heure_serveur.dt.date
    else:
        # Mode constant (preuves) : sans colonne de décalage, on remonte au serveur par -7.
        out["jour_ftmo"] = (out["DateTime"] + pd.Timedelta(hours=-SERVER_TO_NY_HOURS)).dt.date
    return out


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────


def clean_ftmo_h1(
    raw_path: Path = DEFAULT_RAW, out_path: Path | None = DEFAULT_OUT
) -> pd.DataFrame:
    """
    Chaîne complète : lecture -> conversion NY -> discontinuités -> spread manquant -> journée FTMO.

    Vérifie par assertion qu'aucune ligne n'a disparu en route : ce module annote, il ne
    filtre jamais. Une régression sur ce point serait silencieuse sans cette garde.
    """
    brut = load_raw(raw_path)
    n_entree = len(brut)

    df = convert_server_to_ny(brut)
    df = add_gap_columns(df)
    df = flag_missing_spread(df)
    df = add_ftmo_trading_day(df)

    if len(df) != n_entree:
        raise AssertionError(
            f"Le nettoyage a changé le nombre de lignes : {n_entree} -> {len(df)}. "
            "Ce module ne doit JAMAIS filtrer (décision B5)."
        )

    # Doublons : on REFUSE, on ne filtre pas. La nuance compte — retirer une ligne violerait
    # B5 et masquerait le problème, alors qu'un export contenant deux fois le même horodatage
    # est un export cassé, qu'il faut regarder avant d'aller plus loin.
    #
    # Sans cette garde, un doublon produit un écart de 0 minute classé en « trou », et tout le
    # reste du pipeline continue comme si de rien n'était.
    n_doublons = int(df["DateTime"].duplicated().sum())
    if n_doublons:
        exemples = sorted(set(df.loc[df["DateTime"].duplicated(keep=False), "DateTime"]))[:3]
        raise ValueError(
            f"{n_doublons} horodatage(s) en double dans l'export, par exemple {exemples}. "
            "Export cassé : vérifier la source avant de poursuivre (rien n'est filtré ici)."
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def load_clean(path: Path) -> pd.DataFrame:
    """
    Relit un fichier nettoyé en **restaurant les types**.

    Nécessaire parce qu'un CSV ne transporte pas les types : `jour_ftmo` est écrit comme une
    date mais relu comme du texte, ce qui casse silencieusement toute comparaison de dates.
    Passer par cette fonction plutôt que par `pd.read_csv` évite d'avoir à y penser à chaque
    fois — et évite l'erreur qu'on ne voit pas, celle où la comparaison renvoie simplement
    False au lieu de lever.
    """
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df["jour_ftmo"] = pd.to_datetime(df["jour_ftmo"]).dt.date
    return df


def resume(df: pd.DataFrame) -> str:
    """Résumé lisible, imprimé après un run — pour voir tout de suite si quelque chose cloche."""
    kinds = df["gap_kind"].value_counts()
    return "\n".join(
        [
            f"  barres            : {len(df):,}",
            f"  période (NY)      : {df['DateTime'].min()} -> {df['DateTime'].max()}",
            f"  jours calendaires : {df['DateTime'].dt.date.nunique():,}",
            f"  journées FTMO     : {df['jour_ftmo'].nunique():,}  (18:00 NY -> 16:00 NY le lendemain)",
            "  discontinuités    : " + ", ".join(f"{k}={v:,}" for k, v in kinds.items()),
            f"  spread manquant   : {df['spread_missing'].sum():,} "
            f"({100 * df['spread_missing'].mean():.1f} %)",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Nettoyage FTMO H1 (heure serveur -> heure NY).")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="export MQL5 brut")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fichier nettoyé à écrire")
    args = ap.parse_args()

    df = clean_ftmo_h1(args.raw, args.out)
    print(f"\nNettoyage FTMO H1 — {args.raw.name} -> {args.out}")
    print(resume(df))
    print()


if __name__ == "__main__":
    main()
