"""
Preuve du nettoyeur FTMO H1 — tâche 1.3.

Quinze vérifications. Le nettoyeur n'est « fait » que si elles passent toutes.

Les plus importantes sont la n°4 — **la preuve empirique du décalage dominant de -7 h** — et la
n°14 — **la preuve des DEUX régimes** (correction B4 du 2026-08-10 : -6 h jusqu'au 2021-03-27).
Tout le reste du projet en dépend — un fuseau faux décale les sessions, donc les labels, donc
le juge, et rien ne planterait pour le signaler.

Lancer :  .venv/bin/python src/sanity_check_ftmo_cleaner.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ftmo_cleaner import (
    BAR_MINUTES,
    DEFAULT_RAW,
    SERVER_TO_NY_HOURS,
    SERVER_TO_NY_HOURS_ANCIEN,
    add_ftmo_trading_day,
    add_gap_columns,
    clean_ftmo_h1,
    convert_server_to_ny,
    detecter_decalages_serveur_ny,
    load_clean,
    load_raw,
    resume,
)

#: Ouverture du cash NASDAQ : 09h30 NY. La barre qui la contient est celle de 09h.
HEURE_OUVERTURE_CASH = 9
#: Clôture du cash NASDAQ : 16h00 NY. L'activité s'effondre à partir de cette barre.
HEURE_CLOTURE_CASH = 16
#: Le saut d'amplitude attendu entre 8 h et 9 h (ouverture du cash). Mesuré ~1,95x.
SAUT_OUVERTURE_MIN = 1.5
#: L'effondrement d'activité attendu entre 15 h et 16 h (clôture du cash). Mesuré ~0,30x.
CHUTE_CLOTURE_MAX = 0.5
#: Mois franchement en heure d'été / d'hiver aux États-Unis (mars et novembre exclus : bascule).
MOIS_ETE = [4, 5, 6, 7, 8, 9, 10]
MOIS_HIVER = [12, 1, 2]
#: Les semaines où les États-Unis et l'Europe ne sont PAS sur le même régime d'heure d'été.
#: Printemps : du 2ᵉ dimanche de mars (bascule US) au dernier dimanche de mars (bascule EU).
#: Automne : du dernier dimanche d'octobre (bascule EU) au 1ᵉʳ dimanche de novembre (bascule US).
#: Seule période capable de distinguer un serveur réglé sur le DST US d'un serveur réglé sur le
#: DST européen. Mesuré : le saut reste sur la barre de 9 h -> c'est bien le DST américain.
FENETRES_DESALIGNEMENT_DST = [
    ("2021-03-14", "2021-03-28"),
    ("2021-10-31", "2021-11-07"),
    ("2022-03-13", "2022-03-27"),
    ("2022-10-30", "2022-11-06"),
    ("2023-03-12", "2023-03-26"),
    ("2023-10-29", "2023-11-05"),
    ("2024-03-10", "2024-03-31"),
    ("2024-10-27", "2024-11-03"),
    ("2025-03-09", "2025-03-30"),
    ("2025-10-26", "2025-11-02"),
    ("2026-03-08", "2026-03-29"),
]
#: Début de l'historique vraiment intraday (avant : des barres D1 déguisées, décision B3).
DEBUT_INTRADAY = "2021-01-04"


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """Vérifie qu'un garde-fou lève — **et pour la bonne raison** (piège n°4)."""
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(f"{quoi} : levée pour une AUTRE raison — « {str(e)[:90]} »")
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def _profil_horaire(dates: pd.Series, valeurs: pd.Series) -> np.ndarray:
    """Médiane de `valeurs` pour chacune des 24 heures. Renvoie un tableau indexé par l'heure."""
    heures: np.ndarray = np.asarray(dates.dt.hour, dtype=int)
    v: np.ndarray = np.asarray(valeurs, dtype=float)
    return np.array(
        [np.median(v[heures == h]) if (heures == h).any() else np.nan for h in range(24)]
    )


def _marqueurs_cash(df: pd.DataFrame) -> tuple[float, float]:
    """
    Les deux rapports qui signent la session cash : saut à l'ouverture, chute à la clôture.

    Des RAPPORTS plutôt que des niveaux : ça les rend insensibles au régime de volatilité, donc
    comparables entre 2021 et 2026, et entre l'été et l'hiver.
    """
    dates = pd.Series(df["DateTime"])
    amplitude = _profil_horaire(dates, pd.Series(df["High"] - df["Low"]))
    volume = _profil_horaire(dates, pd.Series(df["Volume"]))
    saut = float(amplitude[HEURE_OUVERTURE_CASH] / amplitude[HEURE_OUVERTURE_CASH - 1])
    chute = float(volume[HEURE_CLOTURE_CASH] / volume[HEURE_CLOTURE_CASH - 1])
    return saut, chute


# ──────────────────────────────────────────────────────────────────────────


def check_1_aucune_ligne_perdue(brut: pd.DataFrame, propre: pd.DataFrame) -> None:
    """Le nettoyeur annote, il ne filtre jamais (décision B5)."""
    if len(brut) != len(propre):
        _echec(f"lignes perdues : {len(brut)} -> {len(propre)}")
    print(f"PASS — 1. aucune ligne perdue : {len(brut):,} en entrée, {len(propre):,} en sortie")


def check_2_horodatages_uniques_et_croissants(df: pd.DataFrame) -> None:
    """Un doublon d'horodatage fausserait tout comptage de barres et tout `searchsorted`."""
    n_dup = int(df["DateTime"].duplicated().sum())
    if n_dup:
        _echec(f"{n_dup} horodatage(s) en double")
    if not df["DateTime"].is_monotonic_increasing:
        _echec("les horodatages ne sont pas strictement croissants")
    print("PASS — 2. horodatages uniques et strictement croissants")


def check_3_coherence_ohlc(df: pd.DataFrame) -> None:
    """High doit majorer Open/Close/Low, Low doit les minorer. Sinon la donnée est corrompue."""
    haut = df[["Open", "Close", "Low"]].max(axis=1).to_numpy(dtype=float)
    bas = df[["Open", "Close", "High"]].min(axis=1).to_numpy(dtype=float)
    mauvais_h = int((df["High"].to_numpy(dtype=float) < haut - 1e-9).sum())
    mauvais_b = int((df["Low"].to_numpy(dtype=float) > bas + 1e-9).sum())
    if mauvais_h or mauvais_b:
        _echec(f"OHLC incohérent : {mauvais_h} High trop bas, {mauvais_b} Low trop haut")
    print("PASS — 3. cohérence OHLC (High majore, Low minore) sur toutes les barres")


def check_4_preuve_du_decalage(brut: pd.DataFrame) -> None:
    """
    LA preuve du -7 h, et le cœur de ce fichier.

    ──────────────────────────────────────────────────────────────────────────
    Le raisonnement, parce qu'une première version de ce test était FAUSSE
    ──────────────────────────────────────────────────────────────────────────
    Première idée, abandonnée : comparer les frontières de week-end entre l'été et l'hiver, en
    espérant qu'un mauvais décalage produise « deux régimes ». **Ça ne peut pas marcher** : un
    décalage constant translate TOUTES les heures de la même façon, donc aucun décalage ne
    créera jamais deux régimes. Le test passait pour -6, -7 et -8 indifféremment — il ne
    prouvait rien.

    Ce qui discrimine vraiment, c'est un **repère extérieur connu dans la donnée elle-même** :
    le cash NASDAQ ouvre à **09h30 NY** et ferme à **16h00 NY**. Ce sont des heures publiques,
    indépendantes de FTMO. L'ouverture du cash provoque un pic d'amplitude sans équivalent dans
    la journée, et la clôture un effondrement immédiat de l'activité.

    Deux marqueurs, mesurés sur des RAPPORTS et non sur un maximum :

    - **le saut à l'ouverture** : l'amplitude de la barre de 9 h vaut ~2x celle de 8 h ;
    - **l'effondrement à la clôture** : l'activité de 16 h tombe à ~0,3x celle de 15 h.

    ⚠️ Pourquoi pas simplement « le pic d'amplitude est à 9 h » — c'était la version
    précédente, et elle était **fragile** : en été, 9 h et 10 h ne diffèrent que de 0,8 %, si
    bien que le maximum bascule de l'un à l'autre au gré du bruit. Le test aurait échoué sur
    une sous-période sans qu'aucun décalage ne soit faux. Un rapport de 2x, lui, ne bascule pas.

    Le test tourne aussi **saison par saison**. C'est la vraie garde anti-DST : si le serveur
    cessait de suivre le changement d'heure américain, l'été et l'hiver se décaleraient d'une
    heure l'un par rapport à l'autre, et un test sur données agrégées ne le verrait pas.
    """
    lignes: list[tuple[int, float, float, bool]] = []
    for h in (-10, -9, -8, -7, -6, -5, -4):
        df = convert_server_to_ny(brut, hours=h)
        df = df[df["DateTime"] >= DEBUT_INTRADAY]
        saut, chute = _marqueurs_cash(pd.DataFrame(df))
        lignes.append((h, saut, chute, saut > SAUT_OUVERTURE_MIN and chute < CHUTE_CLOTURE_MAX))

    print("\n       décalage | saut 9h/8h | activité 16h/15h |")
    print("       ---------+------------+------------------+")
    for h, saut, chute, ok in lignes:
        verdict = "  <- COHÉRENT avec le cash NASDAQ" if ok else ""
        marque = "*" if h == SERVER_TO_NY_HOURS else " "
        print(f"      {marque}{h:+7d} h |   x{saut:.2f}    |      x{chute:.2f}       |{verdict}")
    print()

    valides = [h for h, _, _, ok in lignes if ok]
    if valides != [SERVER_TO_NY_HOURS]:
        _echec(
            f"le décalage n'est pas identifié de façon unique : décalages cohérents = {valides}, "
            f"attendu [{SERVER_TO_NY_HOURS}]"
        )

    # Garde anti-DST. On teste les saisons ET, surtout, les FENÊTRES DE DÉSALIGNEMENT :
    # les quelques semaines par an où les États-Unis ont changé d'heure et l'Europe pas encore
    # (ou l'inverse). C'est le seul moment où un serveur réglé sur le DST européen se
    # trahirait — un test sur l'année entière ne le verrait jamais, ces fenêtres pesant moins
    # de 7 % des barres.
    df = convert_server_to_ny(brut, hours=SERVER_TO_NY_HOURS)
    df = df[df["DateTime"] >= DEBUT_INTRADAY]
    dates_ny = pd.Series(df["DateTime"])
    mois: np.ndarray = np.asarray(dates_ny.dt.month, dtype=int)

    desalignes = np.zeros(len(df), dtype=bool)
    for debut, fin in FENETRES_DESALIGNEMENT_DST:
        desalignes |= ((dates_ny >= debut) & (dates_ny < fin)).to_numpy()

    fenetres: list[str] = []
    for nom, masque in (
        ("été", np.isin(mois, MOIS_ETE)),
        ("hiver", np.isin(mois, MOIS_HIVER)),
        ("désalignement US/EU", desalignes),
    ):
        s, c = _marqueurs_cash(pd.DataFrame(df[masque]))
        if not (s > SAUT_OUVERTURE_MIN and c < CHUTE_CLOTURE_MAX):
            _echec(
                f"la signature du cash ne tient pas sur « {nom} » (saut x{s:.2f}, chute x{c:.2f})"
            )
        fenetres.append(f"{nom} x{s:.2f}")

    print(
        f"PASS — 4. le décalage de {SERVER_TO_NY_HOURS} h est le SEUL cohérent avec le cash "
        f"NASDAQ, et il tient sur chaque fenêtre ({', '.join(fenetres)}) — le serveur suit bien "
        "le DST américain"
    )


def check_5_rollover_a_17h(df: pd.DataFrame) -> None:
    """
    Le rollover quotidien doit tomber à 17:00 NY (= minuit serveur).

    On le repère par les trous d'exactement une barre : la barre de 17:00 est absente, donc
    l'écart mesuré à 18:00 vaut 120 minutes. Si la majorité de ces trous ne tombait pas à
    18:00, notre lecture de la journée serait fausse — et avec elle le label (C9) et le
    plancher journalier du juge (F2).
    """
    recent = df[df["DateTime"] >= DEBUT_INTRADAY]
    trous_1h = recent[(recent["gap_kind"] == "trou") & (recent["minutes_since_prev"] == 120)]
    if len(trous_1h) == 0:
        _echec("aucun trou d'une barre trouvé — structure de session inattendue")
    heures_reprise: np.ndarray = np.asarray(pd.Series(trous_1h["DateTime"]).dt.hour, dtype=int)
    part_18h = float((heures_reprise == 18).mean())
    if part_18h < 0.90:
        _echec(f"seuls {part_18h:.1%} des trous d'une barre tombent à 18:00 NY (attendu > 90 %)")
    print(
        f"PASS — 5. rollover à 17:00 NY : {part_18h:.1%} des {len(trous_1h):,} trous d'une barre "
        "reprennent à 18:00"
    )


def check_6_journee_ftmo_coherente(df: pd.DataFrame) -> None:
    """
    Une journée FTMO doit commencer le soir (17-18 h NY) et finir à 16 h le lendemain, sans
    jamais s'étaler sur plus de 24 heures.

    C'est la garde contre un regroupement silencieusement faux : si on revenait par erreur à la
    date calendaire, une « journée » couvrirait deux morceaux de sessions différentes, et rien
    d'autre ne le signalerait.
    """
    recent = df[df["DateTime"] >= DEBUT_INTRADAY]
    bornes = recent.groupby("jour_ftmo", observed=True).agg(
        debut=("DateTime", "min"), fin=("DateTime", "max")
    )
    ecart = pd.Series(bornes["fin"]) - pd.Series(bornes["debut"])
    duree_h: np.ndarray = np.asarray(ecart.dt.total_seconds(), dtype=float) / 3600.0
    if float(duree_h.max()) > 24.0:
        _echec(f"une journée FTMO s'étale sur {duree_h.max():.1f} h (> 24 h)")

    h_debut: np.ndarray = np.asarray(pd.Series(bornes["debut"]).dt.hour, dtype=int)
    h_fin: np.ndarray = np.asarray(pd.Series(bornes["fin"]).dt.hour, dtype=int)
    part_debut = float(np.isin(h_debut, [17, 18]).mean())
    part_fin = float((h_fin == HEURE_CLOTURE_CASH).mean())
    if part_debut < 0.90 or part_fin < 0.90:
        _echec(
            f"journées mal bornées : début 17-18 h = {part_debut:.1%}, fin 16 h = {part_fin:.1%}"
        )
    print(
        f"PASS — 6. journée FTMO cohérente : {part_debut:.1%} commencent à 17-18 h, "
        f"{part_fin:.1%} finissent à 16 h, durée max {duree_h.max():.0f} h"
    )


def check_7_spread_manquant(df: pd.DataFrame) -> None:
    """
    `spread_missing` doit dire exactement « le spread est absent », et le manque doit être
    **localisé**.

    ⚠️ La première version de ce contrôle **ne pouvait pas échouer** : elle se contentait de
    lister les années dégradées, et sortait en `PASS` quand il n'y en avait aucune. Le test de
    mutation l'a démasquée — on pouvait faire en sorte qu'aucun spread manquant ne soit jamais
    signalé, et ce contrôle applaudissait.

    Deux vérifications maintenant, l'une sur l'invariant, l'autre sur la forme du manque :

    1. `spread_missing` équivaut exactement à `Spread <= 0` — sinon le drapeau ment ;
    2. le manque est **concentré** sur une période identifiée, pas diffus. Un manque diffus
       signalerait un défaut d'export ; une concentration signale une limite côté serveur,
       qu'on peut traiter par imputation en connaissance de cause.
    """
    recent = df[df["DateTime"] >= "2021-01-01"]
    manquant: np.ndarray = np.asarray(recent["spread_missing"], dtype=bool)
    attendu: np.ndarray = np.asarray(recent["Spread"], dtype=float) <= 0
    if not bool((manquant == attendu).all()):
        _echec(
            f"`spread_missing` ne correspond pas à « Spread <= 0 » sur "
            f"{int((manquant != attendu).sum())} barre(s)"
        )

    annees: np.ndarray = np.asarray(pd.Series(recent["DateTime"]).dt.year, dtype=int)
    taux = {int(a): float(manquant[annees == a].mean()) for a in np.unique(annees)}
    degradees = {a: v for a, v in taux.items() if v > 0.10}
    propres = [a for a, v in taux.items() if v <= 0.10]

    if not degradees:
        _echec(
            "aucune année dégradée alors que le NASDAQ FTMO en a une (2021, ~73 % de spread "
            "absent). Si le drapeau ne repère plus rien, c'est lui qui est cassé, pas la donnée."
        )
    if len(degradees) > 2:
        _echec(f"manque diffus sur {len(degradees)} années {sorted(degradees)} — export suspect")

    detail = ", ".join(f"{a} ({100 * v:.0f} %)" for a, v in sorted(degradees.items()))
    print(f"PASS — 7. `spread_missing` exact, et manque localisé : {detail} ; propre sur {propres}")


def check_8_aucune_journee_weekend(df: pd.DataFrame) -> None:
    """
    Aucune journée FTMO ne doit tomber un samedi ou un dimanche.

    Ce n'est vrai aujourd'hui que **parce que les données le veulent bien** : il n'existe aucune
    barre le vendredi après 17 h NY, donc rien ne bascule vers le samedi. Rien dans le code ne
    l'interdit. Si le broker ouvrait un jour une session le vendredi soir, on se retrouverait
    avec des « journées FTMO » du samedi sans que rien ne le signale.
    """
    recent = df[df["DateTime"] >= DEBUT_INTRADAY]
    jours: np.ndarray = np.asarray(pd.to_datetime(pd.Series(recent["jour_ftmo"])).dt.dayofweek)
    mauvais = np.isin(jours, [5, 6])
    if bool(mauvais.any()):
        exemples = sorted(set(np.asarray(recent["jour_ftmo"])[mauvais]))[:3]
        _echec(f"{int(mauvais.sum())} barre(s) sur une journée FTMO de week-end : {exemples}")
    print("PASS — 8. aucune journée FTMO ne tombe un samedi ou un dimanche")


def check_9_les_gardes_savent_refuser(brut: pd.DataFrame) -> None:
    """
    **Tests négatifs.** Les deux garde-fous du nettoyeur doivent savoir lever.

    Ni l'un ni l'autre n'avait jamais été exécuté — la mesure de couverture l'a montré. Un
    `raise` jamais atteint est du code mort qu'on croit protecteur.
    """
    ampute = pd.DataFrame(brut.drop(columns=["Spread"]))
    chemin = DEFAULT_RAW.parent / "_test_colonne_absente.csv"
    ampute.to_csv(chemin, index=False, date_format="%Y.%m.%d %H:%M:%S")
    try:
        load_raw(chemin)
    except ValueError as e:
        if "absentes" not in str(e).lower():
            _echec(f"export incomplet refusé mais pour une AUTRE raison : {e}")
    else:
        _echec("load_raw a accepté un export auquel il manque une colonne")
    finally:
        chemin.unlink(missing_ok=True)

    doublons = pd.concat([brut.head(3), brut.head(3)]).sort_values("DateTime")
    chemin2 = DEFAULT_RAW.parent / "_test_doublons.csv"
    doublons.rename(columns={"Volume": "TickVolume"}).to_csv(
        chemin2, index=False, date_format="%Y.%m.%d %H:%M:%S"
    )
    try:
        clean_ftmo_h1(chemin2, out_path=None)
    except ValueError as e:
        if "double" not in str(e).lower():
            _echec(f"export à doublons refusé mais pour une AUTRE raison : {e}")
    else:
        _echec("clean_ftmo_h1 a accepté un export contenant des horodatages en double")
    finally:
        chemin2.unlink(missing_ok=True)

    print("PASS — 9. les garde-fous refusent un export incomplet ET un export à doublons")


def check_10_ecriture_et_relecture(brut: pd.DataFrame) -> None:
    """
    Le fichier écrit doit se relire **à l'identique**, types compris.

    Le chemin d'écriture n'était jamais exercé par les preuves (elles passent `out_path=None`),
    et c'est pourtant celui qu'utilise le script en ligne de commande. C'est exactement là que
    s'était glissé le défaut de `jour_ftmo`, qui revenait en texte après un aller-retour CSV.
    """
    chemin = DEFAULT_RAW.parent / "_test_aller_retour.csv"
    try:
        ecrit = clean_ftmo_h1(DEFAULT_RAW, out_path=chemin)
        relu = load_clean(chemin)
        if len(relu) != len(ecrit):
            _echec(f"aller-retour : {len(ecrit)} lignes écrites, {len(relu)} relues")
        if not pd.Series(relu["DateTime"]).equals(pd.Series(ecrit["DateTime"])):
            _echec("aller-retour : les horodatages ne correspondent plus")
        if list(relu.columns) != list(ecrit.columns):
            _echec(f"aller-retour : colonnes différentes {list(relu.columns)}")
        typ = type(pd.Series(relu["jour_ftmo"]).iloc[0]).__name__
        if typ != "date":
            _echec(f"aller-retour : jour_ftmo relu comme {typ} au lieu de date")
        print(f"PASS — 10. aller-retour disque fidèle ({len(relu):,} lignes, jour_ftmo en {typ})")
    finally:
        chemin.unlink(missing_ok=True)


def check_11_resume_lisible(df: pd.DataFrame) -> None:
    """Le résumé imprimé après un run ne doit pas planter — c'est la sortie visible du script."""
    texte = resume(df)
    if "journées FTMO" not in texte:
        _echec(f"résumé incomplet : {texte[:80]!r}")
    print(f"PASS — 11. le résumé s'affiche ({len(texte.splitlines())} lignes)")


def check_13_classification_des_ecarts(df: pd.DataFrame) -> None:
    """
    `gap_kind` doit dire la vérité — chaque catégorie vérifiée contre sa définition.

    ⚠️ Ce contrôle est né d'un trou trouvé par le test de mutation : `gap_kind` et `gap_flag`
    étaient **produits mais jamais vérifiés**. On pouvait mettre la durée nominale d'une barre
    à 30 minutes et les 22 preuves passaient toutes — plus une seule barre n'était classée
    « contigu », et personne ne s'en apercevait.

    Trois vérifications, chacune sur la définition de la catégorie :

    - ``contigu``  : l'écart vaut EXACTEMENT la durée d'une barre, et c'est le cas courant ;
    - ``weekend``  : la barre précédente est un vendredi, la courante un dimanche ou un lundi ;
    - ``trou``     : l'écart n'est ni une barre, ni un week-end.
    """
    recent = df[df["DateTime"] >= DEBUT_INTRADAY]
    kind: np.ndarray = np.asarray(recent["gap_kind"], dtype=object)
    minutes: np.ndarray = np.asarray(recent["minutes_since_prev"], dtype=float)
    flag: np.ndarray = np.asarray(recent["gap_flag"], dtype=bool)

    contigu = kind == "contigu"
    if not bool((minutes[contigu] == BAR_MINUTES).all()):
        mauvais = sorted(set(minutes[contigu][minutes[contigu] != BAR_MINUTES]))[:5]
        _echec(f"des barres « contigu » n'ont pas un écart de {BAR_MINUTES} min : {mauvais}")

    part_contigu = float(contigu.mean())
    if part_contigu < 0.90:
        _echec(
            f"seulement {part_contigu:.1%} des barres sont contiguës — attendu > 90 %. "
            "Soit les données sont trouées, soit la durée nominale d'une barre est fausse."
        )

    if not bool((flag == ~contigu).all()):
        _echec("`gap_flag` ne correspond pas à « n'est pas contigu »")

    weekend = kind == "weekend"
    dow_prec: np.ndarray = np.asarray(pd.Series(recent["DateTime"]).shift(1).dt.dayofweek)
    dow_cour: np.ndarray = np.asarray(pd.Series(recent["DateTime"]).dt.dayofweek)
    mal_classes = weekend & ~((dow_prec == 4) & np.isin(dow_cour, [6, 0]))
    if bool(mal_classes.any()):
        _echec(f"{int(mal_classes.sum())} écart(s) classé(s) « weekend » à tort")

    print(
        f"PASS — 13. classification des écarts exacte : {part_contigu:.1%} contigus "
        f"(tous à {BAR_MINUTES} min), {int(weekend.sum())} week-ends bien formés"
    )


def check_12_ligne_de_commande(monkey_out: Path) -> None:
    """
    Le point d'entrée en ligne de commande doit tourner de bout en bout.

    C'est ce que Hugo lance réellement, et c'était la plus grosse zone jamais exécutée par les
    preuves : `main()`, l'analyse des arguments, l'écriture disque, l'affichage. Une erreur là
    casse la commande sans qu'aucune preuve ne s'en aperçoive.
    """
    import ftmo_cleaner

    argv = sys.argv
    sys.argv = ["ftmo_cleaner.py", "--raw", str(DEFAULT_RAW), "--out", str(monkey_out)]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as sortie:
            ftmo_cleaner.main()
        if not monkey_out.exists():
            _echec("la ligne de commande n'a pas écrit son fichier de sortie")
        if "journées FTMO" not in sortie.getvalue():
            _echec("la ligne de commande n'affiche pas son résumé")
        print(f"PASS — 12. la ligne de commande tourne et écrit {monkey_out.name}")
    finally:
        sys.argv = argv
        monkey_out.unlink(missing_ok=True)


def check_14_deux_regimes_et_ouvertures_a_18h(df: pd.DataFrame) -> None:
    """
    **La preuve de la correction B4 (2026-08-10).** Avant elle, un -7 constant mettait les
    9 premières semaines de l'ère intraday en avance d'une heure : sessions 17:00 -> 15:00
    « NY », ouverture du cash à 8 h 30 — et les tolérances agrégées de la preuve n°4 ne le
    voyaient pas (la fenêtre pèse 3-4 % de chaque coupe).

    Trois vérifications, chacune discriminante :

    1. la donnée **porte son régime** : `decalage_serveur_ny` existe, vaut -6 puis -7, avec
       EXACTEMENT une bascule ;
    2. **toute reprise de session ouvre à 18:00 NY** — l'horloge extérieure, sur TOUTE la
       série, ère D1 comprise. Seule tolérance : les reprises en cours de session après un
       trou de feed (6 sur ~2 200) ;
    3. la fenêtre corrigée porte la **signature du cash NASDAQ mesurée sur elle seule** —
       avant la correction, le saut 9h/8h y valait ~x1,1 (le vrai saut était à 8h30).
    """
    if "decalage_serveur_ny" not in df.columns:
        _echec("la colonne `decalage_serveur_ny` manque : la donnée ne porte plus son régime")
    dec: np.ndarray = np.asarray(df["decalage_serveur_ny"], dtype=int)
    valeurs = set(np.unique(dec).tolist())
    if valeurs != {SERVER_TO_NY_HOURS_ANCIEN, SERVER_TO_NY_HOURS}:
        _echec(f"décalages inattendus : {sorted(valeurs)}, attendu [-7, -6]")
    changements = int((dec[1:] != dec[:-1]).sum())
    if changements != 1:
        _echec(f"{changements} bascule(s) de régime dans la donnée, attendu exactement 1")

    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60.0
    reprises = pd.Series(df["DateTime"]).dt.hour[np.asarray(ecart > BAR_MINUTES)]
    part_18 = float((reprises == 18).mean())
    if part_18 < 0.995:
        _echec(
            f"seules {part_18:.2%} des reprises de session ouvrent à 18:00 NY — la conversion "
            "à deux régimes ne tient plus (avant la correction B4 : les 9 semaines de T1 2021 "
            "ouvraient à 17:00)"
        )

    fenetre = df[(df["DateTime"] >= "2021-01-21") & (df["DateTime"] < "2021-03-28")]
    saut, chute = _marqueurs_cash(pd.DataFrame(fenetre))
    if not (saut > SAUT_OUVERTURE_MIN and chute < CHUTE_CLOTURE_MAX):
        _echec(
            f"la fenêtre T1 2021 ne porte pas la signature du cash NASDAQ mesurée sur elle "
            f"seule (saut 9h/8h x{saut:.2f}, chute 16h/15h x{chute:.2f}) — l'heure y est fausse"
        )
    print(
        f"PASS — 14. deux régimes (-6 puis -7, 1 bascule), {part_18:.1%} des reprises à 18:00 "
        f"NY, et la fenêtre T1 2021 porte la signature du cash (saut x{saut:.2f})"
    )


def check_15_les_gardes_de_regime_savent_refuser() -> None:
    """
    Les deux gardes de `detecter_decalages_serveur_ny` doivent LEVER, pas deviner.

    Un export dont la première ouverture ne signe aucun régime, ou qui montre plus d'une
    bascule (ou une bascule à l'envers), ne ressemble plus à ce qui a été vérifié : le
    convertir en silence, c'est réintroduire la classe de bug que B4 vient de payer.
    """
    base = {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}

    non_signee = pd.DataFrame(
        {"DateTime": pd.date_range("2024-06-03 05:00", periods=5, freq="h"), **base}
    )
    try:
        detecter_decalages_serveur_ny(non_signee)
    except ValueError as e:
        if "signe aucun régime" not in str(e):
            _echec(f"première ouverture non signée refusée pour une AUTRE raison : {e}")
    else:
        _echec("une première ouverture à 05:00 serveur a été acceptée sans régime")

    morceaux = [
        pd.DataFrame({"DateTime": pd.date_range(f"{jour} {h:02d}:00", periods=5, freq="h"), **base})
        for jour, h in (("2024-06-03", 0), ("2024-06-10", 1), ("2024-06-17", 0))
    ]
    double = pd.concat(morceaux, ignore_index=True)
    try:
        detecter_decalages_serveur_ny(double)
    except ValueError as e:
        if "bascule" not in str(e):
            _echec(f"double bascule refusée mais pour une AUTRE raison : {e}")
    else:
        _echec("un export à DEUX bascules de régime a été converti en silence")

    print("PASS — 15. les gardes de régime refusent l'inconnu : ouverture non signée, re-bascule")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU NETTOYEUR FTMO H1 — tâche 1.3")
    print("=" * 78 + "\n")

    brut = load_raw(DEFAULT_RAW)
    propre = clean_ftmo_h1(DEFAULT_RAW, out_path=None)

    check_1_aucune_ligne_perdue(brut, propre)
    check_2_horodatages_uniques_et_croissants(propre)
    check_3_coherence_ohlc(propre)
    check_4_preuve_du_decalage(brut)
    check_5_rollover_a_17h(propre)
    check_6_journee_ftmo_coherente(propre)
    check_7_spread_manquant(propre)
    check_8_aucune_journee_weekend(propre)
    check_9_les_gardes_savent_refuser(brut)
    check_10_ecriture_et_relecture(brut)
    check_11_resume_lisible(propre)
    check_12_ligne_de_commande(DEFAULT_RAW.parent / "_test_cli.csv")
    check_13_classification_des_ecarts(propre)
    check_14_deux_regimes_et_ouvertures_a_18h(propre)
    check_15_les_gardes_de_regime_savent_refuser()
    check_93_les_gardes_du_NETTOYEUR_mordent()

    # Garde de non-régression : les briques prises isolément doivent redonner le tout.
    manuel = add_ftmo_trading_day(add_gap_columns(convert_server_to_ny(brut)))
    if not manuel["DateTime"].equals(propre["DateTime"]):
        _echec("la composition manuelle des étapes ne redonne pas le résultat de clean_ftmo_h1")

    print("\n" + "=" * 78)
    print("LES 16 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_les_gardes_du_NETTOYEUR_mordent() -> None:
    """
    Deux gardes qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    ⚠️ La seconde est la plus importante du module : **ce nettoyeur annote, il ne filtre JAMAIS**
    (B5). Une régression qui ferait disparaître des lignes serait parfaitement silencieuse —
    l'export final aurait simplement moins de barres.
    """
    import ftmo_cleaner

    # ① des horodatages non triés : la détection de régime lirait des reprises
    df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2021-03-02 10:00", "2021-03-01 10:00"]),
            "Open": [1.0, 1.0],
            "High": [1.0, 1.0],
            "Low": [1.0, 1.0],
            "Close": [1.0, 1.0],
        }
    )
    _leve(
        lambda: ftmo_cleaner.detecter_decalages_serveur_ny(df),
        ValueError,
        "horodatages non triés",
        "non triés",
    )
    # ② une étape du nettoyage qui FILTRE : la garde de non-filtrage doit l'attraper. On trafique
    #    la dernière étape le temps du contrôle, sans toucher au module sur disque.
    vraie = ftmo_cleaner.add_ftmo_trading_day
    ftmo_cleaner.add_ftmo_trading_day = lambda d: vraie(d).iloc[:-1]
    try:
        _leve(
            lambda: ftmo_cleaner.clean_ftmo_h1(out_path=None),
            AssertionError,
            "une étape du nettoyage qui filtre",
            "JAMAIS filtrer",
        )
    finally:
        ftmo_cleaner.add_ftmo_trading_day = vraie
    # anti-vacuité : une fois l'étape rétablie, le nettoyage repasse
    if len(ftmo_cleaner.clean_ftmo_h1(out_path=None)) == 0:
        _echec("anti-vacuité : le nettoyage ne rend plus rien une fois l'étape rétablie")
    print("  93. gardes du nettoyeur : horodatages non triés, et le nettoyage qui FILTRE (B5)  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
