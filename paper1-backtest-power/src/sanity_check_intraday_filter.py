"""
Preuve du filtre intraday — tâche 1.4.

Dix vérifications, dont **trois qui provoquent délibérément un échec** : un garde-fou qui
n'a jamais été vu refuser quelque chose n'est pas un garde-fou, c'est une décoration.

Lancer :  .venv/bin/python src/sanity_check_intraday_filter.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ftmo_cleaner import load_clean
from intraday_filter import (
    DEFAULT_IN,
    MIN_BARRES_SEANCE,
    assert_aucune_barre_journaliere,
    assert_signature_journaliere,
    detecter_barres_journalieres,
    filtrer_intraday,
)


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _panel_synthetique(n_seances: int, barres_par_seance: int, amplitude: float) -> pd.DataFrame:
    """Petit panel fabriqué à la main, pour exercer le filtre hors des vraies données."""
    lignes = []
    for j in range(n_seances):
        jour = pd.Timestamp("2030-01-01") + pd.Timedelta(days=j)
        for b in range(barres_par_seance):
            t = jour + pd.Timedelta(hours=18 + b)
            lignes.append(
                {
                    "DateTime": t,
                    "Open": 100.0,
                    "High": 100.0 + amplitude,
                    "Low": 100.0,
                    "Close": 100.0,
                    "jour_ftmo": jour.date(),
                }
            )
    return pd.DataFrame(lignes)


# ──────────────────────────────────────────────────────────────────────────


def check_1_signature_reelle(df: pd.DataFrame) -> None:
    """Sur les vraies données, ce qui est retiré doit avoir la signature d'une bougie D1."""
    _, r = filtrer_intraday(df)
    if r.part_heure_dominante < 0.99:
        _echec(f"barres retirées dispersées ({r.part_heure_dominante:.0%} sur une heure)")
    if r.amplitude_mediane_retiree < 2 * r.amplitude_mediane_gardee:
        _echec("les barres retirées n'ont pas une amplitude de journée")
    print(
        f"PASS — 1. signature : {r.n_barres_retirees:,} barres retirées, "
        f"{r.part_heure_dominante:.0%} à {r.heure_dominante_retiree} h, amplitude "
        f"{r.amplitude_mediane_retiree:.0f} pts contre {r.amplitude_mediane_gardee:.0f} pts"
    )


def check_2_frontiere_detectee(df: pd.DataFrame) -> None:
    """
    La frontière trouvée doit être POSTÉRIEURE à la date qu'on avait écrite en dur.

    C'est le check qui documente l'erreur d'origine : la décision B3 annonçait le
    2021-01-03, la mesure dit **2021-01-22**. Un filtre sur la date aurait laissé passer
    13 barres journalières en plein dans le dev.
    """
    _, r = filtrer_intraday(df)
    premiere = pd.Timestamp(str(r.premiere_seance_intraday))
    if premiere <= pd.Timestamp("2021-01-03"):
        _echec(f"frontière détectée {premiere.date()} — attendue après le 2021-01-03")
    apres_date_en_dur: np.ndarray = (pd.Series(df["DateTime"]) >= "2021-01-03").to_numpy()
    journalieres: np.ndarray = detecter_barres_journalieres(df).to_numpy()
    n_apres_date_en_dur = int((apres_date_en_dur & journalieres).sum())
    print(
        f"PASS — 2. frontière mesurée au {premiere.date()}, pas au 2021-01-03 : un filtre sur "
        f"la date en dur aurait laissé passer {n_apres_date_en_dur} barre(s) journalière(s)"
    )


def check_3_plus_aucune_apres_filtrage(df: pd.DataFrame) -> None:
    """Après filtrage, la garde ne doit plus rien trouver — sinon le filtre est incomplet."""
    garde, _ = filtrer_intraday(df)
    assert_aucune_barre_journaliere(garde)
    tailles = garde.groupby("jour_ftmo", observed=True).size().to_numpy()
    print(
        f"PASS — 3. après filtrage : {len(garde):,} barres, séance la plus courte "
        f"{int(tailles.min())} barres (seuil {MIN_BARRES_SEANCE})"
    )


def check_4_la_garde_sait_refuser(df: pd.DataFrame) -> None:
    """
    **Test négatif.** On réinjecte une bougie journalière et la garde DOIT lever.

    Sans ce test, rien ne prouve que `assert_aucune_barre_journaliere` fasse autre chose que
    renvoyer `None` en toutes circonstances.
    """
    garde, _ = filtrer_intraday(df)
    intruse = pd.DataFrame([garde.iloc[0].to_dict()])
    intruse.loc[0, "DateTime"] = pd.Timestamp("2019-06-03 17:00:00")
    intruse.loc[0, "jour_ftmo"] = pd.Timestamp("2019-06-03").date()
    pollue = pd.concat([intruse, garde], ignore_index=True)
    try:
        assert_aucune_barre_journaliere(pollue)
    except AssertionError:
        print("PASS — 4. la garde REFUSE bien un panel où une bougie journalière a été réinjectée")
        return
    _echec("la garde a laissé passer une bougie journalière réinjectée")


def check_5_refus_si_signature_absente() -> None:
    """
    **Test négatif.** Des séances courtes mais SANS signature journalière doivent bloquer.

    Panel fabriqué : des séances d'UNE barre, toutes à la même heure — donc le garde-fou sur
    l'heure est satisfait — mais d'amplitude **normale**. Seul le second marqueur peut alors
    s'y opposer.

    ⚠️ La première version de ce test se croyait sur l'amplitude alors qu'elle déclenchait le
    garde-fou sur l'heure (ses barres étaient réparties sur 2 heures). La mesure de couverture
    l'a révélé : la ligne du garde-fou d'amplitude n'avait **jamais été exécutée**.
    """
    courtes = _panel_synthetique(n_seances=3, barres_par_seance=1, amplitude=1.0)
    longues = _panel_synthetique(n_seances=5, barres_par_seance=20, amplitude=1.0)
    longues["DateTime"] = longues["DateTime"] + pd.Timedelta(days=10)
    longues["jour_ftmo"] = pd.Series(longues["DateTime"]).dt.date
    panel = pd.concat([courtes, longues], ignore_index=True)

    masque = detecter_barres_journalieres(panel)
    if not bool(masque.any()):
        _echec("le panel de test ne déclenche même pas le critère de comptage")
    try:
        assert_signature_journaliere(panel, masque)
    except AssertionError:
        print(
            "PASS — 5. le filtre REFUSE de couper des séances courtes dont l'amplitude est "
            "normale (garde-fou n°2, distinct de celui sur l'heure)"
        )
        return
    _echec("le filtre a accepté de supprimer des barres qui n'ont pas la signature journalière")


def check_6_refus_si_heures_dispersees() -> None:
    """
    **Test négatif.** Des barres d'amplitude journalière mais dispersées dans la journée
    doivent bloquer aussi — les deux marqueurs sont exigés, pas l'un ou l'autre.
    """
    courtes = _panel_synthetique(n_seances=6, barres_par_seance=1, amplitude=50.0)
    heures = np.array([2, 7, 11, 14, 19, 22])
    base = pd.to_datetime(pd.Series(courtes["jour_ftmo"]).astype(str))
    courtes["DateTime"] = base + pd.to_timedelta(heures, unit="h")
    longues = _panel_synthetique(n_seances=5, barres_par_seance=20, amplitude=1.0)
    longues["DateTime"] = longues["DateTime"] + pd.Timedelta(days=10)
    longues["jour_ftmo"] = pd.Series(longues["DateTime"]).dt.date
    panel = pd.concat([courtes, longues], ignore_index=True)

    masque = detecter_barres_journalieres(panel)
    try:
        assert_signature_journaliere(panel, masque)
    except AssertionError:
        print("PASS — 6. le filtre REFUSE de couper des barres dont les heures sont dispersées")
        return
    _echec("le filtre a accepté de supprimer des barres réparties sur toute la journée")


def check_7_rapport_lisible(df: pd.DataFrame) -> None:
    """Le rapport doit s'imprimer sans planter — c'est ce que voit l'utilisateur du script."""
    _, r = filtrer_intraday(df)
    texte = str(r)
    if "barres" not in texte or "signature" not in texte:
        _echec(f"rapport incomplet : {texte[:80]!r}")
    print(f"PASS — 7. le rapport s'affiche ({len(texte.splitlines())} lignes)")


def check_8_masque_vide_ne_plante_pas() -> None:
    """
    Un panel déjà propre ne doit rien casser.

    Cas jamais exercé jusqu'ici : quand il n'y a **rien** à retirer, la vérification de
    signature part sur un ensemble vide. Une division par zéro ou un `argmax` sur un tableau
    vide y serait passée inaperçue.
    """
    propre = _panel_synthetique(n_seances=5, barres_par_seance=20, amplitude=1.0)
    garde, r = filtrer_intraday(propre)
    if r.n_barres_retirees != 0 or len(garde) != len(propre):
        _echec(f"un panel propre a été amputé : {r.n_barres_retirees} barre(s) retirée(s)")
    assert_aucune_barre_journaliere(garde)
    print("PASS — 8. un panel sans bougie journalière traverse le filtre intact")


def check_9_message_clair_si_panel_brut() -> None:
    """Un panel non nettoyé doit produire un message qui dit quoi faire, pas un `KeyError` nu."""
    brut = pd.DataFrame(
        {"DateTime": pd.to_datetime(["2024-03-04 18:00"]), "High": [1.0], "Low": [0.0]}
    )
    try:
        detecter_barres_journalieres(brut)
    except ValueError as e:
        if "clean_ftmo_h1" not in str(e):
            _echec(f"message peu utile : {e}")
        print("PASS — 9. un panel non nettoyé donne un message explicite, pas un KeyError")
        return
    _echec("un panel sans colonne jour_ftmo n'a pas été rejeté")


def check_10_ligne_de_commande(sortie: Path) -> None:
    """Le point d'entrée en ligne de commande doit tourner de bout en bout et écrire son fichier."""
    import intraday_filter

    argv = sys.argv
    sys.argv = ["intraday_filter.py", "--input", str(DEFAULT_IN), "--out", str(sortie)]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as flux:
            intraday_filter.main()
        if not sortie.exists():
            _echec("la ligne de commande n'a pas écrit son fichier de sortie")
        if "séances retirées" not in flux.getvalue():
            _echec("la ligne de commande n'affiche pas son rapport")
        print(f"PASS — 10. la ligne de commande tourne et écrit {sortie.name}")
    finally:
        sys.argv = argv
        sortie.unlink(missing_ok=True)


def check_11_refus_si_amplitude_non_finie() -> None:
    """
    **Test négatif — piège n°8, audit du 2026-08-18.** Une amplitude `NaN` désactivait ce
    garde-fou **en silence** : `nan > 0` vaut False, donc la condition entière était fausse et
    l'assertion ne levait jamais.

    Or c'est précisément ce garde-fou qui empêche le filtre de devenir destructeur si les données
    changent. Il se serait tu **au moment exact où il aurait dû parler** — sur des OHLC troués,
    c'est-à-dire le cas qui justifie son existence.
    """
    courtes = _panel_synthetique(n_seances=6, barres_par_seance=1, amplitude=50.0)
    longues = _panel_synthetique(n_seances=5, barres_par_seance=20, amplitude=1.0)
    longues["DateTime"] = longues["DateTime"] + pd.Timedelta(days=10)
    longues["jour_ftmo"] = pd.Series(longues["DateTime"]).dt.date
    panel = pd.concat([courtes, longues], ignore_index=True)
    # Un seul High manquant suffit : la médiane des amplitudes devient NaN.
    panel.loc[panel.index[-1], "High"] = float("nan")

    masque = detecter_barres_journalieres(panel)
    try:
        assert_signature_journaliere(panel, masque)
    except AssertionError as e:
        if "non finie" not in str(e).lower():
            _echec(f"levée pour une AUTRE raison — « {str(e)[:90]} »")
        print("PASS — 11. le filtre REFUSE de couper quand une amplitude est non finie")
        return
    _echec("une amplitude NaN a désactivé le garde-fou d'amplitude sans lever")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU FILTRE INTRADAY — tâche 1.4")
    print("=" * 78 + "\n")

    df = load_clean(DEFAULT_IN)
    check_1_signature_reelle(df)
    check_2_frontiere_detectee(df)
    check_3_plus_aucune_apres_filtrage(df)
    check_4_la_garde_sait_refuser(df)
    check_5_refus_si_signature_absente()
    check_6_refus_si_heures_dispersees()
    check_7_rapport_lisible(df)
    check_8_masque_vide_ne_plante_pas()
    check_9_message_clair_si_panel_brut()
    check_10_ligne_de_commande(DEFAULT_IN.parent / "_test_cli_filtre.csv")
    check_11_refus_si_amplitude_non_finie()

    print("\n" + "=" * 78)
    print("LES 11 PREUVES PASSENT (dont 5 tests négatifs).")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
