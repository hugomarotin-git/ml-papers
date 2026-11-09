"""
Preuve des poids d'échantillon — tâches 3.2 et 3.3.

Douze vérifications. Ce module ne décide rien, mais il produit **le chiffre qui gouverne tous
les intervalles de confiance du projet** : le nombre d'exemples effectifs. S'il est trop
optimiste, chaque test de significativité qui suit devient trop permissif — et on se convainc
d'un edge qui n'existe pas.

L'essentiel des contrôles porte sur des panels **fabriqués**, où l'unicité se calcule à la main.
Sur les vraies données on ne peut que constater ; sur trois labels tracés au crayon, on peut
prouver.

Lancer :  .venv/bin/python src/sanity_check_poids.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ftmo_cleaner import DATA_DIR
from label_triple_barriere import load_labelle
from periodes import restreindre_au_dev
from poids_echantillon import (
    DECROISSANCE_NEUTRE,
    calculer_poids,
    concurrence,
    decroissance_temporelle,
    exemples_effectifs,
    poids_apprentissage,
    rapport_tilt_horaire,
    tilt_horaire,
    unicite,
)


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


def _leve(fn, attendu: type[Exception], quoi: str, contient: str = "") -> None:
    """
    Vérifie qu'un garde-fou lève — **et pour la bonne raison** (piège n°4 du projet).

    Contrôler le seul type d'exception laisse passer une preuve décorative : une AUTRE erreur du
    même type, levée plus haut, la validerait sans que la garde visée existe.
    """
    try:
        fn()
    except attendu as e:
        if contient and contient.lower() not in str(e).lower():
            _echec(f"{quoi} : levée pour une AUTRE raison — « {str(e)[:90]} »")
        return
    except Exception as e:
        _echec(f"{quoi} : attendu {attendu.__name__}, reçu {type(e).__name__} ({e})")
    _echec(f"{quoi} : aucune erreur levée, le garde-fou est décoratif")


def _panel(vies: list[tuple[int, int]], n: int) -> pd.DataFrame:
    """
    Panel jouet : `n` barres horaires, et une liste de labels donnés par (début, fin) en indices.

    Permet de calculer l'unicité **à la main** et de comparer. C'est la seule façon de prouver
    la formule ; sur les vraies données on ne pourrait que constater un résultat plausible.
    """
    t: list[pd.Timestamp] = list(pd.date_range("2024-06-03 18:00", periods=n, freq="h"))
    issue: list[str | None] = [None] * n
    debut: list[pd.Timestamp | None] = [None] * n
    fin: list[pd.Timestamp | None] = [None] * n
    for d, f in vies:
        issue[d - 1] = "haut"  # le candidat précède son entrée d'une barre
        debut[d - 1] = t[d]
        fin[d - 1] = t[f]
    return pd.DataFrame(
        {
            "DateTime": t,
            "issue": issue,
            "label_entree_dt": pd.to_datetime(pd.Series(debut)),
            "label_fin_dt": pd.to_datetime(pd.Series(fin)),
        }
    )


# ──────────────────────────────────────────────────────────────────────────


def check_1_concurrence_a_la_main() -> None:
    """
    Trois labels aux vies connues, concurrence calculée au crayon.

    Vies : [1-3], [2-4], [6-7]. À la barre 2 il y en a deux, à la barre 5 aucun.
    """
    df = _panel([(1, 3), (2, 4), (6, 7)], n=10)
    obtenu = list(concurrence(df))
    attendu = [0, 1, 2, 2, 1, 0, 1, 1, 0, 0]
    if obtenu != attendu:
        _echec(f"concurrence = {obtenu}, attendu {attendu}")
    print(f"PASS — 1. concurrence exacte sur un panel tracé à la main : {attendu}")


def check_2_unicite_a_la_main() -> None:
    """
    Unicité calculée au crayon sur le même panel.

    Label [1-3] : concurrences 1, 2, 2 -> moyenne de 1/1, 1/2, 1/2 = 0,667.
    Label [2-4] : concurrences 2, 2, 1 -> même chose = 0,667.
    Label [6-7] : concurrences 1, 1 -> unicité 1,0, il n'a jamais partagé un instant.
    """
    df = _panel([(1, 3), (2, 4), (6, 7)], n=10)
    u = unicite(df)
    obtenus = [float(x) for x in u[~np.isnan(u)]]
    attendus = [2 / 3, 2 / 3, 1.0]
    if not np.allclose(obtenus, attendus):
        _echec(
            f"unicité = {[round(x, 3) for x in obtenus]}, attendu {[round(x, 3) for x in attendus]}"
        )
    print(f"PASS — 2. unicité exacte : {[round(x, 3) for x in obtenus]} (calculée au crayon)")


def check_3_label_isole_vaut_un() -> None:
    """Un label qui ne chevauche personne doit valoir exactement 1,0 — le cas de référence."""
    df = _panel([(1, 3)], n=6)
    u = unicite(df)
    seul = float(u[~np.isnan(u)][0])
    if not np.isclose(seul, 1.0):
        _echec(f"un label isolé vaut {seul}, attendu 1.0")
    print("PASS — 3. un label sans chevauchement vaut exactement 1,0")


def check_4_bornes_de_lunicite(pond: pd.DataFrame) -> None:
    """L'unicité vit dans `]0, 1]` : au-dessus de 1 la formule serait fausse, à 0 elle diviserait."""
    u = np.asarray(pond["unicite"], dtype=float)
    v = u[~np.isnan(u)]
    if float(v.max()) > 1.0 + 1e-9 or float(v.min()) <= 0.0:
        _echec(f"unicité hors de ]0, 1] : min {v.min():.4f}, max {v.max():.4f}")
    print(f"PASS — 4. unicité dans ]0, 1] : min {v.min():.3f}, max {v.max():.3f}")


def check_5_effectifs_inferieurs_au_brut(pond: pd.DataFrame) -> None:
    """
    Les exemples effectifs doivent être **strictement inférieurs** au nombre de labels.

    Si les deux coïncidaient, ça voudrait dire qu'aucun label n'en chevauche un autre — donc
    que le calcul de concurrence ne fonctionne pas. C'est le contrôle qui attraperait une
    correction silencieusement inopérante.
    """
    n = int(pd.Series(pond["issue"]).notna().sum())
    eff = exemples_effectifs(pond)
    if eff >= n:
        _echec(f"{eff:,.0f} exemples effectifs pour {n:,} labels — la correction ne fait rien")
    if eff <= 0:
        _echec("exemples effectifs nuls")
    print(
        f"PASS — 5. {eff:,.0f} exemples effectifs pour {n:,} labels (x{n / eff:.1f} de réduction)"
    )


def check_6_concurrence_coherente_avec_lhorizon(pond: pd.DataFrame) -> None:
    """
    La concurrence ne peut pas dépasser l'horizon.

    Au plus `horizon` labels peuvent être vivants en même temps, puisqu'un label démarre à
    chaque barre et vit au plus `horizon` barres. Une valeur supérieure signalerait un
    double-comptage.
    """
    c = np.asarray(pond["concurrence"], dtype=float)
    from label_triple_barriere import HORIZON_BARRES

    if float(c.max()) > HORIZON_BARRES:
        _echec(f"concurrence max {c.max():.0f} > horizon {HORIZON_BARRES}")
    print(
        f"PASS — 6. concurrence max {int(c.max())}, cohérente avec un horizon de {HORIZON_BARRES}"
    )


def check_7_decroissance_neutre_par_defaut(pond: pd.DataFrame) -> None:
    """
    Par défaut, la décroissance temporelle ne doit **rien** changer (décision D2).

    Codée mais désactivée. L'activer serait une décision à justifier, donc un essai à compter —
    elle ne doit pas s'appliquer par inadvertance.
    """
    f = np.asarray(pond["facteur_recence"], dtype=float)
    v = f[~np.isnan(f)]
    if not np.allclose(v, 1.0):
        _echec(
            f"la décroissance n'est pas neutre par défaut : min {v.min():.3f}, max {v.max():.3f}"
        )
    u = np.asarray(pond["unicite"], dtype=float)
    w = np.asarray(pond["poids_echantillon"], dtype=float)
    if not np.allclose(u[~np.isnan(u)], w[~np.isnan(w)]):
        _echec("poids != unicité alors que la décroissance est neutre")
    print(f"PASS — 7. décroissance neutre par défaut (c = {DECROISSANCE_NEUTRE}), poids = unicité")


def check_8_decroissance_fonctionne_quand_activee() -> None:
    """
    **Test négatif de l'inverse.** Activée, elle doit réellement pénaliser l'ancien.

    Sans ce contrôle, rien ne distinguerait « neutre par défaut » de « cassée ».
    """
    df = _panel([(i, i + 2) for i in range(1, 40, 3)], n=60)
    u = unicite(df)
    f = decroissance_temporelle(u, c=0.2)
    v = f[~np.isnan(f)]
    if v[0] >= v[-1]:
        _echec(f"la décroissance ne pénalise pas l'ancien : début {v[0]:.3f}, fin {v[-1]:.3f}")
    if not np.isclose(v[-1], 1.0, atol=0.05):
        _echec(f"la plus récente devrait peser ~1,0, elle pèse {v[-1]:.3f}")
    print(
        f"PASS — 8. décroissance active : la plus ancienne pèse {v[0]:.2f}, la plus récente {v[-1]:.2f}"
    )


def check_9_aucun_poids_sur_les_barres_sans_label(pond: pd.DataFrame) -> None:
    """Une barre sans label ne doit porter ni unicité ni poids — sinon elle entrerait dans la somme."""
    sans = np.asarray(pd.Series(pond["issue"]).isna())
    u = np.asarray(pond["unicite"], dtype=float)
    if not bool(np.isnan(u[sans]).all()):
        _echec(f"{int((~np.isnan(u[sans])).sum())} barre(s) sans label portent une unicité")
    print(f"PASS — 9. les {int(sans.sum()):,} barres sans label ne portent aucun poids")


def check_10_refus_dun_panel_non_labelle() -> None:
    """Un panel sans les colonnes du label doit dire quoi faire, pas produire un `KeyError`."""
    nu = pd.DataFrame({"DateTime": pd.to_datetime(["2024-06-03 18:00"])})
    try:
        concurrence(nu)
    except ValueError as e:
        if "label_triple_barriere" not in str(e):
            _echec(f"message peu utile : {e}")
        print("PASS — 10. un panel non labellisé donne un message explicite")
        return
    _echec("un panel sans colonnes de label n'a pas été rejeté")


def check_12_date_de_fin_hors_panel_refusee() -> None:
    """
    Une date de FIN absente du panel doit être refusée, jamais décalée en silence.

    `searchsorted` ne signale pas une date absente : il renvoie un point d'insertion. Sans la
    garde symétrique sur `fin`, un panel où la barre de résolution manque (découpe par heures
    en phase 8, par exemple) décalerait la fenêtre de vie du label sans rien lever — unicité
    fausse, exemples effectifs faux, intervalles de confiance faux.
    """
    df = _panel([(1, 3)], n=6)
    df.loc[0, "label_fin_dt"] = df["label_fin_dt"].iloc[0] + pd.Timedelta(minutes=30)
    try:
        concurrence(df)
    except AssertionError as e:
        if "fin" not in str(e).lower():
            _echec(f"date de fin absente refusée mais pour une AUTRE raison : {e}")
        print("PASS — 12. une date de fin absente du panel est refusée, pas décalée en silence")
        return
    _echec(
        "une date de fin absente du panel a été acceptée : searchsorted a décalé la fenêtre "
        "de vie du label en silence"
    )


def check_13_decroissance_non_finie_refusee() -> None:
    """
    **Piège n°8, audit du 2026-08-18.** `c` — le réglage de décroissance temporelle (D2) —
    n'était validé nulle part. Un `NaN` traversait tout : `c == 1.0` est False, puis `c >= 0`
    est False aussi, donc la branche « troncature » était prise et la pente valait `NaN`.

    Résultat : des poids d'échantillon **tous NaN**. Un entraînement pondéré là-dessus ne pèse
    plus rien — et ça ne plante pas, ça produit un modèle entraîné à plat sans que rien ne le
    signale. `c` est un réglage de recherche : il arrivera un jour d'un balayage programmatique.
    """
    unicites = np.array([1.0, 0.5, 0.25, 1.0])
    for valeur, quoi in ((float("nan"), "NaN"), (float("inf"), "inf")):
        _leve(
            lambda v=valeur: decroissance_temporelle(unicites, c=v),
            ValueError,
            f"une décroissance {quoi}",
            contient="fini",
        )
    _leve(
        lambda: decroissance_temporelle(unicites, c=-1.0),
        ValueError,
        "c = -1 (division par zéro dans la pente)",
        contient="> -1",
    )
    _leve(
        lambda: decroissance_temporelle(unicites, c=-2.0),
        ValueError,
        "c < -1 (pente inversée, tous les poids écrasés à 0)",
        contient="> -1",
    )
    # Le domaine légitime passe toujours : la garde n'est pas un mur.
    for valide in (1.0, 0.5, 0.0, -0.5):
        sortie = decroissance_temporelle(unicites, c=valide)
        if not np.all(np.isfinite(sortie)):
            _echec(f"c={valide} est légitime mais produit des poids non finis")
    print(" 13. une décroissance temporelle NaN, infinie ou <= -1 est REFUSÉE  ✓")


def check_11_deterministe(df: pd.DataFrame) -> None:
    """Deux calculs successifs doivent donner exactement le même résultat."""
    a = calculer_poids(df)
    b = calculer_poids(df)
    if not np.allclose(
        np.nan_to_num(np.asarray(a["poids_echantillon"], float)),
        np.nan_to_num(np.asarray(b["poids_echantillon"], float)),
    ):
        _echec("le calcul des poids n'est pas déterministe")
    print("PASS — 11. calcul déterministe")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DES POIDS D'ÉCHANTILLON — tâches 3.2 et 3.3")
    print("=" * 78 + "\n")

    check_1_concurrence_a_la_main()
    check_2_unicite_a_la_main()
    check_3_label_isole_vaut_un()
    check_8_decroissance_fonctionne_quand_activee()
    check_10_refus_dun_panel_non_labelle()
    check_12_date_de_fin_hors_panel_refusee()
    check_13_decroissance_non_finie_refusee()
    check_90_le_tilt_horaire_des_poids_est_MESURE()
    check_91_un_label_qui_se_resout_AVANT_son_entree_est_refuse()
    check_93_date_d_ENTREE_hors_panel_refusee()
    check_95_la_deflation_est_LOCALE_a_ce_qu_on_lui_donne()
    check_96_les_gardes_du_poids_d_apprentissage_MORDENT()

    print()
    dev = restreindre_au_dev(load_labelle(DATA_DIR / "v2" / "nas100_h1_labelle.csv"))
    pond = calculer_poids(dev)
    check_4_bornes_de_lunicite(pond)
    check_5_effectifs_inferieurs_au_brut(pond)
    check_6_concurrence_coherente_avec_lhorizon(pond)
    check_7_decroissance_neutre_par_defaut(pond)
    check_9_aucun_poids_sur_les_barres_sans_label(pond)
    check_11_deterministe(dev)
    check_94_le_poids_d_APPRENTISSAGE_est_deflate_par_heure(pond)

    n = int(pd.Series(pond["issue"]).notna().sum())
    eff = exemples_effectifs(pond)
    erreur_type = 0.5 / np.sqrt(eff) * 100
    print()
    print("  ── LE CHIFFRE QUI GOUVERNE TOUS LES INTERVALLES DE CONFIANCE ──")
    print(f"    dev : {n:,} labels  ->  {eff:,.0f} exemples effectifs ({100 * eff / n:.0f} %)")
    print(f"    erreur-type sur un taux de réussite : {erreur_type:.2f} point")
    print(f"    mur du hasard (3 écarts-types)      : {50 + 3 * erreur_type:.1f} %")

    print("\n" + "=" * 78)
    print("LES 19 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_90_le_tilt_horaire_des_poids_est_MESURE() -> None:
    """
    ⚠️⚠️ Trouvé par la revue de la phase 8 (passe 2, 2026-09-20) : les poids d'unicité encodent
    l'HEURE d'un facteur **4,5** sur le dev (0,195 à 2h contre 0,878 à 10h), parce que les barres
    de 11h à 16h n'ont aucun label et qu'un label de 10h vit donc seul.

    Le calcul est juste ; c'est la **discordance** qui pose problème — le modèle serait entraîné
    pondéré et mesuré non pondéré, et recevrait un a priori d'heure par la porte de derrière.
    Ce check ne corrige rien : il rend le tilt **impossible à ignorer**.
    """
    rng = np.random.default_rng(3)
    n = 400
    dt = pd.date_range("2022-01-03 02:00", periods=n, freq="h")
    # Un poids volontairement lié à l'heure : 1,0 les heures paires, 0,2 les impaires.
    heures = np.asarray(pd.Series(dt).dt.hour, dtype=int)
    df = pd.DataFrame(
        {
            "DateTime": dt,
            "poids_echantillon": np.where(heures % 2 == 0, 1.0, 0.2)
            + rng.normal(scale=1e-6, size=n),
        }
    )
    profil = tilt_horaire(df)
    if len(profil) != len(np.unique(heures)):
        _echec(f"{len(profil)} heures rapportées pour {len(np.unique(heures))} présentes")
    paires = [profil[h] for h in profil.index if h % 2 == 0]
    impaires = [profil[h] for h in profil.index if h % 2 == 1]
    if not (min(paires) > 0.9 and max(impaires) < 0.3):
        _echec(
            f"le profil ne retrouve pas le tilt planté : paires {paires[:3]}, impaires {impaires[:3]}"
        )
    rapport = rapport_tilt_horaire(df)
    if abs(rapport - 5.0) > 0.01:
        _echec(f"rapport max/min = {rapport:.3f} au lieu de 5,0")
    # Restreint à un sous-ensemble d'heures, le rapport suit.
    if abs(rapport_tilt_horaire(df, (2, 4, 6)) - 1.0) > 0.01:
        _echec("sur des heures toutes paires, le rapport devrait valoir 1")
    # Des poids plats donnent un rapport de 1 : le diagnostic n'invente pas de tilt.
    plat = df.assign(poids_echantillon=np.full(n, 0.7))
    if abs(rapport_tilt_horaire(plat) - 1.0) > 1e-9:
        _echec("des poids constants produisent un tilt")
    _leve(
        lambda: tilt_horaire(df, colonne="inexistante"),
        ValueError,
        "une colonne absente",
        "absente",
    )
    _leve(
        lambda: tilt_horaire(df.assign(poids_echantillon=np.nan)),
        ValueError,
        "une colonne entièrement vide",
        "vide",
    )
    print(f"  90. tilt horaire mesuré : rapport {rapport:.2f} sur un tilt planté de 5,0  ✓")


def check_91_un_label_qui_se_resout_AVANT_son_entree_est_refuse() -> None:
    """
    ⚠️ Garde jamais vue refuser avant le 2026-09-20 (revue de la phase 8, passe 5).

    Un label dont la fin précède l'entrée donne une durée de vie **négative** : la somme préfixe
    de `unicite` porterait alors sur un intervalle à l'envers, et rendrait une unicité négative ou
    absurde — que le reste de la chaîne prendrait pour un poids valide.
    """
    n, horizon = 40, 6
    dt = pd.date_range("2022-01-03 02:00", periods=n, freq="h")
    # ⚠️ Les `horizon` DERNIÈRES barres n'ont PAS de label : leur fin tomberait après la fin du
    # panel, et c'est une AUTRE garde qui parlerait. Un décor invalide ne prouve rien de la garde
    # qu'on vise — mon premier jet faisait exactement ça, et la preuve accusait la mauvaise garde.
    issues: list[str | None] = ["haut"] * (n - horizon) + [None] * horizon
    fins = list(dt + pd.Timedelta(hours=horizon))
    sain = pd.DataFrame(
        {
            "DateTime": dt,
            "jour_ftmo": [d.date() for d in dt],
            "issue": issues,
            "label_entree_dt": [
                d if i is not None else pd.NaT for d, i in zip(dt, issues, strict=True)
            ],
            "label_fin_dt": [
                f if i is not None else pd.NaT for f, i in zip(fins, issues, strict=True)
            ],
        }
    )
    u = unicite(sain)
    if not np.all((u[np.isfinite(u)] > 0) & (u[np.isfinite(u)] <= 1)):
        _echec("un panel sain produit des unicités hors de ]0, 1]")
    # La fin RECULE avant l'entrée sur une seule barre.
    casse = sain.copy()
    # `.copy()` obligatoire : `to_numpy()` sur une Series rend parfois une vue en LECTURE SEULE.
    f2 = np.array(pd.to_datetime(casse["label_fin_dt"]).to_numpy(), copy=True)
    e2 = np.array(pd.to_datetime(casse["label_entree_dt"]).to_numpy(), copy=True)
    # La fin RECULE d'une barre AVANT l'entrée, sur une barre labellisée du milieu.
    f2[n // 2] = e2[n // 2] - np.timedelta64(1, "h")
    casse["label_fin_dt"] = f2
    _leve(
        lambda: unicite(casse),
        AssertionError,
        "un label qui se résout avant son entrée",
        "avant son entrée",
    )
    print("  91. un label dont la fin précède l'entrée est refusé — pas d'unicité négative  ✓")


def check_93_date_d_ENTREE_hors_panel_refusee() -> None:
    """
    La symétrique de `check_12`, qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    `check_12` couvrait la date de FIN ; la garde sur la date d'ENTRÉE n'était exercée par
    personne. Même mécanique, même silence : `searchsorted` rend un point d'insertion, jamais
    une absence, donc la fenêtre de vie du label serait décalée — unicité fausse, effectifs D4
    faux, intervalles de confiance faux.
    """
    df = _panel([(1, 3)], n=6)
    df["label_entree_dt"] = pd.to_datetime(df["label_entree_dt"])
    df.loc[0, "label_entree_dt"] = df["label_entree_dt"].iloc[0] + pd.Timedelta(minutes=30)
    _leve(
        lambda: concurrence(df),
        AssertionError,
        "date d'entrée absente du panel",
        "date d'entrée",
    )
    # anti-vacuité : le même panel, intact, passe
    if concurrence(_panel([(1, 3)], n=6)) is None:
        _echec("anti-vacuité : la concurrence ne se calcule plus sur un panel intact")
    print("  93. une date d'ENTRÉE absente du panel est refusée, jamais décalée en silence  ✓")


def check_94_le_poids_d_APPRENTISSAGE_est_deflate_par_heure(pond: pd.DataFrame) -> None:
    """
    **Le tilt horaire disparaît du poids d'apprentissage — et il reste quelque chose après.**

    Deux anti-vacuités, et ce sont elles qui portent la preuve. ① Le tilt BRUT doit encore valoir
    ~4,5 : si l'unicité cessait d'encoder l'heure, un tilt déflaté de 1,0 ne prouverait rien.
    ② Les poids déflatés ne doivent pas être tous égaux — une fonction qui rendrait `1,0` partout
    obtiendrait un tilt parfait en ayant détruit toute l'information. C'est le mode de panne le
    plus plausible de cette fonction, et le seul qui ne planterait jamais.
    """
    brut = rapport_tilt_horaire(pond, tuple(range(2, 11)))
    if not 4.0 < brut < 5.0:
        _echec(f"anti-vacuité : le tilt BRUT vaut {brut:.2f}, attendu ~4,5 — rien à déflater")

    deflate = pond.copy()
    deflate["poids_appris"] = poids_apprentissage(pond)
    profil = tilt_horaire(deflate, "poids_appris")
    ecart = float(np.abs(profil.to_numpy() - 1.0).max())
    if ecart > 1e-9:
        _echec(f"une heure ne pèse pas 1,0 après déflation : écart max {ecart:.2e}")

    v = np.asarray(deflate["poids_appris"], dtype=float)
    v = v[np.isfinite(v)]
    if float(v.std()) < 0.10:
        _echec(
            f"anti-vacuité : écart-type des poids déflatés {v.std():.4f} — la déflation a "
            "aplati les poids au lieu de retirer l'heure"
        )

    # L'ordre DANS une heure est intact : c'est toute l'information qu'on prétend garder.
    heures = np.asarray(pd.to_datetime(pd.Series(pond["DateTime"])).dt.hour, dtype=int)
    u = np.asarray(pond["unicite"], dtype=float)
    d = np.asarray(deflate["poids_appris"], dtype=float)
    for h in range(2, 11):
        m = (heures == h) & np.isfinite(u)
        if int(m.sum()) < 2:
            continue
        if not np.array_equal(np.argsort(np.argsort(u[m])), np.argsort(np.argsort(d[m]))):
            _echec(f"l'ordre des poids est modifié à l'intérieur de {h}h")

    print(
        f"  94. (8.8) le poids d'APPRENTISSAGE est déflaté : tilt {brut:.2f} -> 1,00, "
        f"écart-type conservé {v.std():.3f}, ordre intact dans chaque heure  ✓"
    )


def check_95_la_deflation_est_LOCALE_a_ce_qu_on_lui_donne() -> None:
    """
    **La déflation se calcule sur les lignes reçues, jamais sur un ailleurs.**

    C'est la propriété qui rend l'appel sur un fold de train honnête : les moyennes horaires du
    fold de test ne doivent pas entrer dans les poids d'entraînement. Rien ne le signalerait — les
    poids resteraient plausibles, et la fuite serait invisible jusqu'au hold-out.
    """
    t = list(pd.date_range("2024-06-04 02:00", periods=4, freq="h"))
    df = pd.DataFrame({"DateTime": t + t, "unicite": [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]})
    tout = poids_apprentissage(df)
    moitie = poids_apprentissage(pd.DataFrame(df.iloc[:4]).reset_index(drop=True))
    # Sur le sous-ensemble, chaque heure n'a qu'une ligne : chacune vaut donc exactement 1,0.
    if not np.allclose(moitie, 1.0):
        _echec(f"une heure à une seule ligne ne vaut pas 1,0 : {moitie}")
    # Sur le tout, la même ligne est comparée à sa voisine de la même heure : elle ne vaut plus 1.
    if np.allclose(tout[:4], 1.0):
        _echec("anti-vacuité : le fixture ne distingue pas les deux périmètres")
    if np.allclose(tout[:4], moitie):
        _echec(
            "la déflation d'un sous-ensemble donne le même poids que celle du tout : elle "
            "n'est pas locale, donc un fold de train verrait les moyennes du test"
        )
    print("  95. (8.8) la déflation est LOCALE aux lignes reçues — pas de fuite de fold  ✓")


def check_96_les_gardes_du_poids_d_apprentissage_MORDENT() -> None:
    """Les quatre entrées malades sont refusées, et le panel sain repasse."""
    t = list(pd.date_range("2024-06-04 02:00", periods=3, freq="h"))
    sain = pd.DataFrame({"DateTime": t, "unicite": [0.5, 0.25, 1.0]})
    _leve(
        lambda: poids_apprentissage(pd.DataFrame({"DateTime": t})),
        ValueError,
        "colonne d'unicité absente",
        "absente",
    )
    _leve(
        lambda: poids_apprentissage(pd.DataFrame({"unicite": [0.5, 0.25, 1.0]})),
        ValueError,
        "colonne DateTime absente",
        "DateTime",
    )
    _leve(
        lambda: poids_apprentissage(pd.DataFrame({"DateTime": t, "unicite": [0.5, -0.25, 1.0]})),
        ValueError,
        "unicité négative",
        "apprendre son contraire",
    )
    _leve(
        lambda: poids_apprentissage(
            pd.DataFrame({"DateTime": t, "unicite": [np.nan, np.nan, np.nan]})
        ),
        ValueError,
        "aucune unicité finie",
        "rien à pondérer",
    )
    if not np.all(np.isfinite(poids_apprentissage(sain))):
        _echec("anti-vacuité : le panel SAIN ne produit plus de poids")
    print("  96. (8.8) les 4 gardes du poids d'apprentissage mordent, le panel sain passe  ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
