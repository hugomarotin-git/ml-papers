"""
Preuve du label triple-barrière — tâches 2.2, 2.3 et 2.4.

Seize vérifications. C'est le premier module qui **décide** quelque chose : une erreur ici ne
casserait rien, elle apprendrait simplement au modèle la mauvaise cible. Aucune étape ultérieure
ne pourrait la rattraper, ni même la voir.

La plus importante est la n°1 : **l'absence de look-ahead**, testée en tronquant réellement les
données. Un label qui a vu le futur produit un modèle qui semble excellent et ne vaut rien.

Lancer :  .venv/bin/python src/sanity_check_label.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ftmo_cleaner import DATA_DIR, load_clean
from intraday_filter import filtrer_intraday
from label_triple_barriere import (
    ATR_PERIODE,
    GAINS,
    HORIZON_BARRES,
    ISSUE_BAS,
    ISSUE_DOUBLE,
    ISSUE_HAUT,
    ISSUE_TIMEOUT,
    K_ATR,
    RAISON_ATR,
    RAISON_CONTIGUITE,
    RAISON_FIN,
    construire_labels,
)
from periodes import restreindre_au_dev


def _echec(msg: str) -> None:
    raise AssertionError(f"ÉCHEC — {msg}")


#: Une séance FTMO démarre à 18:00 NY et traverse minuit. Les panels fabriqués doivent donc
#: partir de 18:00, sinon leur horizon franchit une frontière de journée et aucun label ne sort —
#: c'est exactement le piège que la règle C9 est censée poser, et il vaut mieux le connaître ici
#: que le découvrir sur un test qui échoue sans raison apparente.
DEBUT_SEANCE = pd.Timestamp("2024-06-03 18:00")


def _panel(ohlc: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """
    Panel fabriqué : une suite de barres horaires CONTIGUËS dans une seule journée FTMO.

    Les horodatages sont générés, pas écrits à la main : ça évite de fabriquer par
    inadvertance un trou ou un franchissement de séance qui ferait échouer le test pour une
    raison sans rapport avec ce qu'il teste.
    """
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", [DEBUT_SEANCE + pd.Timedelta(hours=i) for i in range(len(ohlc))])
    df["jour_ftmo"] = (pd.Series(df["DateTime"]) + pd.Timedelta(hours=7)).dt.date
    # Les barres étant générées contiguës, `gap_kind` vaut « contigu » partout sauf la première.
    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60
    df["gap_kind"] = np.where(ecart == 60, "contigu", "premiere_barre")
    return df


#: Barres d'amplitude constante servant à amorcer l'ATR : il converge vers ~9,4, donc des
#: barrières à environ +/- 14 autour d'un prix de 100.
#:
#: Il en faut `ATR_PERIODE + 1`, pas `ATR_PERIODE` : le True Range de la toute première barre
#: est indéfini (aucune clôture précédente), donc la première moyenne complète n'arrive qu'à
#: l'indice `ATR_PERIODE`. C'est la même raison qui produit les 14 `atr_indisponible` sur les
#: vraies données — une borne, pas une anomalie.
BASE_ATR = [(100.0, 105.0, 95.0, 100.0)] * (ATR_PERIODE + 1)
#: Indice de la barre candidate dans les panels fabriqués : le premier ATR valide.
T_CANDIDAT = ATR_PERIODE
#: Barres calmes qui ne franchissent rien, pour compléter un horizon.
CALME = (100.0, 101.0, 99.0, 100.0)


# ────────────────────────── 2.2 — les règles de construction ──────────────────────────


def check_2_entree_a_open_suivant(lab: pd.DataFrame) -> None:
    """L'entrée se fait à l'Open de t+1, jamais au Close de t (décision C5)."""
    avec = lab[pd.Series(lab["issue"]).notna()]
    ouverture = pd.Series(lab["Open"]).to_numpy()
    idx = np.asarray(avec.index, dtype=int)
    attendu = ouverture[idx + 1]
    obtenu = np.asarray(avec["prix_entree"], dtype=float)
    if not np.allclose(attendu, obtenu):
        n = int((~np.isclose(attendu, obtenu)).sum())
        _echec(f"{n} entrée(s) ne valent pas Open[t+1]")
    print(f"PASS — 2. entrée = Open[t+1] sur les {len(avec):,} labels")


def check_3_barrieres_symetriques(lab: pd.DataFrame) -> None:
    """RR 1:1 — le TP et le SL sont exactement à la même distance, égale à k x ATR."""
    avec = lab[pd.Series(lab["issue"]).notna()]
    e = np.asarray(avec["prix_entree"], float)
    haut = np.asarray(avec["barriere_haute"], float) - e
    bas = e - np.asarray(avec["barriere_basse"], float)
    if not np.allclose(haut, bas):
        _echec("les deux barrières ne sont pas à la même distance de l'entrée")
    if not np.allclose(haut, K_ATR * np.asarray(avec["atr"], float)):
        _echec(f"la distance ne vaut pas {K_ATR} x ATR")
    print(f"PASS — 3. barrières symétriques à {K_ATR} x ATR, RR 1:1")


def check_4_ordre_des_touches() -> None:
    """
    Sur un cas construit à la main : la barrière touchée **en premier** doit gagner, quelle que
    soit celle qui est touchée ensuite.
    """
    t = T_CANDIDAT
    haut_dabord = [
        *BASE_ATR,
        CALME,
        (100.0, 116.0, 99.0, 115.0),  # haute franchie
        (115.0, 116.0, 80.0, 85.0),  # basse ensuite
        CALME,
        CALME,
        CALME,
        CALME,
    ]
    lab = construire_labels(_panel(haut_dabord))
    if lab["issue"].iloc[t] != ISSUE_HAUT:
        _echec(f"haute touchée en premier mais issue = {lab['issue'].iloc[t]}")

    bas_dabord = [
        *BASE_ATR,
        CALME,
        (100.0, 101.0, 84.0, 85.0),  # basse franchie
        (85.0, 120.0, 84.0, 118.0),  # haute ensuite
        CALME,
        CALME,
        CALME,
        CALME,
    ]
    lab2 = construire_labels(_panel(bas_dabord))
    if lab2["issue"].iloc[t] != ISSUE_BAS:
        _echec(f"basse touchée en premier mais issue = {lab2['issue'].iloc[t]}")
    print("PASS — 4. la barrière touchée en premier l'emporte, dans les deux sens")


def check_5_convention_pessimiste() -> None:
    """
    **Le cas `double`.** Les deux barrières franchies sur la MÊME bougie : les deux directions
    perdent (décision C12).

    Une bougie ne dit pas dans quel ordre ses extrêmes ont été atteints. Trancher serait
    inventer ; on compte donc le stop des deux côtés — « ça peut avoir gagné, mais on met le
    stop au cas où ».
    """
    t = T_CANDIDAT
    barres = [
        *BASE_ATR,
        CALME,
        (100.0, 120.0, 80.0, 100.0),  # les DEUX dans la même bougie
        CALME,
        CALME,
        CALME,
        CALME,
        CALME,
    ]
    lab = construire_labels(_panel(barres))
    if lab["issue"].iloc[t] != ISSUE_DOUBLE:
        _echec(f"issue = {lab['issue'].iloc[t]} au lieu de « double »")
    if (lab["label_long"].iloc[t], lab["label_short"].iloc[t]) != (-1, -1):
        _echec("le cas double ne fait pas perdre les DEUX directions")
    print("PASS — 5. convention pessimiste : les deux barrières sur une bougie -> les deux perdent")


def check_6_timeout() -> None:
    """Aucune barrière franchie dans l'horizon : issue `timeout`, aucune direction ne gagne."""
    t = T_CANDIDAT
    lab = construire_labels(_panel([*BASE_ATR, *([CALME] * 7)]))
    if lab["issue"].iloc[t] != ISSUE_TIMEOUT:
        _echec(f"issue = {lab['issue'].iloc[t]} au lieu de « timeout »")
    if (lab["label_long"].iloc[t], lab["label_short"].iloc[t]) != (0, 0):
        _echec("un timeout ne devrait faire ni gagner ni perdre")
    print("PASS — 6. timeout correctement détecté, aucune direction ne gagne")


def check_7_contiguite(lab: pd.DataFrame) -> None:
    """
    Les 7 barres d'un label — candidat, entrée, horizon — doivent **se suivre** (décision C9).

    ⚠️ Ce contrôle remplace une version fausse. L'ancienne vérifiait que l'entrée et la fin
    étaient dans la même journée FTMO, **sans vérifier que le candidat y était aussi**. Un
    candidat à la clôture d'une séance recevait donc une entrée à l'ouverture de la suivante :
    décider vendredi 16 h pour entrer dimanche 17 h. 1 419 labels étaient dans ce cas.

    On vérifie ici la contiguïté directement, pas l'appartenance à une séance : c'est plus fort,
    et ça attrape en plus les trous internes à une séance.
    """
    avec = lab[pd.Series(lab["issue"]).notna()]
    contigu = np.asarray(pd.Series(lab["gap_kind"]) == "contigu", dtype=bool)
    idx = np.asarray(avec.index, dtype=int)
    for decalage in range(1, HORIZON_BARRES + 1):
        if not bool(contigu[idx + decalage].all()):
            n = int((~contigu[idx + decalage]).sum())
            _echec(f"{n} label(s) ont une rupture à la barre +{decalage}")

    # Et la conséquence : candidat, entrée et fin partagent forcément la même séance.
    jour = np.asarray(pd.Series(lab["jour_ftmo"]), dtype=object)
    if not bool((jour[idx] == jour[idx + HORIZON_BARRES]).all()):
        _echec("des labels contigus franchissent pourtant une séance — définition incohérente")
    print(f"PASS — 7. les {len(avec):,} labels reposent sur 7 barres contiguës, même séance")


def check_8_duree_impliquee_par_la_contiguite(lab: pd.DataFrame) -> None:
    """
    La contiguïté impose mécaniquement la durée du trade.

    ⚠️ **Attention au comptage, c'est une source de confusion.** Un horodatage marque le
    **début** d'une barre, pas sa fin — on l'a prouvé indirectement en montrant que l'ouverture
    du cash à 09h30 apparaît dans la barre de 09h.

    Donc pour 6 barres consécutives :

    - l'écart entre le **premier et le dernier horodatage** vaut 5 h (6 barres, 5 intervalles) ;
    - mais **le trade vit 6 h** : il entre à l'ouverture de la première et peut se clôturer
      jusqu'à la fin de la dernière.

    C'est la seconde valeur qui a un sens de trading. La première n'est qu'une conséquence de
    la convention d'horodatage.

    Ce contrôle a remplacé une borne explicite sur la durée, que la contiguïté avait rendue
    inatteignable — le test de mutation a montré qu'on pouvait la supprimer sans qu'aucune
    preuve ne bronche.
    """
    avec = lab[pd.Series(lab["issue"]).notna()]
    t = pd.Series(lab["DateTime"]).to_numpy()
    idx = np.asarray(avec.index, dtype=int)
    ecart_horodatages = (t[idx + HORIZON_BARRES] - t[idx + 1]) / np.timedelta64(1, "h")
    attendu = float(HORIZON_BARRES - 1)
    if not bool((ecart_horodatages == attendu).all()):
        _echec(
            f"des horizons n'ont pas {attendu:.0f} h d'écart entre horodatages : "
            f"min {ecart_horodatages.min()}, max {ecart_horodatages.max()}"
        )
    print(
        f"PASS — 8. {HORIZON_BARRES} barres contiguës : {attendu:.0f} h entre le premier et le "
        f"dernier horodatage, donc **{HORIZON_BARRES} h de vie réelle du trade**"
    )


def check_9_labels_derives_coherents(lab: pd.DataFrame) -> None:
    """`label_long` et `label_short` doivent découler mécaniquement de `issue`."""
    avec = lab[pd.Series(lab["issue"]).notna()]
    for issue, (attendu_l, attendu_s) in GAINS.items():
        m = np.asarray(avec["issue"] == issue)
        if not m.any():
            continue
        if not bool((np.asarray(avec["label_long"])[m] == attendu_l).all()):
            _echec(f"label_long faux pour l'issue « {issue} »")
        if not bool((np.asarray(avec["label_short"])[m] == attendu_s).all()):
            _echec(f"label_short faux pour l'issue « {issue} »")
    print("PASS — 9. `label_long` et `label_short` cohérents avec `issue` sur les 4 issues")


def check_14_trou_refuse_sur_cas_fabrique() -> None:
    """
    **Le test du bug du 2026-08-08**, sur un cas fabriqué.

    Un trou au milieu de la séance doit faire refuser le label, même si toutes les barres
    restent dans la même journée FTMO. C'est exactement ce que l'ancienne règle laissait
    passer : elle regardait la séance, pas la continuité.

    Fabriqué plutôt que cherché dans les données, pour la même raison que d'habitude — un cas
    qu'on ne rencontre pas aujourd'hui doit quand même être exercé.
    """
    heures = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16]
    ohlc = [*BASE_ATR, *([CALME] * (len(heures) - len(BASE_ATR)))]
    depart = pd.Timestamp("2024-06-03 17:00")
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", [depart + pd.Timedelta(hours=(h - 17) % 24) for h in heures])
    df["jour_ftmo"] = (pd.Series(df["DateTime"]) + pd.Timedelta(hours=7)).dt.date
    ecart = pd.Series(df["DateTime"]).diff().dt.total_seconds() / 60
    df["gap_kind"] = np.where(ecart == 60, "contigu", "trou")

    if pd.Series(df["jour_ftmo"]).nunique() != 1:
        _echec("le panel fabriqué enjambe plusieurs séances — il ne teste pas ce qu'on croit")

    lab = construire_labels(df)
    if lab["raison_sans_label"].iloc[T_CANDIDAT] != RAISON_CONTIGUITE:
        _echec(
            "un horizon troué a été labellisé "
            f"(raison = {lab['raison_sans_label'].iloc[T_CANDIDAT]!r}, "
            f"issue = {lab['issue'].iloc[T_CANDIDAT]!r})"
        )
    print(
        "PASS — 14. un TROU au milieu d'une séance fait refuser le label, séance identique ou non"
    )


def check_16_gap_kind_perime_refuse() -> None:
    """
    **La colonne `gap_kind` peut mentir, et le module doit s'en apercevoir.**

    Elle est écrite par le nettoyeur puis jamais recalculée : si des lignes sont retirées
    ensuite (filtre intraday, découpe par heures en phase 8), une barre peut rester marquée
    « contigu » vers une barre qui n'existe plus. La relire telle quelle ferait enjamber la
    coupure à un label — exactement la classe de bug C9. Le module doit refuser un panel dont
    les horodatages contredisent la colonne, jamais labelliser dessus.
    """
    heures = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16]
    ohlc = [*BASE_ATR, *([CALME] * (len(heures) - len(BASE_ATR)))]
    depart = pd.Timestamp("2024-06-03 17:00")
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"])
    df.insert(0, "DateTime", [depart + pd.Timedelta(hours=(h - 17) % 24) for h in heures])
    df["jour_ftmo"] = (pd.Series(df["DateTime"]) + pd.Timedelta(hours=7)).dt.date
    # Le mensonge : la colonne prétend que tout est contigu, alors que les horodatages
    # sautent de 11 h à 15 h — c'est ce qui arrive quand des lignes sont retirées APRÈS
    # le calcul de `gap_kind`.
    df["gap_kind"] = "contigu"

    try:
        construire_labels(df)
    except AssertionError as e:
        if "périmée" not in str(e):
            _echec(f"panel menteur refusé mais pour une AUTRE raison : {e}")
        print("PASS — 16. un gap_kind périmé (contigu contredit par les horodatages) est refusé")
        return
    _echec(
        "un panel dont gap_kind ment (« contigu » sur un trou de 4 h) a été labellisé : la "
        "contiguïté C9 a été lue sur une colonne périmée sans re-vérification"
    )


def check_15_candidat_et_entree_contigus(lab: pd.DataFrame) -> None:
    """
    **Le cœur du bug.** L'entrée doit suivre le candidat de 60 minutes exactement.

    Si on attend plus, le signal qui a servi à décider a vieilli — et dans le pire cas on
    traverse un week-end. Ce contrôle isole cette condition, indépendamment du reste de
    l'horizon, parce que c'est elle qui manquait.
    """
    avec = lab[pd.Series(lab["issue"]).notna()]
    horodatage = pd.Series(lab["DateTime"]).to_numpy()
    idx = np.asarray(avec.index, dtype=int)
    minutes = (horodatage[idx + 1] - horodatage[idx]) / np.timedelta64(1, "m")
    if not bool((minutes == 60).all()):
        pire = float(np.max(minutes))
        _echec(
            f"{int((minutes != 60).sum())} label(s) entrent plus tard que +60 min (jusqu'à {pire:.0f} min)"
        )
    print(f"PASS — 15. l'entrée suit le candidat de 60 min exactement, sur {len(avec):,} labels")


# ────────────────────────── 2.3 — l'équilibre des classes ──────────────────────────


def check_10_equilibre_des_classes(lab: pd.DataFrame) -> None:
    """
    Aucune classe ne doit écraser les autres (décision C10).

    Un label à 95 % d'une seule valeur ne s'apprend pas : le modèle prédirait cette valeur
    partout et aurait raison. Le problème serait alors dans le **cadrage** — barrières trop
    larges, horizon trop court — pas dans le modèle.
    """
    avec = pd.Series(lab["issue"]).dropna()
    parts = {str(k): float(v) for k, v in avec.value_counts(normalize=True).items()}
    dominante = max(parts, key=lambda k: parts[k])
    if parts[dominante] > 0.60:
        _echec(f"classe « {dominante} » à {100 * parts[dominante]:.0f} % — le label est dégénéré")
    if parts.get(ISSUE_HAUT, 0) < 0.10 or parts.get(ISSUE_BAS, 0) < 0.10:
        _echec("une direction est quasi absente : cadrage à revoir")
    detail = " | ".join(
        f"{k} {100 * v:.1f} %" for k, v in sorted(parts.items(), key=lambda x: -x[1])
    )
    print(f"PASS — 10. classes exploitables : {detail}")


def check_11_pas_de_biais_directionnel_massif(lab: pd.DataFrame) -> None:
    """
    `haut` et `bas` doivent rester comparables.

    Un déséquilibre marqué signifierait que le label encode surtout la dérive de l'instrument —
    et le modèle apprendrait « le NASDAQ monte » plutôt qu'un edge. C'est précisément ce que le
    twin-null est là pour démasquer plus tard ; autant le voir dès le label.
    """
    avec = pd.Series(lab["issue"]).dropna()
    h = int((avec == ISSUE_HAUT).sum())
    b = int((avec == ISSUE_BAS).sum())
    ratio = h / max(b, 1)
    if not 0.7 <= ratio <= 1.43:
        _echec(f"déséquilibre haut/bas marqué : {h:,} contre {b:,} (ratio {ratio:.2f})")
    print(f"PASS — 11. haut/bas équilibrés : {h:,} contre {b:,} (ratio {ratio:.2f})")


# ────────────────────────── 2.4 — les barres sans label ──────────────────────────


def check_12_toute_barre_sans_label_a_une_raison(lab: pd.DataFrame) -> None:
    """
    Chaque barre non labellisée doit porter une raison connue, et chaque raison doit être une
    borne structurelle — pas une anomalie.
    """
    sans = pd.Series(lab["raison_sans_label"])
    avec = pd.Series(lab["issue"])
    if bool((sans.isna() & avec.isna()).any()):
        _echec(f"{int((sans.isna() & avec.isna()).sum())} barre(s) sans label ET sans raison")
    if bool((sans.notna() & avec.notna()).any()):
        _echec("des barres ont à la fois un label et une raison de ne pas en avoir")
    connues = {RAISON_ATR, RAISON_CONTIGUITE, RAISON_FIN}
    trouvees = set(sans.dropna().unique())
    if not trouvees <= connues:
        _echec(f"raison inconnue : {trouvees - connues}")
    detail = " | ".join(f"{k} {int(v):,}" for k, v in sans.value_counts().items())
    print(f"PASS — 12. toute barre sans label a une raison connue : {detail}")


def check_13_deterministe(df: pd.DataFrame) -> None:
    """
    Deux constructions successives doivent donner exactement le même résultat.

    Le sous-échantillon est découpé sur des **journées FTMO entières**. Découper sur un nombre
    arbitraire de lignes tronquerait une séance en plein milieu, et la garde
    `assert_aucune_barre_journaliere` le refuserait à juste titre — elle ne peut pas distinguer
    une séance réellement courte d'une séance coupée par un `head(n)`.
    """
    jours = list(dict.fromkeys(pd.Series(df["jour_ftmo"])))[:130]
    extrait = pd.DataFrame(df[np.isin(np.asarray(df["jour_ftmo"], dtype=object), jours)])
    a = construire_labels(extrait)
    b = construire_labels(extrait)
    if not pd.Series(a["issue"]).equals(pd.Series(b["issue"])):
        _echec("la construction du label n'est pas déterministe")
    print(
        f"PASS — 13. construction déterministe ({len(extrait):,} barres, {len(jours)} séances entières)"
    )


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


def check_17_k_atr_non_fini_ou_nul_refuse() -> None:
    """
    **Piège n°8, audit du 2026-08-18.** `k_atr` n'était validé nulle part.

    Un `NaN` donne des barrières `NaN`. Or toute comparaison avec `NaN` est fausse : **aucune
    barrière ne serait jamais franchie**, et 100 % des barres sortiraient en `timeout`. Le fichier
    de labels ressemblerait alors à « ce marché n'a pas de structure » — la conclusion la plus
    coûteuse qu'on puisse tirer, et elle viendrait d'un bug, pas du marché.

    À 0, les deux barrières collent au prix d'entrée : tout devient `double`. En négatif, la
    barrière haute passe **sous** la basse.
    """
    df = _panel([*BASE_ATR, *([CALME] * 7)])
    for valeur, quoi in (
        (float("nan"), "NaN"),
        (float("inf"), "inf"),
        (0.0, "nul"),
        (-1.5, "négatif"),
    ):
        _leve(
            lambda v=valeur: construire_labels(df, k_atr=v),
            ValueError,
            f"un k_atr {quoi}",
            contient="k_atr",
        )
    print(" 17. un k_atr NaN, infini, nul ou négatif est REFUSÉ (piège n°8)  ✓")


def main() -> None:
    print("\n" + "=" * 78)
    print("PREUVE DU LABEL TRIPLE-BARRIÈRE — tâches 2.2, 2.3 et 2.4")
    print("=" * 78 + "\n")

    df = restreindre_au_dev(
        filtrer_intraday(load_clean(DATA_DIR / "v2" / "nas100_h1_clean.csv"))[0]
    )
    lab = construire_labels(df)

    check_2_entree_a_open_suivant(lab)
    check_3_barrieres_symetriques(lab)
    check_4_ordre_des_touches()
    check_5_convention_pessimiste()
    check_6_timeout()
    check_7_contiguite(lab)
    check_8_duree_impliquee_par_la_contiguite(lab)
    check_9_labels_derives_coherents(lab)
    check_14_trou_refuse_sur_cas_fabrique()
    check_15_candidat_et_entree_contigus(lab)
    check_16_gap_kind_perime_refuse()
    check_17_k_atr_non_fini_ou_nul_refuse()
    check_93_un_panel_SANS_gap_kind_est_refuse()
    print()
    check_10_equilibre_des_classes(lab)
    check_11_pas_de_biais_directionnel_massif(lab)
    print()
    check_12_toute_barre_sans_label_a_une_raison(lab)
    check_13_deterministe(df)

    print("\n" + "=" * 78)
    print("LES 17 PREUVES PASSENT.")
    print("=" * 78 + "\n")


def check_93_un_panel_SANS_gap_kind_est_refuse() -> None:
    """
    La garde qu'aucune preuve ne voyait refuser — audit du 2026-09-21.

    `gap_kind` sert à vérifier la **contiguïté** des 7 barres (C9) — la règle dont l'absence a
    failli tuer le projet le 2026-08-08. Un panel sans cette colonne ne peut pas être labellisé.
    """
    df = pd.DataFrame(
        {
            "DateTime": pd.date_range("2021-03-01 18:00", periods=5, freq="h"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "jour_ftmo": pd.Timestamp("2021-03-02").date(),
        }
    )
    _leve(
        lambda: construire_labels(df),
        ValueError,
        "panel sans gap_kind",
        "colonnes absentes",
    )
    print(
        "  93. un panel sans `gap_kind` est refusé — la contiguïté C9 ne serait pas vérifiable  ✓"
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n{e}\n", file=sys.stderr)
        raise SystemExit(1) from e
