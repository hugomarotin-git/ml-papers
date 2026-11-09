"""
Poids d'échantillon — tâche 3.1.

Corrige le fait que **les labels se chevauchent**. Celui de 10 h regarde les 6 heures suivantes,
celui de 11 h regarde presque les mêmes : ils ne décrivent pas deux événements différents, mais
le même bout de marché vu depuis deux points de départ. Méthode de référence : López de Prado,
*Advances in Financial Machine Learning*, chapitre 4.

Sans correction, deux choses se cassent — et la seconde ne se voit nulle part.

**Côté statistique.** On croit disposer de 25 302 observations ; il y en a beaucoup moins en
information. Tout intervalle de confiance calculé sur le nombre brut est trop étroit, et on se
convainc d'un edge qui n'existe pas.

**Côté apprentissage.** Une longue période calme produit des dizaines de labels qui disent la
même chose ; un moment bref et décisif n'en produit qu'un ou deux. Un modèle entraîné sans
poids compte la période calme des dizaines de fois — il apprend surtout ce qui se répète.

────────────────────────────────────────────────────────────────────────────────────────
Les trois notions, dans l'ordre
────────────────────────────────────────────────────────────────────────────────────────
1. **Concurrence** : à chaque barre, combien de labels sont vivants simultanément.
2. **Unicité** d'un label : la moyenne de `1 / concurrence` sur sa durée de vie. Vaut 1,0 pour
   un label qui n'a jamais partagé un instant avec un autre, moins sinon.
3. **Poids** = unicité (fois la décroissance temporelle si on l'active, neutre par défaut).

La somme des unicités donne le **nombre d'exemples effectifs** — le seul chiffre qui doive
entrer dans un intervalle de confiance ou dans un dimensionnement de CPCV.

────────────────────────────────────────────────────────────────────────────────────────
La durée de vie s'arrête à la RÉSOLUTION, pas à la fin de l'horizon
────────────────────────────────────────────────────────────────────────────────────────
Un trade qui touche sa barrière à la deuxième barre libère la place : il ne doit pas peser
comme un timeout qui occupe les six. Le module lit donc `label_fin_dt`, produit par le label.

⚠️ Il **ne recalcule pas** le label (décision D3). Dupliquer la boucle de résolution des
barrières créerait deux implémentations qui peuvent diverger en silence — le genre d'écart
qu'on ne découvre que des mois plus tard. Ici on vérifie la cohérence de ce qui arrive, sans
le refaire.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ftmo_cleaner import DATA_DIR
from label_triple_barriere import load_labelle

#: Décroissance temporelle, au sens d'AFML 4.5.2.
#:
#: `1.0` = **neutre**, aucune décroissance : c'est le défaut (décision D2). Une valeur dans
#: `]0, 1[` fait décroître linéairement le poids des observations anciennes jusqu'à `c` ; une
#: valeur négative annule complètement la partie la plus ancienne.
#:
#: Codé mais désactivé : disponible sans être imposé. L'activer serait une décision à justifier,
#: donc un essai à compter.
DECROISSANCE_NEUTRE: float = 1.0

DEFAULT_IN: Path = DATA_DIR / "v2" / "nas100_h1_labelle.csv"
DEFAULT_OUT: Path = DATA_DIR / "v2" / "nas100_h1_pondere.csv"


def _bornes_des_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Indices de début et de fin de vie de chaque label, dans le repère des barres.

    Vérifie au passage la cohérence de ce que le label a produit — sans le recalculer.
    """
    manquantes = {"DateTime", "label_entree_dt", "label_fin_dt", "issue"} - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Colonnes absentes : {sorted(manquantes)}. Ce module attend la sortie de "
            "`label_triple_barriere.construire_labels`."
        )

    horodatage = np.asarray(pd.to_datetime(pd.Series(df["DateTime"])), dtype="datetime64[ns]")
    a_un_label = np.asarray(pd.Series(df["issue"]).notna())
    debut_dt = np.asarray(pd.to_datetime(pd.Series(df["label_entree_dt"])), dtype="datetime64[ns]")
    fin_dt = np.asarray(pd.to_datetime(pd.Series(df["label_fin_dt"])), dtype="datetime64[ns]")

    debut = np.searchsorted(horodatage, debut_dt[a_un_label])
    fin = np.searchsorted(horodatage, fin_dt[a_un_label])

    if bool((fin < debut).any()):
        raise AssertionError("un label se résout avant son entrée — sortie du label incohérente")
    if bool((horodatage[debut] != debut_dt[a_un_label]).any()):
        raise AssertionError("une date d'entrée de label ne correspond à aucune barre")
    # La borne de FIN se vérifie comme celle de début : `searchsorted` ne signale jamais une
    # date absente, il renvoie un point d'insertion. Sans cette garde, un panel où la barre de
    # résolution manque (découpe par heures en phase 8, par exemple) décalerait la fenêtre de
    # vie du label EN SILENCE — concurrence et unicité fausses, donc effectifs et IC faux.
    if bool((fin >= len(horodatage)).any()) or bool((horodatage[fin] != fin_dt[a_un_label]).any()):
        raise AssertionError("une date de FIN de label ne correspond à aucune barre")
    return a_un_label, debut, fin


def concurrence(df: pd.DataFrame) -> np.ndarray:
    """
    Nombre de labels vivants à chaque barre.

    Calculé par un tableau de différences puis une somme cumulée plutôt que par une double
    boucle : sur 25 000 labels la version naïve serait inutilement lente, et surtout plus
    difficile à relire.
    """
    _, debut, fin = _bornes_des_labels(df)
    n = len(df)
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, debut, 1)
    np.add.at(delta, fin + 1, -1)
    return np.cumsum(delta)[:n]


def unicite(df: pd.DataFrame) -> np.ndarray:
    """
    Unicité moyenne de chaque label — dans `]0, 1]`, une valeur par barre labellisée.

    Moyenne de `1 / concurrence` sur la durée de vie du label. Calculée par somme préfixe,
    donc en un seul passage quelle que soit la longueur des horizons.

    ⚠️⚠️ **La durée de vie part de `label_entree_dt` (t+1), PAS de la barre de décision — et c'est
    VOULU.** Signalé comme une divergence par la revue de la phase 8 (passe 4, 2026-09-20), puis
    écarté après vérification : `cpcv._dates_purgees` part bien de la barre de décision, mais les
    deux répondent à des questions **différentes**.

    - **Ici, l'unicité mesure le chevauchement des LABELS** : ce qui détermine la valeur d'un label,
      c'est le chemin de prix entre l'**entrée** et la sortie. La barre de décision ne contribue
      pas à cette valeur — elle porte les features, pas l'issue.
    - **Là-bas, la purge protège d'une fuite d'INFORMATION** : une obs de train dont le label finit
      sur la barre où une obs de test lit ses features partage de l'information avec elle. Partir
      de la décision y est délibérément **plus conservateur**.

    Mesuré : l'effectif D4 vaut **4 045** avec la convention d'entrée contre **2 400** avec celle
    de décision. Aligner les deux **rétrécirait à tort** les intervalles de confiance dans un sens,
    ou les élargirait à tort dans l'autre. **Ne pas « harmoniser » ces deux conventions.**
    """
    a_un_label, debut, fin = _bornes_des_labels(df)
    conc = concurrence(df).astype(float)
    inverse = np.divide(1.0, conc, out=np.zeros_like(conc), where=conc > 0)
    prefixe = np.concatenate(([0.0], np.cumsum(inverse)))
    somme = prefixe[fin + 1] - prefixe[debut]
    longueur = (fin - debut + 1).astype(float)

    out = np.full(len(df), np.nan)
    out[a_un_label] = somme / longueur
    return out


def decroissance_temporelle(unicites: np.ndarray, c: float = DECROISSANCE_NEUTRE) -> np.ndarray:
    """
    Facteur de récence (AFML 4.5.2), appliqué sur l'unicité **cumulée** et non sur le temps
    calendaire.

    La nuance est volontaire : ce qui doit décroître, c'est le poids d'une observation en
    fonction de la quantité d'information vue depuis, pas en fonction du nombre de jours
    écoulés. Deux mois de marché calme apportent moins d'information qu'une semaine agitée.

    `c = 1` renvoie exactement 1 partout — le défaut, strictement neutre.
    """
    # ⚠️ **`c` n'était validé nulle part — corrigé le 2026-08-18** (piège n°8, audit du 18/08).
    # Un `NaN` traversait tout : `c == 1.0` est False, puis `c >= 0` est False aussi, donc la
    # branche « troncature » était prise et `pente` valait `NaN`. Résultat : des poids
    # d'échantillon **tous NaN**, sur lesquels un entraînement pondéré ne pèse plus rien —
    # et ça ne plante pas, ça donne un modèle entraîné à plat sans que rien ne le dise.
    # `c` est un réglage de recherche (D2) : il arrivera un jour d'un balayage programmatique.
    if not math.isfinite(c):
        raise ValueError(
            f"c (décroissance temporelle) doit être un nombre fini, reçu {c!r}. Un NaN produirait "
            "des poids d'échantillon tous NaN, sans rien faire planter."
        )
    if not c > -1.0:
        raise ValueError(
            f"c doit être > -1, reçu {c!r}. À -1 exactement le calcul de la pente divise par "
            "zéro ; en dessous, la pente change de signe et tous les poids sont écrasés à 0 par "
            "le `clip` — un jeu d'entraînement entièrement neutralisé, en silence."
        )
    valides = ~np.isnan(unicites)
    if c == 1.0:
        return np.where(valides, 1.0, np.nan)

    cumul = np.cumsum(np.where(valides, unicites, 0.0))
    total = float(cumul[-1]) if len(cumul) else 0.0
    if total <= 0:
        return np.where(valides, 1.0, np.nan)

    pente = (1.0 - c) / total if c >= 0 else 1.0 / ((c + 1.0) * total)
    constante = 1.0 - pente * total
    facteur = np.clip(constante + pente * cumul, 0.0, None)
    return np.where(valides, facteur, np.nan)


def calculer_poids(df: pd.DataFrame, c: float = DECROISSANCE_NEUTRE) -> pd.DataFrame:
    """Ajoute `concurrence`, `unicite`, `facteur_recence` et `poids_echantillon`."""
    out = df.copy()
    out["concurrence"] = concurrence(df)
    u = unicite(df)
    out["unicite"] = u
    f = decroissance_temporelle(u, c)
    out["facteur_recence"] = f
    out["poids_echantillon"] = u * f
    return out


def poids_apprentissage(df: pd.DataFrame, colonne: str = "unicite") -> np.ndarray:
    """
    **Le poids d'APPRENTISSAGE — l'unicité déflatée par heure.** Tâche 8.8, révise I17.

    `poids_echantillon` est le poids de **MESURE** : il porte l'effectif D4, les intervalles de
    confiance et le dimensionnement du CPCV, il est juste, et il ne bouge pas. Celui-ci est le
    poids qu'on donne au **modèle**, et il répond à un défaut que l'autre n'a pas à corriger :
    l'unicité encode l'**heure** d'un facteur 4,5, et un modèle entraîné dessus recevrait un
    a priori d'heure par la porte de derrière — exactement ce que I16 refuse d'exposer comme
    feature, et I14 de laisser candidater.

    ────────────────────────────────────────────────────────────────────────────────────────
    Pourquoi une déflation, et pas un autre périmètre de calcul
    ────────────────────────────────────────────────────────────────────────────────────────
    Mesuré le 2026-09-22 sur le dev (9 027 labels candidats), quatre constructions :

        (a) concurrence contre tous les labels ......... rapport max/min  4,51
        (b1) concurrence contre les seuls candidats .... 3,20  et le profil devient une CUVETTE
        (b2) moyenne sur les seules barres candidates .. 4,26  et DÉGÉNÉRÉE : nan à 10h
        (c) unicité déflatée par heure ................. 1,00

    ⚠️ **Aucun changement de périmètre ne fait tomber le tilt** : un label de 10h vit seul de 11h
    à 16h *quel que soit* le calcul, parce que C9 interdit qu'un label y soit posé. Le tilt est une
    propriété de la **structure de la séance**. Seule une déflation explicite peut l'enlever —
    le même principe que `profil_horaire` applique déjà aux features.

    ⚠️ **Le prix, mesuré et assumé** : la déflation jette **63 %** de la variance des poids — celle
    qui est *entre* les heures — et conserve les **37 %** qui départagent des labels comparables.
    Ce n'est pas un aplatissement : à 8h, un label pèse 0,167 au p10 contre 1,000 au p90, soit un
    rapport de **1 à 6 à heure égale**, et c'est exactement la correction de chevauchement que les
    sample weights doivent porter.

    ────────────────────────────────────────────────────────────────────────────────────────
    ⚠️⚠️ Rien n'est stocké, et c'est une garantie, pas une paresse
    ────────────────────────────────────────────────────────────────────────────────────────
    La déflation se calcule sur **les lignes qu'on lui donne**. Il faut donc l'appeler sur le
    **fold d'entraînement**, jamais sur le panel entier : les moyennes horaires d'un fold de test
    n'ont rien à faire dans les poids d'un fold de train.

    En ne l'écrivant dans aucune colonne et dans aucun CSV, on rend cette faute **impossible par
    construction** plutôt que recommandée par un commentaire — même raison que G5 pour la barre du
    null. `check_95` vérifie cette localité, et sa mutation la casse.

    Une heure qui ne porte qu'un seul label rend `1,0` pour ce label : c'est la bonne réponse —
    sans voisin dans son heure, il n'y a rien à départager — et non un silence à corriger.
    """
    if colonne not in df.columns:
        raise ValueError(
            f"Colonne « {colonne} » absente : il n'y a pas d'unicité à déflater. Entraîner sur "
            "des poids inventés serait pire que sur aucun poids."
        )
    if "DateTime" not in df.columns:
        raise ValueError("Colonne DateTime absente : la déflation horaire n'a pas d'heure.")
    u = np.asarray(pd.Series(df[colonne]).to_numpy(), dtype=float)
    heures = np.asarray(pd.to_datetime(pd.Series(df["DateTime"])).dt.hour, dtype=int)
    # `NaN` = pas de label sur cette barre, c'est la convention du module. Tout le reste doit
    # être strictement positif : un poids nul retirerait l'exemple en silence, un poids négatif
    # ferait APPRENDRE L'INVERSE de l'exemple sans que rien ne plante.
    porte = np.isfinite(u)
    if not bool(porte.any()):
        raise ValueError(f"Aucune unicité finie dans « {colonne} » : rien à pondérer.")
    if bool((u[porte] <= 0).any()):
        raise ValueError(
            f"{int((u[porte] <= 0).sum())} unicité(s) nulle(s) ou négative(s) : un poids nul "
            "retire l'exemple en silence, un poids négatif fait apprendre son contraire."
        )
    out = np.full(len(df), np.nan)
    for h in np.unique(heures[porte]):
        dans_l_heure = porte & (heures == h)
        out[dans_l_heure] = u[dans_l_heure] / float(u[dans_l_heure].mean())
    return out


def tilt_horaire(df: pd.DataFrame, colonne: str = "poids_echantillon") -> pd.Series:
    """
    Le poids moyen par heure — et il n'est PAS plat, c'est le point.

    ⚠️⚠️ **Trouvé le 2026-09-20 par la revue de la phase 8 (passe 2).** L'unicité se calcule
    contre **toutes** les barres labellisées. Or les barres de 11h à 16h n'ont **aucun label**
    (contiguïté C9), donc un label posé à 10h vit de 11h à 16h avec une concurrence qui retombe à
    1, pendant qu'un label posé à 2h partage toute sa vie avec cinq voisins. Mesuré sur le dev,
    dans la fenêtre de décision 2h-10h :

        2h 0,195 · 3h 0,229 · 4h 0,242 · 5h 0,249 · 6h 0,261
        7h 0,304 · 8h 0,464 · 9h 0,781 · 10h 0,878      → **rapport 4,5 entre 10h et 2h**

    **Le calcul est juste** : un label de 10h porte réellement plus d'information neuve. Ce qui
    pose problème est la **DISCORDANCE** — un modèle entraîné pondéré apprend 4,5 fois plus de
    l'ouverture américaine, alors que `selection_i3.mesurer_critere` compte chaque trade à 1 et
    que la règle top-2 choisit sur les neuf heures. Et surtout, le modèle reçoit ainsi un **a
    priori d'heure par la porte de derrière**, exactement ce que I16 a refusé d'exposer comme
    feature.

    ⚠️ **Décision ouverte (phase 9)** : entraîner sans pondération dans la fenêtre, recalculer
    l'unicité contre les seuls candidats de la fenêtre, ou pondérer aussi le critère. Aucune n'est
    gratuite — décision à prendre en phase 9.
    """
    if colonne not in df.columns:
        raise ValueError(f"Colonne « {colonne} » absente : rien à mesurer.")
    heures = np.asarray(pd.to_datetime(pd.Series(df["DateTime"])).dt.hour, dtype=int)
    valeurs = np.asarray(pd.to_numeric(pd.Series(df[colonne]), errors="coerce"), dtype=float)
    garde = np.isfinite(valeurs)
    if not bool(garde.any()):
        raise ValueError(f"Colonne « {colonne} » entièrement vide.")
    moyennes = {
        int(h): float(valeurs[garde & (heures == h)].mean()) for h in np.unique(heures[garde])
    }
    return pd.Series(moyennes).sort_index()


def rapport_tilt_horaire(df: pd.DataFrame, heures: tuple[int, ...] | None = None) -> float:
    """Le rapport entre le poids moyen le plus lourd et le plus léger, sur les heures données."""
    profil = tilt_horaire(df)
    if heures is not None:
        profil = profil[[h for h in heures if h in profil.index]]
    if len(profil) == 0 or float(profil.min()) <= 0:
        return float("nan")
    return float(profil.max() / profil.min())


def exemples_effectifs(df: pd.DataFrame) -> float:
    """
    **Le chiffre qui compte.** Somme des unicités = nombre d'observations réellement
    indépendantes.

    C'est lui, et jamais le nombre de lignes, qui doit entrer dans un intervalle de confiance,
    dans un test de significativité ou dans un dimensionnement de CPCV. Utiliser le nombre brut
    rétrécit artificiellement les intervalles et fabrique de la certitude.
    """
    return float(np.nansum(np.asarray(df["unicite"], dtype=float)))


def resume(df: pd.DataFrame) -> str:
    """Résumé à lire après chaque run."""
    u = np.asarray(df["unicite"], dtype=float)
    valides = ~np.isnan(u)
    conc = np.asarray(df["concurrence"], dtype=float)
    n_lab = int(valides.sum())
    eff = exemples_effectifs(df)
    return "\n".join(
        [
            f"  labels                 : {n_lab:,}",
            f"  concurrence moyenne    : {conc.mean():.2f}  (max {int(conc.max())})",
            f"  unicité moyenne        : {np.nanmean(u):.3f}  (min {np.nanmin(u):.3f}, max {np.nanmax(u):.3f})",
            f"  EXEMPLES EFFECTIFS     : {eff:,.0f}",
            f"  facteur de réduction   : x{n_lab / max(eff, 1):.1f}  "
            f"({100 * eff / max(n_lab, 1):.1f} % du nombre brut)",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calcule les poids d'échantillon (unicité AFML ch. 4)."
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--decroissance",
        type=float,
        default=DECROISSANCE_NEUTRE,
        help="1.0 = neutre (défaut). L'activer est une décision à justifier.",
    )
    args = ap.parse_args()

    df = load_labelle(args.input)
    pondere = calculer_poids(df, args.decroissance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pondere.to_csv(args.out, index=False)

    print(f"\nPoids d'échantillon — {args.input.name} -> {args.out.name}")
    if args.decroissance != DECROISSANCE_NEUTRE:
        print(f"  ⚠️  décroissance temporelle ACTIVÉE (c = {args.decroissance})")
    print(resume(pondere))
    # ⚠️ Le tilt horaire est IMPRIMÉ à chaque génération : un poids qui varie d'un facteur 4,5
    # selon l'heure est un a priori que le modèle recevra sans qu'on le lui ait donné.
    profil = tilt_horaire(pondere)
    fenetre = tuple(range(2, 11))
    print(
        f"\n  tilt horaire des poids (rapport max/min sur 2h-10h) : "
        f"{rapport_tilt_horaire(pondere, fenetre):.2f}"
    )
    print("    " + " ".join(f"{h}h {profil.get(h, float('nan')):.3f}" for h in fenetre))
    print()


if __name__ == "__main__":
    main()
