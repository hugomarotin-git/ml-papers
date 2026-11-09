"""
Paper 1, exemple hors finance — une porte séquentielle à budget d'erreurs sur un flux de
décisions tabulaires (réservations d'hôtel).

Le jeu : « Hotel booking demand » (Antonio, de Almeida, Nunes, 2019, Data in Brief 22, 41-49),
119 390 réservations de deux hôtels portugais, 2015-2017, avec l'issue binaire `is_canceled`
et un enjeu par ligne : prix moyen par nuit (`adr`) x nuits. Copie publique utilisée :
tidytuesday 2020-02-11. Le fichier vit HORS dépôt, dans `data/hors_finance/hotels.csv`.

Le système jugé : un décideur qui, pour chaque réservation, prédit « annule » ou « honore ».
Modèle stylisé : décision juste → +enjeu (chambre revendue ou conservée à bon escient),
décision fausse → -enjeu (chambre invendue ou double réservation compensée). Comme dans le
contrôle positif du pipeline financier, le signal est PLANTÉ par alignement : la prédiction
est juste avec probabilité `edge` (0,5 = hasard pur, le null). Ce module triche donc exprès,
il lit l'issue réelle : il ne sert qu'au contrôle, jamais à décider.

La porte (analogue des règles FTMO 2-step) : un capital K, deux phases successives (cible +10 %
puis +5 % du capital), perte maximale 10 % du capital de départ de la phase, et un budget de
N décisions pour tout boucler. Un chemin « passe » s'il franchit les deux cibles sans toucher
la perte maximale avant d'avoir épuisé le budget ; il « casse » s'il touche la perte ; il est
« censuré » si le budget s'épuise avant conclusion. La borne basse = passés / chemins,
les censurés comptés comme des échecs — exactement la lecture du juge financier.

Le dimensionnement : une taille telle qu'une décision d'enjeu médian risque `risque` x K.
Les enjeux restent hétérogènes (winsorisés au p99 pour qu'une seule ligne ne vaille pas un
quart du capital) : c'est la queue lourde des gains réels qu'on veut garder. `enjeux_unitaires`
les remplace tous par la médiane, pour isoler l'effet de la seule porte.

Les départs : un chemin démarre à chaque `pas`-ième réservation du flux (ordre d'arrivée),
parmi celles qui disposent du budget entier devant elles. Les chemins d'une même seed partagent
le même tirage de justesse : ils se chevauchent comme les chemins du juge financier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_NULL = 0.5
MOIS = {
    m: i + 1
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    )
}


def charger_flux(chemin: Path, winsor: float = 0.99) -> pd.DataFrame:
    """Le flux de réservations trié par date d'arrivée : colonnes `date`, `issue`, `enjeu`."""
    d = pd.read_csv(chemin)
    nuits = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    enjeu = d["adr"] * nuits
    garder = (d["adr"] > 0) & (nuits > 0) & (d["adr"] < 1000)
    d = d[garder].copy()
    enjeu = enjeu[garder]
    plafond = float(enjeu.quantile(winsor))
    d["enjeu"] = np.minimum(enjeu.to_numpy(dtype=float), plafond)
    annee = pd.Series(d["arrival_date_year"]).to_numpy(dtype=int)
    mois = np.array([MOIS[m] for m in pd.Series(d["arrival_date_month"]).tolist()], dtype=int)
    jour = pd.Series(d["arrival_date_day_of_month"]).to_numpy(dtype=int)
    dates = pd.to_datetime(
        [f"{a:04d}-{m:02d}-{j:02d}" for a, m, j in zip(annee, mois, jour, strict=True)]
    )
    flux = pd.DataFrame(
        {
            "date": dates,
            "issue": pd.Series(d["is_canceled"]).to_numpy(dtype=int),
            "enjeu": np.asarray(d["enjeu"], dtype=float),
        }
    )
    flux = flux.sort_values("date", kind="stable").reset_index(drop=True)
    return pd.DataFrame(flux)


@dataclass(frozen=True)
class PorteConfig:
    capital: float = 50_000.0
    cible_phase_1: float = 0.10
    cible_phase_2: float = 0.05
    perte_max: float = 0.10
    budget: int = 250  # décisions


@dataclass(frozen=True)
class ResultatPorte:
    n_chemins: int
    n_passes: int
    n_casses: int
    n_censures: int

    @property
    def p_censure(self) -> float:
        return self.n_censures / self.n_chemins if self.n_chemins else 0.0

    @property
    def borne_basse(self) -> float:
        return self.n_passes / self.n_chemins if self.n_chemins else 0.0


def _premier_franchissement(
    cum: np.ndarray, haut: float, bas: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pour chaque ligne de `cum` (chemins x pas), l'indice du premier pas où le cumul atteint
    `haut` ou tombe à `bas`, et lequel des deux. Indice = nb de colonnes si jamais atteint.
    """
    _, t = cum.shape
    touche_haut = cum >= haut
    touche_bas = cum <= bas
    i_haut = np.where(touche_haut.any(axis=1), touche_haut.argmax(axis=1), t)
    i_bas = np.where(touche_bas.any(axis=1), touche_bas.argmax(axis=1), t)
    # Le bas gagne à égalité : on ne crédite pas une cible atteinte par un pas qui casse aussi.
    gagne = i_haut < i_bas
    return np.where(gagne, i_haut, i_bas), gagne


def jouer_porte(gains: np.ndarray, departs: np.ndarray, config: PorteConfig) -> ResultatPorte:
    """
    Joue la porte à deux phases depuis chaque départ, sur le vecteur des gains par décision
    (déjà signés et dimensionnés). Entièrement vectorisé : chemins x budget.
    """
    n_dep = len(departs)
    if n_dep == 0:
        raise ValueError("Aucun départ ne dispose du budget entier.")
    idx = departs[:, None] + np.arange(config.budget)[None, :]
    fen = gains[idx]  # chemins x budget
    cum = np.cumsum(fen, axis=1)
    k = config.capital
    # Phase 1.
    fin_1, passe_1 = _premier_franchissement(cum, k * config.cible_phase_1, -k * config.perte_max)
    # Phase 2 : repart du solde au moment du passage, avec le budget restant.
    t = config.budget
    pos = np.arange(t)[None, :]
    fin_1_safe = np.minimum(fin_1, t - 1)
    base = cum[np.arange(n_dep), fin_1_safe][:, None]
    cum_2 = np.where(pos > fin_1[:, None], cum - base, 0.0)
    fin_2, passe_2 = _premier_franchissement(cum_2, k * config.cible_phase_2, -k * config.perte_max)
    passe = passe_1 & passe_2 & (fin_2 < t)
    casse = (~passe_1 & (fin_1 < t)) | (passe_1 & ~passe_2 & (fin_2 < t))
    censure = ~passe & ~casse
    return ResultatPorte(n_dep, int(passe.sum()), int(casse.sum()), int(censure.sum()))


def taille_pour_risque(enjeux: np.ndarray, risque: float, capital: float) -> float:
    """La taille qui fait risquer `risque` x capital à une décision d'enjeu médian."""
    return risque * capital / float(np.median(enjeux))


def departs_utilisables(n_flux: int, budget: int, pas: int, n_utilisable: int | None) -> np.ndarray:
    """
    Les départs : toutes les positions qui ont `budget` décisions devant elles, un sur `pas`.
    `n_utilisable` restreint aux N DERNIÈRES positions utilisables (flux tronqué par la fin,
    comme le dev financier tronqué aux dernières années).
    """
    max_dep = n_flux - budget
    if max_dep <= 0:
        return np.array([], dtype=int)
    debut = 0 if n_utilisable is None else max(max_dep - n_utilisable, 0)
    return np.arange(debut, max_dep, pas, dtype=int)


@dataclass(frozen=True)
class PointPorte:
    budget: int
    risque: float
    edge: float
    n_replications: int
    p50: float
    p95: float
    barre_null_p95: float
    censure_mediane: float
    censure_null_mediane: float
    n_chemins: int

    @property
    def interpretable(self) -> bool:
        return self.censure_null_mediane <= 0.10 and self.censure_mediane <= 0.10

    @property
    def detectable(self) -> bool:
        return self.interpretable and self.p50 > self.barre_null_p95


def mesurer_porte(
    flux: pd.DataFrame,
    budget: int,
    risque: float,
    edges: list[float],
    seeds: list[int],
    pas: int = 1,
    n_utilisable: int | None = None,
    enjeux_unitaires: bool = False,
    capital: float = 50_000.0,
) -> list[PointPorte]:
    """Le contrôle positif de la porte : chaque edge et le null, mêmes départs, mêmes enjeux."""
    config = PorteConfig(capital=capital, budget=budget)
    enjeux = flux["enjeu"].to_numpy(dtype=float)
    if enjeux_unitaires:
        enjeux = np.full_like(enjeux, float(np.median(enjeux)))
    taille = taille_pour_risque(enjeux, risque, capital)
    departs = departs_utilisables(len(enjeux), budget, pas, n_utilisable)
    a_mesurer = sorted({EDGE_NULL, *edges})
    bornes: dict[float, list[float]] = {}
    censures: dict[float, list[float]] = {}
    for edge in a_mesurer:
        b, c = [], []
        for seed in seeds:
            alea = np.random.default_rng(seed)
            juste = alea.random(len(enjeux)) < edge
            gains = np.where(juste, 1.0, -1.0) * taille * enjeux
            r = jouer_porte(gains, departs, config)
            b.append(r.borne_basse)
            c.append(r.p_censure)
        bornes[edge], censures[edge] = b, c
    barre = float(np.percentile(bornes[EDGE_NULL], 95))
    cens_null = float(np.median(censures[EDGE_NULL]))
    return [
        PointPorte(
            budget=budget,
            risque=risque,
            edge=edge,
            n_replications=len(seeds),
            p50=float(np.median(bornes[edge])),
            p95=float(np.percentile(bornes[edge], 95)),
            barre_null_p95=barre,
            censure_mediane=float(np.median(censures[edge])),
            censure_null_mediane=cens_null,
            n_chemins=len(departs),
        )
        for edge in a_mesurer
    ]


def point_en_dict(p: PointPorte) -> dict:
    d = asdict(p)
    d["interpretable"] = p.interpretable
    d["detectable"] = p.detectable
    return d
