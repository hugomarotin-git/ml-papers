"""
Rend les modules de `src/` importables depuis `juge/` — à importer **en premier**.

Les deux dossiers sont lancés avec leur propre répertoire en `cwd`, donc en `sys.path` : un
module de `juge/` ne voit pas `src/` sans ça. L'alternative — recopier dans le juge les
constantes dont il a besoin — est exactement ce que F6 et C3 interdisent : une constante
dupliquée marche le jour où on l'écrit et diverge en silence le jour où l'original change.

Un seul endroit fait ce branchement, pour qu'il se voie et se corrige à un seul endroit.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Le dossier `src/` du projet, voisin de `juge/`.
SRC: Path = Path(__file__).resolve().parent.parent / "src"

if not SRC.is_dir():  # pragma: no cover - ne peut arriver qu'après un déplacement de fichiers
    raise RuntimeError(
        f"Dossier source introuvable : {SRC}. Le juge importe ses constantes de `src/` "
        "(horizon du label, notamment) — il ne les recopie pas."
    )

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def horizon_du_label() -> int:
    """
    L'horizon du triple-barrière, lu à sa **source unique** : `src/label_triple_barriere.py`.

    L'import est fait ici, dans une fonction, et pas en tête de module — délibérément. Un import
    en tête dépendrait de l'ordre dans lequel le formateur range le bloc d'imports : si un jour
    il plaçait `label_triple_barriere` avant `chemins`, `sys.path` ne serait pas encore branché
    et l'import échouerait. Faire dépendre la correction du tri d'un linter est une fragilité
    gratuite. Dans une fonction, le branchement ci-dessus a forcément déjà eu lieu.
    """
    from label_triple_barriere import HORIZON_BARRES

    return HORIZON_BARRES
