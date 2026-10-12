"""Reparto de las fotos de rebajes en train, valid y test.

Reglas:

- **Ninguna calle en dos conjuntos.** Fotos de la misma calle se parecen demasiado: si una quedara en
  entrenamiento y otra en prueba, el resultado de la prueba sería optimista.
- Aproximadamente 60 / 20 / 20 y **estratificado por tipo**: cada conjunto tiene una proporción parecida de
  cada tipo de foto.
- **El test se guarda en un archivo la primera vez y después no cambia**, para poder comparar modelos
  entrenados en momentos distintos con las mismas fotos. Las calles nuevas van solo a train y valid; si
  aparece una foto nueva de una calle del test, queda afuera de todo (no se puede mover sin ensuciar el test).
"""

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photos import CurbPhoto

TARGETS: Mapping[str, float] = {"train": 0.60, "valid": 0.20, "test": 0.20}
TEST_FILE = "test.json"
ASSIGNMENTS_FILE = "assignments.json"
MIN_GROUPS = 3


class SplitError(ValueError):
    """No se puede repartir con los datos que hay."""


@dataclass(frozen=True)
class Assignment:
    """Resultado del reparto."""

    splits: Mapping[str, list[CurbPhoto]]
    left_out: list[CurbPhoto]  # fotos nuevas de calles que ya están en el test

    def summary(self) -> dict[str, Any]:
        """Cifras por conjunto: fotos, calles y fotos de cada tipo."""
        return {
            name: {
                "fotos": len(photos),
                "calles": len({p.group for p in photos}),
                "tipos": dict(sorted(Counter(p.kind for p in photos).items())),
            }
            for name, photos in self.splits.items()
        } | {"fuera_del_test": len(self.left_out)}


def _by_street(photos: Sequence[CurbPhoto]) -> dict[str, list[CurbPhoto]]:
    groups: dict[str, list[CurbPhoto]] = defaultdict(list)
    for photo in photos:
        groups[photo.group].append(photo)
    return groups


def _distribute(
    groups: Mapping[str, Sequence[CurbPhoto]], fractions: Mapping[str, float], seed: int
) -> dict[str, list[str]]:
    """Reparte calles entre conjuntos acercándose a las fracciones y a la proporción de cada tipo.

    Va de las calles más grandes a las más chicas y le da cada una al conjunto al que más le falta de los
    tipos que esa calle trae (ante un empate, desempata un orden estable con la semilla).
    """
    totals: Counter[str] = Counter(p.kind for photos in groups.values() for p in photos)
    wanted = {
        name: {kind: fraction * count for kind, count in totals.items()}
        for name, fraction in fractions.items()
    }
    have: dict[str, Counter[str]] = {name: Counter() for name in fractions}
    tie_break = {key: random.Random(f"{seed}|{key}").random() for key in groups}  # noqa: S311
    chosen: dict[str, list[str]] = {name: [] for name in fractions}
    for key in sorted(groups, key=lambda k: (-len(groups[k]), tie_break[k])):
        kinds = Counter(p.kind for p in groups[key])
        best = max(
            fractions,
            key=lambda name: (
                sum(
                    count * (wanted[name][kind] - have[name][kind]) / totals[kind]
                    for kind, count in kinds.items()
                ),
                -sum(have[name].values()) / max(fractions[name], 1e-9),
            ),
        )
        chosen[best].append(key)
        have[best].update(kinds)
    return chosen


def assign(
    photos: Sequence[CurbPhoto], seed: int = 0, fixed_test: Mapping[str, Any] | None = None
) -> Assignment:
    """Reparte las fotos. Con `fixed_test` (el contenido de `test.json`) respeta ese test."""
    groups = _by_street(photos)
    left_out: list[CurbPhoto] = []
    if fixed_test is None:
        if len(groups) < MIN_GROUPS:
            raise SplitError(
                f"hacen falta al menos {MIN_GROUPS} calles distintas para repartir (hay {len(groups)})"
            )
        chosen = _distribute(groups, TARGETS, seed)
    else:
        test_streets = set(fixed_test["calles"])
        test_files = set(fixed_test["archivos"])
        rest = {key: items for key, items in groups.items() if key not in test_streets}
        if len(rest) < 2:
            raise SplitError("fuera del test quedan menos de 2 calles: no se puede separar train de valid")
        train_share = TARGETS["train"] / (TARGETS["train"] + TARGETS["valid"])
        chosen = _distribute(rest, {"train": train_share, "valid": 1 - train_share}, seed)
        chosen["test"] = [key for key in groups if key in test_streets]
        for key in chosen["test"]:
            left_out.extend(p for p in groups[key] if p.file not in test_files)
    ordered = ("train", "valid", "test")
    splits = {
        name: sorted(
            (
                p
                for key in chosen.get(name, [])
                for p in groups[key]
                if name != "test" or fixed_test is None or p.file in set(fixed_test["archivos"])
            ),
            key=lambda p: p.file,
        )
        for name in ordered
    }
    return Assignment(splits, sorted(left_out, key=lambda p: p.file))


def load_fixed_test(splits_dir: Path) -> dict[str, Any] | None:
    path = splits_dir / TEST_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not {"calles", "archivos"} <= set(data):
        raise SplitError(f"{TEST_FILE} no tiene el formato esperado: {{calles, archivos}}")
    return data  # type: ignore[no-any-return]


def save_test(splits_dir: Path, assignment: Assignment, seed: int) -> None:
    """Guarda el test la primera vez. Si el archivo ya existe, no lo toca."""
    path = splits_dir / TEST_FILE
    if path.exists():
        return
    splits_dir.mkdir(parents=True, exist_ok=True)
    test = assignment.splits["test"]
    path.write_text(
        json.dumps(
            {
                "semilla": seed,
                "calles": sorted({p.group for p in test}),
                "archivos": sorted(p.file for p in test),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_assignments(splits_dir: Path, assignment: Assignment) -> None:
    """`assignments.json`: a qué conjunto fue cada foto (queda en tu carpeta, fuera del repo)."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    data = {name: [p.file for p in photos] for name, photos in assignment.splits.items()}
    data["fuera_del_test"] = [p.file for p in assignment.left_out]
    (splits_dir / ASSIGNMENTS_FILE).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
