"""3DRealCar para re-ID: poses → ángulo de vista, partición agrupada por auto y armado de pares.

Todo es puro (sin red ni modelos), para poder probarlo con datos sintéticos.

- **Ángulo de vista**: cada foto trae la pose de la cámara de ARKit (`cameraPoseARFrame`, 4x4 por filas, y
  arriba).
  El auto es el punto del plano horizontal más cercano a todos los ejes ópticos; el ángulo de una vista es el
  azimut de la cámara alrededor de ese punto. Dos vistas del mismo auto se separan por la diferencia de azimut
  (no hace falta saber hacia dónde "mira" el auto).
- **Partición**: agrupada por auto (ningún auto en dos particiones), estratificada por luz, tipo y color, con
  semilla.
- **Pares**: mismo auto con una separación angular mínima; autos distintos, todos y "difíciles" (misma marca y
  color).
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_MIN_SEPARATION_DEG = 30.0
ANGLE_BINS: Sequence[tuple[float, float]] = ((30.0, 60.0), (60.0, 90.0), (90.0, 135.0), (135.0, 180.0001))
POSE_LENGTH = 16
MIN_RAYS = 3  # con menos ejes ópticos el punto de cruce no está definido
DETERMINANT_FLOOR = 1e-6


@dataclass(frozen=True)
class Car:
    """Un auto con sus anotaciones (las de `annotation2000.xlsx`)."""

    name: str
    brand: str
    model: str
    lighting: str  # standard | reflective | dark
    kind: str  # Sedan, SUV, ...
    color: str


@dataclass(frozen=True)
class View:
    """Una foto de un auto con su azimut (grados, 0-360)."""

    car: str
    frame: int
    azimuth: float


def camera_rays(poses: Sequence[Sequence[float]]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centros de cámara y direcciones de vista proyectados al plano horizontal `(x, z)` (ARKit: y es arriba).

    La pose es cámara→mundo y la cámara mira por su eje `-z`. Una dirección casi vertical se descarta (`nan`).
    """
    centers, directions = [], []
    for pose in poses:
        if len(pose) != POSE_LENGTH:
            raise ValueError("la pose de ARKit tiene que tener 16 números")
        matrix = np.asarray(pose, np.float64).reshape(4, 4)
        forward = -matrix[:3, 2]
        horizontal = np.array([forward[0], forward[2]])
        norm = float(np.linalg.norm(horizontal))
        centers.append([matrix[0, 3], matrix[2, 3]])
        directions.append(horizontal / norm if norm > 1e-6 else [math.nan, math.nan])
    return np.asarray(centers), np.asarray(directions)


def car_center(centers: NDArray[np.float64], directions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Punto del plano más cercano (mínimos cuadrados) a los ejes ópticos; si no se define, el promedio."""
    valid = ~np.isnan(directions).any(axis=1)
    if valid.sum() >= MIN_RAYS:
        a = np.zeros((2, 2))
        b = np.zeros(2)
        for point, direction in zip(centers[valid], directions[valid], strict=True):
            projector = np.eye(2) - np.outer(direction, direction)
            a += projector
            b += projector @ point
        if abs(np.linalg.det(a)) > DETERMINANT_FLOOR:
            return np.asarray(np.linalg.solve(a, b))
    return np.asarray(centers.mean(axis=0))


def azimuths(poses: Sequence[Sequence[float]]) -> list[float]:
    """Azimut (grados, 0-360) de la cámara de cada pose alrededor del auto."""
    if not poses:
        return []
    centers, directions = camera_rays(poses)
    center = car_center(centers, directions)
    offsets = centers - center
    return [float(math.degrees(math.atan2(x, z)) % 360.0) for x, z in offsets]


def angle_difference(a: float, b: float) -> float:
    """Diferencia mínima entre dos azimuts, en grados (0-180)."""
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


COMMON_COLORS = frozenset(
    {"白色", "黑色", "银色", "灰色"}
)  # blanco, negro, plata y gris: la gran mayoría de 3DRealCar


def stratum(car: Car) -> tuple[str, str, str]:
    """Luz, tipo de carrocería y color (los colores poco frecuentes van juntos en `otro`)."""
    return car.lighting, car.kind, car.color if car.color in COMMON_COLORS else "otro"


def group_split(
    cars: Sequence[Car], fractions: Sequence[float] = DEFAULT_FRACTIONS, seed: int = 0
) -> dict[str, str]:
    """`auto → partición`, con cada auto en una sola, estratificado por (luz, tipo, color) y con semilla fija.

    Dentro de cada estrato los autos se sortean y se reparten según `fractions`; los estratos chicos
    (pocos autos) se completan en orden `train`, `val`, `test` para que cada partición los vea.
    """
    if len(fractions) != len(SPLITS) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("las fracciones tienen que ser tres y sumar 1")
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for car in sorted(cars, key=lambda c: c.name):
        strata[stratum(car)].append(car.name)
    assignment: dict[str, str] = {}
    for key in sorted(strata):
        names = strata[key]
        # Un sorteo por estrato (semilla + estrato): sumar autos de otro estrato no mueve a los demás
        rng = random.Random(f"{seed}|{'|'.join(key)}")  # noqa: S311 - partición de datos, no criptografía
        rng.shuffle(names)
        n = len(names)
        n_test = round(n * fractions[2])
        n_val = round(n * fractions[1])
        if n >= len(SPLITS):  # cada partición ve al menos un auto del estrato
            n_test, n_val = max(1, n_test), max(1, n_val)
        n_train = n - n_val - n_test
        for index, name in enumerate(names):
            assignment[name] = (
                SPLITS[0] if index < n_train else SPLITS[1] if index < n_train + n_val else SPLITS[2]
            )
    return assignment


def same_car_pairs(
    views: Sequence[View], min_separation: float = DEFAULT_MIN_SEPARATION_DEG
) -> list[tuple[int, int, float]]:
    """Pares de vistas del mismo auto separadas por al menos `min_separation` grados: `(i, j, ángulo)`."""
    pairs = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            angle = angle_difference(views[i].azimuth, views[j].azimuth)
            if angle >= min_separation:
                pairs.append((i, j, angle))
    return pairs


def angle_bin(angle: float) -> str | None:
    for low, high in ANGLE_BINS:
        if low <= angle < high:
            return f"{low:.0f}-{min(high, 180):.0f}"
    return None


def different_car_pairs(
    views: Sequence[View],
    cars: Mapping[str, Car],
    limit: int,
    seed: int = 0,
    hard: bool = False,
) -> list[tuple[int, int]]:
    """Hasta `limit` pares de vistas de autos distintos, sorteados con `seed`.

    Con `hard`, solo de autos de la misma marca y el mismo color (los que más se confunden).
    """
    rng = random.Random(seed)  # noqa: S311 - muestreo de pares, no criptografía
    by_car: dict[str, list[int]] = defaultdict(list)
    for index, view in enumerate(views):
        by_car[view.car].append(index)
    names = sorted(by_car)
    if hard:
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in names:
            groups[(cars[name].brand, cars[name].color)].append(name)
        candidates = [
            (a, b) for group in groups.values() for i, a in enumerate(group) for b in group[i + 1 :]
        ]
    else:
        candidates = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    if not candidates:
        return []
    pairs: list[tuple[int, int]] = []
    if len(candidates) * 4 <= limit:  # pocas parejas de autos: se usan todas, con varias vistas cada una
        rng.shuffle(candidates)
        per_pair = max(1, math.ceil(limit / len(candidates)))
        for a, b in candidates:
            for _ in range(per_pair):
                pairs.append((rng.choice(by_car[a]), rng.choice(by_car[b])))
        rng.shuffle(pairs)
        return pairs[:limit]
    seen: set[tuple[int, int]] = set()
    while len(pairs) < limit and len(seen) < limit * 4:
        a, b = rng.choice(candidates)
        pair = (rng.choice(by_car[a]), rng.choice(by_car[b]))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


@dataclass(frozen=True)
class PairSet:
    """Los pares de una partición, como índices sobre la lista de recortes."""

    same: list[tuple[int, int, float]]  # (i, j, ángulo)
    different: list[tuple[int, int]]
    hard: list[tuple[int, int]]


def views_and_cars(items: Sequence[Mapping[str, object]]) -> tuple[list[View], dict[str, Car]]:
    """Vistas y autos de la lista de `crops.json` (un elemento por recorte)."""
    views: list[View] = []
    cars: dict[str, Car] = {}
    for item in items:
        name = str(item["auto"])
        views.append(View(name, int(str(item["foto"])), float(str(item["azimut"]))))
        if name not in cars:
            cars[name] = Car(
                name,
                str(item["marca"]),
                str(item["modelo"]),
                str(item["luz"]),
                str(item["tipo"]),
                str(item["color"]),
            )
    return views, cars


def build_pairs(
    items: Sequence[Mapping[str, object]],
    min_separation: float = DEFAULT_MIN_SEPARATION_DEG,
    different_limit: int = 100_000,
    hard_limit: int = 30_000,
    seed: int = 0,
) -> PairSet:
    """Todos los pares del mismo auto (con la separación mínima) y una muestra de distintos y difíciles."""
    views, cars = views_and_cars(items)
    by_car: dict[str, list[int]] = defaultdict(list)
    for index, view in enumerate(views):
        by_car[view.car].append(index)
    same: list[tuple[int, int, float]] = []
    for indices in by_car.values():
        for i, j, angle in same_car_pairs([views[k] for k in indices], min_separation):
            same.append((indices[i], indices[j], angle))
    return PairSet(
        same=same,
        different=different_car_pairs(views, cars, different_limit, seed),
        hard=different_car_pairs(views, cars, hard_limit, seed, hard=True),
    )
