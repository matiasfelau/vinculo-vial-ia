"""`fotos.csv` del dataset de rebajes: columnas, valores permitidos y validación.

Cada fila describe una foto de `raw/`. Se rechaza cualquier valor desconocido indicando el número de fila
(la fila 2 es la primera de datos: la 1 es el encabezado). Las fotos de tipo `descartada` se excluyen.

`escena` es opcional: las fotos con el mismo valor son del mismo auto en el mismo rebaje y momento, y el
reparto las deja siempre en el mismo conjunto. Vacía, la foto es única.
"""

import csv
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

COLUMNS = (
    "archivo",
    "escena",
    "tipo",
    "rebaje",
    "parecido",
    "iluminacion",
    "clima",
    "oclusion",
    "notas",
)
DISCARDED = "descartada"
SIMILAR_TYPE = "negativo_parecido"
CLEAR_POSITIVE = "positivo_claro"
MARGINAL_POSITIVE = "positivo_marginal"
CAR_NEXT_TO_RAMP = "negativo_auto_junto_rebaje"
CAR_WITHOUT_RAMP = "negativo_auto_sin_rebaje"
RAMP_WITHOUT_CAR = "negativo_rebaje_sin_auto"
TYPES = (
    CLEAR_POSITIVE,
    MARGINAL_POSITIVE,
    CAR_NEXT_TO_RAMP,
    CAR_WITHOUT_RAMP,
    RAMP_WITHOUT_CAR,
    SIMILAR_TYPE,
    DISCARDED,
)
ALLOWED: Mapping[str, tuple[str, ...]] = {
    "tipo": TYPES,
    "rebaje": ("garage", "accesibilidad", "ninguno"),
    "parecido": (
        "cordon_pintado",
        "cordon_roto",
        "boca_tormenta",
        "porton_sin_rebaje",
        "desnivel_obra",
        "otro",
    ),
    "iluminacion": ("dia", "atardecer", "noche"),
    "clima": ("seco", "mojado", "lluvia"),
    "oclusion": ("ninguna", "parcial", "fuerte"),
}
REQUIRED_TEXT = ("archivo",)
SCENE_SHARED = ("rebaje", "iluminacion", "clima")  # no cambia dentro de una escena


class PhotosError(ValueError):
    """`fotos.csv` tiene errores; el mensaje lista todos, con su número de fila."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True)
class CurbPhoto:
    """Una foto válida del dataset de rebajes."""

    file: str
    scene: str
    kind: str
    ramp: str
    similar: str
    lighting: str
    weather: str
    occlusion: str
    notes: str

    @property
    def group(self) -> str:
        """Lo que no se separa al repartir: la escena, o la foto misma si no tiene escena."""
        return f"escena:{self.scene.casefold()}" if self.scene else f"foto:{self.file}"


def load_photos(
    csv_path: Path, raw_dir: Path | None = None, keep: Collection[str] | None = None
) -> list[CurbPhoto]:
    """Fotos válidas, sin las descartadas. Lanza `PhotosError` con todos los problemas encontrados.

    Si se pasa `raw_dir`, además comprueba que cada archivo exista. Con `keep`, solo devuelve esos tipos.
    """
    if not csv_path.is_file():
        raise PhotosError([f"no existe {csv_path.name}"])
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise PhotosError(
                [
                    f"{csv_path.name} necesita las columnas {', '.join(COLUMNS)} "
                    f"(faltan: {', '.join(sorted(missing))})"
                ]
            )
        problems: list[str] = []
        photos: list[CurbPhoto] = []
        seen: dict[str, int] = {}
        scenes: dict[str, tuple[int, dict[str, str]]] = {}
        for row_number, row in enumerate(reader, start=2):
            row_problems: list[str] = []
            values = {column: (row.get(column) or "").strip() for column in COLUMNS}
            for column in REQUIRED_TEXT:
                if not values[column]:
                    row_problems.append(f"falta {column}")
            for column, allowed in ALLOWED.items():
                value = values[column]
                if column == "parecido":
                    if values["tipo"] == SIMILAR_TYPE and not value:
                        row_problems.append("falta parecido (obligatorio con tipo negativo_parecido)")
                    elif value and values["tipo"] != SIMILAR_TYPE:
                        row_problems.append("parecido solo se usa con tipo negativo_parecido")
                    elif value and value not in allowed:
                        row_problems.append(f"parecido desconocido {value!r}")
                elif value not in allowed:
                    row_problems.append(f"{column} desconocido {value!r}")
            file = values["archivo"]
            if (scene := values["escena"].casefold()) and values["tipo"] != DISCARDED:
                first = scenes.setdefault(scene, (row_number, values))
                for column in SCENE_SHARED:
                    if first[1][column] != values[column]:
                        row_problems.append(
                            f"la escena {values['escena']!r} tiene {column} {values[column]!r}, pero la "
                            f"fila {first[0]} tiene {first[1][column]!r} (una escena comparte "
                            f"{', '.join(SCENE_SHARED)})"
                        )
            if file in seen:
                row_problems.append(f"la foto está repetida (ya figura en la fila {seen[file]})")
            elif file:
                seen[file] = row_number
            if raw_dir is not None and file and not (raw_dir / file).is_file():
                row_problems.append("la foto no está en la carpeta")
            if row_problems:
                problems.append(f"fila {row_number}: " + "; ".join(row_problems))
                continue
            photos.append(
                CurbPhoto(
                    file=file,
                    scene=values["escena"],
                    kind=values["tipo"],
                    ramp=values["rebaje"],
                    similar=values["parecido"],
                    lighting=values["iluminacion"],
                    weather=values["clima"],
                    occlusion=values["oclusion"],
                    notes=values["notas"],
                )
            )
    if problems:
        raise PhotosError(problems)
    return [p for p in photos if p.kind != DISCARDED and (keep is None or p.kind in keep)]
