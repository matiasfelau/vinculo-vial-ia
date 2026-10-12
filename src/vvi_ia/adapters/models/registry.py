"""Registro de adaptadores de modelos y armado de `Models` según la selección por entorno.

Cada puerto se elige con `MODEL_<PUERTO>`: `null` (default) o el nombre de un modelo del manifest
que tenga un adaptador registrado para ese puerto. Todo lo seleccionado se verifica al arrancar:
si el modelo no está en el manifest, no es seleccionable, falta el archivo o no coincide el
sha256, se lanza `ModelSetupError` y el servicio no arranca.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vvi_ia.adapters.models.manifest import (
    PORTS,
    Commercial,
    Manifest,
    ManifestError,
    ModelEntry,
    load_manifest,
    sha256_hex,
)
from vvi_ia.adapters.models.null import (
    NullCarSegmenter,
    NullCurbSegmenter,
    NullDamageDetector,
    NullEmbedder,
    NullFaceDetector,
    NullPlateReader,
    NullVehicleDetector,
)
from vvi_ia.adapters.models.onnx_session import OnnxSession
from vvi_ia.service.validate_evidence import Models

logger = logging.getLogger(__name__)

NULL = "null"
# Único puerto que acepta un modelo con commercial=no: el que mejor protege la privacidad puede no tener
# licencia comercial; en el resto, commercial=no solo vive en tools para comparar.
NON_COMMERCIAL_PORTS = frozenset({"face"})

# Puerto del manifest → campo de `Models`.
PORT_FIELDS: Mapping[str, str] = {
    "detector": "vehicle",
    "face": "face",
    "segmenter": "cars",
    "curb": "curb",
    "embedder": "embedder",
    "ocr": "plate",
    "damage": "damage",
}

NULL_ADAPTERS: Mapping[str, Callable[[], Any]] = {
    "detector": NullVehicleDetector,
    "face": NullFaceDetector,
    "segmenter": NullCarSegmenter,
    "curb": NullCurbSegmenter,
    "embedder": NullEmbedder,
    "ocr": NullPlateReader,
    "damage": NullDamageDetector,
}


class ModelSetupError(Exception):
    """Un modelo seleccionado no se puede usar: el servicio no debe arrancar."""


@dataclass(frozen=True)
class LoadedModel:
    """Un modelo del manifest ya verificado, con su sesión de inferencia."""

    entry: ModelEntry
    version: str
    session: OnnxSession
    # Espacio para que los adaptadores de un mismo modelo (por ejemplo `detector` y `face`) compartan
    # lo que construyen una sola vez.
    shared: dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)


@dataclass
class ModelContext:
    """Lo que recibe la fábrica de un adaptador: el modelo cargado y los parámetros del adaptador."""

    model: LoadedModel
    params: Mapping[str, Any] = field(default_factory=dict)


AdapterFactory = Callable[[ModelContext], Any]

# (puerto, nombre del modelo en el manifest) → fábrica. Cada adaptador nuevo se registra acá.
ADAPTERS: dict[tuple[str, str], AdapterFactory] = {}


def register(port: str, name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    """Decorador para registrar la fábrica de un adaptador en un puerto."""
    if port not in PORTS:
        raise ValueError(f"puerto desconocido: {port}")

    def decorator(factory: AdapterFactory) -> AdapterFactory:
        ADAPTERS[(port, name)] = factory
        return factory

    return decorator


def _read_verified(entry: ModelEntry, models_dir: Path | None) -> tuple[bytes, str]:
    if entry.sha256 is None:
        raise ModelSetupError(f"el modelo {entry.name} no tiene sha256 en el manifest")
    if models_dir is None:
        raise ModelSetupError("MODELS_DIR no está definido y hay modelos seleccionados")
    try:
        data = (models_dir / entry.file).read_bytes()
    except OSError:
        raise ModelSetupError(f"no se pudo leer el archivo del modelo {entry.name} en MODELS_DIR") from None
    actual = sha256_hex(data)
    if actual != entry.sha256:
        raise ModelSetupError(f"el sha256 del modelo {entry.name} no coincide con el manifest")
    return data, actual


def build_models(
    selection: Mapping[str, str],
    manifest_path: Path,
    models_dir: Path | None,
    intra_op_threads: int = 0,
    params: Mapping[str, Any] | None = None,
    adapters: Mapping[tuple[str, str], AdapterFactory] | None = None,
) -> Models:
    """Arma `Models` con la selección `{puerto: nombre}`. Lanza `ModelSetupError`.

    Un mismo modelo elegido en dos puertos (por ejemplo `detector` y `face`) se lee, verifica y
    carga una sola vez: los dos adaptadores comparten la sesión.
    """
    registry = ADAPTERS if adapters is None else adapters
    unknown_ports = sorted(set(selection) - set(PORTS))
    if unknown_ports:
        raise ModelSetupError(f"puertos desconocidos: {', '.join(unknown_ports)}")

    manifest: Manifest | None = None
    loaded: dict[str, LoadedModel] = {}
    built: dict[str, Any] = {}
    for port in PORTS:
        name = selection.get(port, NULL)
        if name == NULL:
            built[PORT_FIELDS[port]] = NULL_ADAPTERS[port]()
            continue
        if manifest is None:
            try:
                manifest = load_manifest(manifest_path)
            except ManifestError as exc:
                raise ModelSetupError(str(exc)) from None
        entry = manifest.get(name)
        if entry is None:
            raise ModelSetupError(f"el modelo {name} (puerto {port}) no está en el manifest")
        if entry.commercial == Commercial.NO:
            if port not in NON_COMMERCIAL_PORTS:
                raise ModelSetupError(
                    f"el modelo {name} es commercial=no: no se puede seleccionar en el puerto {port} "
                    f"(solo {', '.join(sorted(NON_COMMERCIAL_PORTS))} admite esa excepción)"
                )
            logger.warning(
                "modelo sin licencia comercial en uso por excepción del puerto",
                extra={"port": port, "model": name},
            )
        if port not in entry.ports:
            raise ModelSetupError(f"el modelo {name} no sirve al puerto {port}")
        factory = registry.get((port, name))
        if factory is None:
            raise ModelSetupError(f"no hay un adaptador registrado para {name} en el puerto {port}")
        if name not in loaded:
            data, sha256 = _read_verified(entry, models_dir)
            try:
                session = OnnxSession(data, intra_op_threads)
            except Exception as exc:
                raise ModelSetupError(
                    f"ONNX Runtime no pudo cargar el modelo {name}: {type(exc).__name__}"
                ) from None
            loaded[name] = LoadedModel(entry=entry, version=entry.version_label(sha256), session=session)
        try:
            built[PORT_FIELDS[port]] = factory(ModelContext(model=loaded[name], params=params or {}))
        except ModelSetupError:
            raise
        except Exception as exc:
            raise ModelSetupError(
                f"el adaptador de {name} para el puerto {port} no se pudo crear: {type(exc).__name__}"
            ) from None
        logger.info(
            "modelo cargado",
            extra={"port": port, "model_version": loaded[name].version, "commercial": entry.commercial.value},
        )
    return Models(**built)
