"""Detector de daños RF-DETR (CarDD): cajas por clase, umbral y registro con un ONNX diminuto."""

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from vvi_ia.adapters.models.manifest import Commercial, load_manifest
from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_damage import DAMAGE_LABELS, MODEL_NAME, RfDetrDamageDetector
from vvi_ia.adapters.models.rfdetr_detector import OUTPUT_NAMES
from vvi_ia.config import load_settings
from vvi_ia.main import adapter_params

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = len(DAMAGE_LABELS) + 1  # las clases más la columna de fondo
QUERIES = 4


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def outputs(*rows: tuple[int, float, tuple[float, float, float, float]]) -> tuple[NDArray[Any], NDArray[Any]]:
    dets = np.zeros((1, QUERIES, 4), np.float32)
    dets[..., 2:] = 0.1
    labels = np.full((1, QUERIES, CLASSES), -6.0, np.float32)
    for query, (column, confidence, box) in enumerate(rows):
        dets[0, query] = box
        labels[0, query, column] = logit(confidence)
    return dets, labels


class FakeSession:
    def __init__(self, result: Sequence[NDArray[Any]] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Sequence[str] | None, Mapping[str, NDArray[Any]]]] = []

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        self.calls.append((output_names, input_feed))
        if isinstance(self.result, Exception):
            raise self.result
        return list(self.result)


def detector(session: FakeSession, min_conf: float = 0.4) -> RfDetrDamageDetector:
    return RfDetrDamageDetector(session, "input", (16, 24), min_conf, f"{MODEL_NAME}@abc")


def photo(height: int = 40, width: int = 80) -> NDArray[np.uint8]:
    return np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_las_clases_siguen_el_orden_de_cardd() -> None:
    assert list(DAMAGE_LABELS.values()) == [
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "lamp_broken",
        "tire_flat",
    ]


def test_devuelve_las_cajas_de_dano_con_su_clase_y_confianza() -> None:
    session = FakeSession(outputs((0, 0.9, (0.5, 0.5, 0.4, 0.5)), (1, 0.6, (0.2, 0.2, 0.1, 0.1))))

    boxes = detector(session).detect(photo(40, 80))

    assert boxes is not None
    assert [(box.label, round(box.conf, 2)) for box in boxes] == [("dent", 0.9), ("scratch", 0.6)]
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == pytest.approx(
        (24.0, 10.0, 56.0, 30.0), abs=1e-3
    )


def test_solo_las_cajas_con_confianza_mayor_al_umbral() -> None:
    session = FakeSession(outputs((0, 0.9, (0.5, 0.5, 0.4, 0.5)), (2, 0.3, (0.2, 0.2, 0.1, 0.1))))

    assert [b.label for b in detector(session, 0.4).detect(photo()) or []] == ["dent"]
    assert [b.label for b in detector(session, 0.2).detect(photo()) or []] == ["dent", "crack"]
    assert detector(session, 0.95).detect(photo()) == []


def test_ignora_la_columna_de_fondo() -> None:
    session = FakeSession(outputs((6, 0.99, (0.5, 0.5, 0.4, 0.4))))  # la columna 6 no es ninguna clase

    assert detector(session).detect(photo()) == []


def test_sin_danos_devuelve_lista_vacia_y_no_none() -> None:
    assert detector(FakeSession(outputs())).detect(photo()) == []


def test_pasa_el_tensor_con_el_nombre_de_entrada() -> None:
    session = FakeSession(outputs())

    detector(session).detect(photo())

    ((names, feed),) = session.calls
    assert tuple(names or ()) == OUTPUT_NAMES
    assert feed["input"].shape == (1, 3, 16, 24)


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    adapter = detector(FakeSession(RuntimeError("detalle interno con datos de la imagen")))

    with caplog.at_level(logging.WARNING):
        assert adapter.detect(photo()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del detector de daños"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "detalle interno" not in caplog.text


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_imagen_invalida_devuelve_none_sin_correr_el_modelo(shape: tuple[int, ...]) -> None:
    session = FakeSession(outputs())

    assert detector(session).detect(np.zeros(shape, np.uint8)) is None
    assert session.calls == []


def test_salida_con_menos_clases_devuelve_none() -> None:
    dets, labels = outputs((0, 0.9, (0.5, 0.5, 0.4, 0.4)))

    assert detector(FakeSession((dets, labels[..., :4]))).detect(photo()) is None


# --- registro: un ONNX diminuto con la misma interfaz que RF-DETR ---------------------------


def tiny_damage(names: Sequence[str] = OUTPUT_NAMES, classes: int = CLASSES) -> bytes:
    dets, labels = outputs((0, 0.9, (0.5, 0.5, 0.4, 0.5)), (4, 0.7, (0.2, 0.2, 0.1, 0.1)))
    constants = [dets, labels[..., :classes]]
    graph = helper.make_graph(
        [helper.make_node("Identity", [f"const_{i}"], [name]) for i, name in enumerate(names)],
        "tiny_damage",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
            for name, value in zip(names, constants, strict=True)
        ],
        [numpy_helper.from_array(value, f"const_{i}") for i, value in enumerate(constants)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def write_env(tmp_path: Path, weights: bytes) -> tuple[Path, Path]:
    models_dir = tmp_path / "weights"
    models_dir.mkdir()
    (models_dir / f"{MODEL_NAME}.onnx").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    entry = {
        "name": MODEL_NAME,
        "version": "de-prueba",
        "ports": ["damage"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2026},
        "source": "generado en el test",
        "commercial": "escalable",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_damage() -> None:
    assert ("damage", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_detecta(tmp_path: Path) -> None:
    weights = tiny_damage()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"damage": MODEL_NAME}, manifest, models_dir, params={"damage_min_conf": 0.4})

    assert models.versions()["damage"] == f"{MODEL_NAME}@{hashlib.sha256(weights).hexdigest()[:12]}"
    boxes = models.damage.detect(photo(40, 80))
    assert boxes is not None and [box.label for box in boxes] == ["dent", "lamp_broken"]


def test_el_umbral_llega_por_los_parametros(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_damage())

    strict = build_models({"damage": MODEL_NAME}, manifest, models_dir, params={"damage_min_conf": 0.8})

    assert [box.label for box in strict.damage.detect(photo()) or []] == ["dent"]


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_un_umbral_fuera_de_rango_no_deja_arrancar(tmp_path: Path, value: float) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_damage())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"damage": MODEL_NAME}, manifest, models_dir, params={"damage_min_conf": value})


@pytest.mark.parametrize(
    "weights",
    [tiny_damage(names=("dets", "otra")), tiny_damage(classes=4)],
    ids=["otra salida", "faltan clases"],
)
def test_un_archivo_que_no_es_el_detector_de_danos_no_deja_arrancar(tmp_path: Path, weights: bytes) -> None:
    manifest, models_dir = write_env(tmp_path, weights)

    with pytest.raises(ModelSetupError):
        build_models({"damage": MODEL_NAME}, manifest, models_dir)


def test_el_umbral_de_danos_viaja_a_los_adaptadores(base_env: pytest.MonkeyPatch) -> None:
    assert adapter_params(load_settings())["damage_min_conf"] == 0.7

    base_env.setenv("DAMAGE_MIN_CONF", "0.55")
    assert adapter_params(load_settings())["damage_min_conf"] == 0.55


def test_el_manifest_del_repo_declara_el_detector_de_danos_como_escalable() -> None:
    entry = load_manifest(REPO_ROOT / "models" / "manifest.json").get(MODEL_NAME)

    assert entry is not None
    assert entry.ports == ("damage",)
    assert entry.commercial == Commercial.ESCALABLE
    assert "CarDD" in entry.notes and "no se publica" in entry.notes
    assert entry.sha256 is not None


@pytest.mark.models
def test_pesos_reales_corren_sobre_una_foto_sin_danos() -> None:
    models = build_models(
        {"damage": MODEL_NAME}, REPO_ROOT / "models" / "manifest.json", Path(os.environ["MODELS_DIR"])
    )

    assert models.damage.detect(np.full((240, 320, 3), 128, np.uint8)) == []
