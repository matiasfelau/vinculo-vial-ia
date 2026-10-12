"""Segmentador de rebajes RF-DETR-Seg: una máscara por clase y unión de instancias, con un ONNX diminuto."""

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_seg import OUTPUT_NAMES
from vvi_ia.adapters.models.rfdetr_seg_curb import (
    CLASSES,
    MODEL_NAME,
    RfDetrSegCurbSegmenter,
    decode_class_masks,
)
from vvi_ia.config import load_settings
from vvi_ia.main import adapter_params

CLASS_COLUMNS = 3  # curb, curb_cut y la columna de fondo
QUERIES = 4
MASK_SIDE = 8
ON, OFF = 6.0, -6.0
CURB, CUT = 0, 1


def labels_for(*rows: tuple[int, float]) -> NDArray[np.float32]:
    """`labels` de una imagen: cada fila es `(columna, logit)` de una consulta; el resto en OFF."""
    out = np.full((1, QUERIES, CLASS_COLUMNS), OFF, np.float32)
    for query, (column, value) in enumerate(rows):
        out[0, query, column] = value
    return out


def masks_for(*regions: str) -> NDArray[np.float32]:
    """`masks`: por consulta, 'left', 'right', 'all' o 'none'."""
    out = np.full((1, QUERIES, MASK_SIDE, MASK_SIDE), -1.0, np.float32)
    for query, region in enumerate(regions):
        if region in {"left", "all"}:
            out[0, query, :, : MASK_SIDE // 2] = 1.0
        if region in {"right", "all"}:
            out[0, query, :, MASK_SIDE // 2 :] = 1.0
    return out


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


def segmenter(
    labels: NDArray[Any], masks: NDArray[Any], min_conf: float = 0.5
) -> tuple[RfDetrSegCurbSegmenter, FakeSession]:
    session = FakeSession([np.zeros((1, QUERIES, 4), np.float32), labels, masks])
    return RfDetrSegCurbSegmenter(session, "input", (16, 24), min_conf, f"{MODEL_NAME}@abc"), session


def photo(height: int = 30, width: int = 50) -> NDArray[np.uint8]:
    return np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_las_clases_son_curb_y_curb_cut_en_ese_orden() -> None:
    assert list(CLASSES) == ["curb", "curb_cut"]


def test_devuelve_una_mascara_por_clase_del_tamano_de_la_foto() -> None:
    adapter, session = segmenter(labels_for((CURB, ON), (CUT, ON)), masks_for("all", "left"))

    result = adapter.curb(photo(30, 50))

    assert result is not None
    curb, cut = result
    assert curb.shape == cut.shape == (30, 50) and curb.dtype == cut.dtype == np.bool_
    assert curb.all()
    assert cut[:, :20].all() and not cut[:, 30:].any()  # solo la mitad izquierda
    ((names, feed),) = session.calls
    assert tuple(names or ()) == OUTPUT_NAMES and feed["input"].shape == (1, 3, 16, 24)


def test_une_las_instancias_de_cada_clase() -> None:
    adapter, _ = segmenter(labels_for((CUT, ON), (CUT, ON), (CURB, ON)), masks_for("left", "right", "all"))

    curb, cut = adapter.curb(photo()) or (None, None)

    assert curb is not None and cut is not None
    assert cut.all()  # izquierda ∪ derecha
    assert curb.all()


def test_ignora_las_instancias_con_poca_confianza() -> None:
    adapter, _ = segmenter(labels_for((CUT, ON), (CUT, -1.0)), masks_for("left", "right"))  # la segunda: 0,27

    cut = (adapter.curb(photo()) or (None, None))[1]

    assert cut is not None and cut[:, :20].all() and not cut[:, 30:].any()


def test_el_umbral_es_el_del_adaptador() -> None:
    labels, masks = labels_for((CUT, 1.0)), masks_for("all")  # confianza 0,73

    assert (segmenter(labels, masks, 0.5)[0].curb(photo()) or (None, None))[1].any()  # type: ignore[union-attr]
    assert not (segmenter(labels, masks, 0.9)[0].curb(photo()) or (None, None))[1].any()  # type: ignore[union-attr]


def test_sin_instancias_devuelve_mascaras_vacias_y_no_none() -> None:
    adapter, _ = segmenter(labels_for(), masks_for())

    result = adapter.curb(photo())

    assert result is not None and not result[0].any() and not result[1].any()


def test_la_columna_de_fondo_no_cuenta() -> None:
    adapter, _ = segmenter(labels_for((2, ON)), masks_for("all"))

    result = adapter.curb(photo())

    assert result is not None and not result[0].any() and not result[1].any()


@pytest.mark.parametrize(
    ("labels", "masks"),
    [
        (np.zeros((QUERIES, CLASS_COLUMNS), np.float32), masks_for()),
        (np.zeros((1, QUERIES, 1), np.float32), masks_for()),
        (labels_for(), np.zeros((1, QUERIES, MASK_SIDE), np.float32)),
        (labels_for(), np.zeros((1, QUERIES + 1, MASK_SIDE, MASK_SIDE), np.float32)),
        (np.full((1, QUERIES, CLASS_COLUMNS), np.nan, np.float32), masks_for()),
        (labels_for(), np.full((1, QUERIES, MASK_SIDE, MASK_SIDE), np.inf, np.float32)),
    ],
)
def test_salida_invalida_devuelve_none(labels: NDArray[Any], masks: NDArray[Any]) -> None:
    adapter, _ = segmenter(labels, masks)

    assert adapter.curb(photo()) is None
    with pytest.raises(ValueError, match="labels|masks|finita"):
        decode_class_masks(labels, masks, (30, 50), 0.5)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_imagen_invalida_devuelve_none_sin_correr_el_modelo(shape: tuple[int, ...]) -> None:
    adapter, session = segmenter(labels_for(), masks_for())

    assert adapter.curb(np.zeros(shape, np.uint8)) is None
    assert session.calls == []


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    adapter = RfDetrSegCurbSegmenter(
        FakeSession(RuntimeError("detalle interno con datos de la imagen")), "input", (16, 24), 0.5, "v"
    )

    with caplog.at_level(logging.WARNING):
        assert adapter.curb(photo()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del segmentador de rebajes"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "detalle interno" not in caplog.text


# --- registro: un ONNX diminuto con la misma interfaz que RF-DETR-Seg ---------------------------------------


def tiny_curb(names: Sequence[str] = OUTPUT_NAMES, classes: int = CLASS_COLUMNS) -> bytes:
    constants = [
        np.zeros((1, QUERIES, 4), np.float32),
        labels_for((CURB, ON), (CUT, ON))[..., :classes],
        masks_for("all", "left"),
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", [f"const_{i}"], [name]) for i, name in enumerate(names)],
        "tiny_curb",
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
        "ports": ["curb"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2026},
        "source": "generado en el test",
        "commercial": "yes",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_curb() -> None:
    assert ("curb", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_segmenta(tmp_path: Path) -> None:
    weights = tiny_curb()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"curb": MODEL_NAME}, manifest, models_dir)

    assert models.versions()["curb"] == f"{MODEL_NAME}@{hashlib.sha256(weights).hexdigest()[:12]}"
    result = models.curb.curb(photo(40, 60))
    assert result is not None
    curb, cut = result
    assert curb.shape == (40, 60) and curb.all() and cut[:, :25].all() and not cut[:, 35:].any()


def test_el_umbral_llega_por_los_parametros(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_curb())

    strict = build_models({"curb": MODEL_NAME}, manifest, models_dir, params={"curb_min_conf": 0.999})

    result = strict.curb.curb(photo())
    assert result is not None and not result[0].any() and not result[1].any()  # confianza ≈ 0,9975


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_un_umbral_fuera_de_rango_no_deja_arrancar(tmp_path: Path, value: float) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_curb())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"curb": MODEL_NAME}, manifest, models_dir, params={"curb_min_conf": value})


@pytest.mark.parametrize(
    "weights",
    [tiny_curb(names=("dets", "labels", "otra")), tiny_curb(classes=1)],
    ids=["otra salida", "faltan clases"],
)
def test_un_archivo_que_no_es_el_de_rebajes_no_deja_arrancar(tmp_path: Path, weights: bytes) -> None:
    manifest, models_dir = write_env(tmp_path, weights)

    with pytest.raises(ModelSetupError):
        build_models({"curb": MODEL_NAME}, manifest, models_dir)


def test_el_umbral_viaja_a_los_adaptadores(base_env: pytest.MonkeyPatch) -> None:
    assert adapter_params(load_settings())["curb_min_conf"] == 0.5

    base_env.setenv("CURB_MIN_CONF", "0.7")
    assert adapter_params(load_settings())["curb_min_conf"] == 0.7
