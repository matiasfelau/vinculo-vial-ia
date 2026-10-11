"""Embedder DINOv2: preproceso, token CLS, normalización L2 y registro (con un ONNX diminuto)."""

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
from onnx import TensorProto, helper

from vvi_ia.adapters.models.dinov2 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIDE,
    MODEL_NAME,
    DinoV2Embedder,
    cls_embedding,
    preprocess,
)
from vvi_ia.adapters.models.manifest import Commercial, load_manifest
from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.domain.reid import GRAY_FILL, cosine

REPO_ROOT = Path(__file__).resolve().parents[2]
CAR = (200, 40, 90)
DIM = 8


def normalized(rgb: Sequence[int]) -> NDArray[np.float32]:
    """Un color RGB tal como llega al modelo después de la normalización ImageNet."""
    return (np.array(rgb, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def scene(height: int = 300, width: int = 500) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Foto con ruido de fondo y un "auto" de color liso en forma de L (la caja no está llena)."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.bool_)
    mask[100:200, 100:400] = True
    mask[100:150, 250:400] = False
    img[mask] = CAR
    return img, mask


class FakeSession:
    """Sesión falsa: devuelve la salida configurada y guarda lo que recibió."""

    def __init__(self, output: NDArray[Any] | Exception) -> None:
        self.output = output
        self.feeds: list[Mapping[str, NDArray[Any]]] = []

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        self.feeds.append(input_feed)
        if isinstance(self.output, Exception):
            raise self.output
        return [self.output]


def tokens(cls: Sequence[float], patches: int = 4) -> NDArray[np.float32]:
    """Salida `(1, 1 + patches, D)` con el CLS dado y parches que apuntan a otro lado."""
    out = np.full((1, 1 + patches, len(cls)), -9.0, np.float32)
    out[0, 0] = cls
    return out


def embedder(output: NDArray[Any] | Exception) -> tuple[DinoV2Embedder, FakeSession]:
    session = FakeSession(output)
    return DinoV2Embedder(session, "pixel_values", f"{MODEL_NAME}@abc"), session


# --- preproceso --------------------------------------------------------------------------


def test_preproceso_devuelve_un_tensor_nchw_de_224() -> None:
    tensor = preprocess(*scene())

    assert tensor is not None
    assert tensor.shape == (1, 3, INPUT_SIDE, INPUT_SIDE)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


def test_preproceso_pone_el_fondo_en_gris_y_normaliza_con_imagenet() -> None:
    tensor = preprocess(*scene())
    assert tensor is not None

    # La caja de la máscara es 100×300 y se estira a 224×224: abajo a la izquierda hay auto y
    # arriba a la derecha está el hueco de la L, que tiene que quedar gris.
    car_pixel = tensor[0, :, 200, 20]
    background_pixel = tensor[0, :, 20, 200]
    np.testing.assert_allclose(car_pixel, normalized(CAR), atol=1e-5)
    np.testing.assert_allclose(background_pixel, normalized((GRAY_FILL,) * 3), atol=1e-5)


def test_preproceso_ignora_lo_que_queda_fuera_de_la_mascara() -> None:
    img, mask = scene()
    other = img.copy()
    other[~mask] = 255 - other[~mask]

    first, second = preprocess(img, mask), preprocess(other, mask)

    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)


def test_preproceso_agranda_un_recorte_chico() -> None:
    img = np.full((40, 60, 3), CAR, np.uint8)
    mask = np.zeros((40, 60), np.bool_)
    mask[10:20, 10:30] = True

    tensor = preprocess(img, mask)

    assert tensor is not None
    assert tensor.shape == (1, 3, INPUT_SIDE, INPUT_SIDE)
    np.testing.assert_allclose(tensor[0, :, 112, 112], normalized(CAR), atol=1e-4)


def test_preproceso_con_mascara_vacia_devuelve_none() -> None:
    img, mask = scene()

    assert preprocess(img, np.zeros_like(mask)) is None


def test_preproceso_no_modifica_la_imagen_original() -> None:
    img, mask = scene()
    before = img.copy()

    preprocess(img, mask)

    np.testing.assert_array_equal(img, before)


# --- token CLS y normalización L2 ----------------------------------------------------------


def test_embed_toma_el_token_cls_y_lo_normaliza_l2() -> None:
    adapter, session = embedder(tokens([3.0, 0.0, 4.0]))

    vector = adapter.embed(*scene())

    assert vector is not None
    assert vector.ndim == 1
    np.testing.assert_allclose(vector, [0.6, 0.0, 0.8], atol=1e-6)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)
    (feed,) = session.feeds
    assert list(feed) == ["pixel_values"]
    assert feed["pixel_values"].shape == (1, 3, INPUT_SIDE, INPUT_SIDE)


def test_el_embedding_sirve_para_la_similitud_coseno_del_dominio() -> None:
    a, _ = embedder(tokens([1.0, 2.0, 3.0]))
    b, _ = embedder(tokens([2.0, 4.0, 6.0]))
    img, mask = scene()

    first, second = a.embed(img, mask), b.embed(img, mask)

    assert first is not None and second is not None
    assert cosine(first, second) == pytest.approx(1.0)


def test_expone_la_version_del_modelo() -> None:
    adapter, _ = embedder(tokens([1.0]))

    assert adapter.version == f"{MODEL_NAME}@abc"


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 5, DIM), np.float32),  # vector nulo
        np.full((1, 5, DIM), np.nan, np.float32),
        np.full((1, 5, DIM), np.inf, np.float32),
        np.ones((1, DIM), np.float32),  # sin el eje de tokens
        np.ones((2, 5, DIM), np.float32),  # lote inesperado
        np.ones((1, 0, DIM), np.float32),  # sin tokens
    ],
)
def test_salida_que_no_sirve_como_embedding_devuelve_none(output: NDArray[Any]) -> None:
    assert cls_embedding(output) is None
    adapter, _ = embedder(output)
    assert adapter.embed(*scene()) is None


def test_mascara_vacia_devuelve_none_sin_correr_el_modelo() -> None:
    adapter, session = embedder(tokens([1.0, 0.0]))
    img, mask = scene()

    assert adapter.embed(img, np.zeros_like(mask)) is None
    assert session.feeds == []


def test_mascara_de_otro_tamano_devuelve_none() -> None:
    adapter, session = embedder(tokens([1.0, 0.0]))
    img, _ = scene()

    assert adapter.embed(img, np.ones((10, 10), np.bool_)) is None
    assert session.feeds == []


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    adapter, _ = embedder(RuntimeError("detalle interno con datos de la imagen"))

    with caplog.at_level(logging.WARNING):
        assert adapter.embed(*scene()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del embedder"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "detalle interno" not in caplog.text


# --- registro: un ONNX diminuto con la misma forma de entrada y salida ----------------------


def tiny_dinov2() -> bytes:
    """`(1, 3, 224, 224)` → `(1, 1, 3)`: el "CLS" es el promedio por canal de la entrada."""
    mean = helper.make_node("ReduceMean", ["pixel_values"], ["mean"], axes=[2, 3], keepdims=0)
    unsqueeze = helper.make_node("Unsqueeze", ["mean", "axis"], ["last_hidden_state"])
    axis = helper.make_tensor("axis", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(
        [mean, unsqueeze],
        "tiny_dinov2",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, INPUT_SIDE, INPUT_SIDE])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 1, 3])],
        [axis],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def wrong_model() -> bytes:
    """Un ONNX válido que no es el embedder: su entrada no acepta una imagen."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "wrong",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return bytes(model.SerializeToString())


def write_env(tmp_path: Path, weights: bytes) -> tuple[Path, Path]:
    models_dir = tmp_path / "weights"
    models_dir.mkdir()
    (models_dir / f"{MODEL_NAME}.onnx").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    entry = {
        "name": MODEL_NAME,
        "version": "de-prueba",
        "ports": ["embedder"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "TMLR", "year": 2024},
        "source": "generado en el test",
        "commercial": "yes",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_embedder() -> None:
    assert ("embedder", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_embebe(tmp_path: Path) -> None:
    weights = tiny_dinov2()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"embedder": MODEL_NAME}, manifest, models_dir)

    sha = hashlib.sha256(weights).hexdigest()
    assert models.versions()["embedder"] == f"{MODEL_NAME}@{sha[:12]}"
    img = np.full((50, 80, 3), CAR, np.uint8)
    vector = models.embedder.embed(img, np.ones((50, 80), np.bool_))
    assert vector is not None
    expected = normalized(CAR)
    np.testing.assert_allclose(vector, expected / np.linalg.norm(expected), atol=1e-4)


def test_un_archivo_que_no_es_el_embedder_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, wrong_model())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"embedder": MODEL_NAME}, manifest, models_dir)


# --- manifest versionado -------------------------------------------------------------------


def test_el_manifest_del_repo_declara_el_embedder() -> None:
    entry = load_manifest(REPO_ROOT / "models" / "manifest.json").get(MODEL_NAME)

    assert entry is not None
    assert entry.ports == ("embedder",)
    assert entry.commercial == Commercial.YES
    assert entry.license == "Apache-2.0"
    assert (entry.paper.venue, entry.paper.year) == ("TMLR", 2024)
    assert entry.sha256 is not None


def test_cada_modelo_del_manifest_tiene_adaptador_en_sus_puertos() -> None:
    manifest = load_manifest(REPO_ROOT / "models" / "manifest.json")

    missing = [
        (port, entry.name)
        for entry in manifest.models
        if entry.commercial != Commercial.NO
        for port in entry.ports
        if (port, entry.name) not in ADAPTERS
    ]

    assert missing == []


# --- integración con los pesos reales (opt-in: requiere MODELS_DIR) --------------------------


@pytest.mark.models
def test_pesos_reales_dan_un_vector_unitario_de_384_y_distinguen_imagenes() -> None:
    models = build_models(
        {"embedder": MODEL_NAME}, REPO_ROOT / "models" / "manifest.json", Path(os.environ["MODELS_DIR"])
    )
    rng = np.random.default_rng(0)
    mask = np.ones((240, 320), np.bool_)
    stripes = np.zeros((240, 320, 3), np.uint8)
    stripes[:, ::16] = 255
    noise = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    a, again, b = (models.embedder.embed(img, mask) for img in (stripes, stripes, noise))

    assert a is not None and again is not None and b is not None
    assert a.shape == (384,)
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-5)
    assert cosine(a, again) == pytest.approx(1.0, abs=1e-6)
    assert cosine(a, b) < 0.9
