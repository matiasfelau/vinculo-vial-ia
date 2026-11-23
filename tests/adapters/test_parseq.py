"""OCR PARSeq: preproceso, decodificación greedy, extracción de la patente y registro con un ONNX diminuto."""

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from vvi_ia.adapters.models.manifest import Commercial, load_manifest
from vvi_ia.adapters.models.parseq import (
    CHARSET,
    EOS_ID,
    MODEL_NAME,
    NUM_CLASSES,
    OUTPUT_NAMES,
    ParseqPlateReader,
    decode_greedy,
    extract_plate,
    extract_plate_chars,
    preprocess,
)
from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.domain.plate import PlateCheck, plate_check

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS = 26
HEIGHT, WIDTH = 32, 128


def token(char: str) -> int:
    return CHARSET.index(char) + 1  # el fin de texto ocupa la clase 0


def logits_for(text: str, probs: Sequence[float] | float = 0.99, steps: int = STEPS) -> NDArray[np.float32]:
    """Logits `(1, steps, 95)` cuyo argmax lee `text` y cuya probabilidad por carácter es `probs`."""
    per_char = [probs] * len(text) if isinstance(probs, float) else list(probs)
    out = np.zeros((1, steps, NUM_CLASSES), np.float32)
    for step, (char, p) in enumerate(zip(text, per_char, strict=True)):
        out[0, step, token(char)] = math.log(p / (1 - p) * (NUM_CLASSES - 1))
    if len(text) < steps:
        out[0, len(text), EOS_ID] = 10.0
    return out


class FakeSession:
    """Sesión falsa: devuelve la salida configurada y guarda lo que recibió."""

    def __init__(self, output: NDArray[Any] | Exception) -> None:
        self.output = output
        self.calls: list[tuple[Sequence[str] | None, Mapping[str, NDArray[Any]]]] = []

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        self.calls.append((output_names, input_feed))
        if isinstance(self.output, Exception):
            raise self.output
        return [self.output]


def reader(output: NDArray[Any] | Exception) -> tuple[ParseqPlateReader, FakeSession]:
    session = FakeSession(output)
    return ParseqPlateReader(session, "image", (HEIGHT, WIDTH), f"{MODEL_NAME}@abc"), session


def crop(height: int = 60, width: int = 180) -> NDArray[np.uint8]:
    return np.random.default_rng(5).integers(0, 256, (height, width, 3), dtype=np.uint8)


# --- preproceso --------------------------------------------------------------------------


def test_preproceso_devuelve_nchw_float32_a_32x128() -> None:
    tensor = preprocess(crop(90, 270), HEIGHT, WIDTH)

    assert tensor.shape == (1, 3, HEIGHT, WIDTH)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("shape", [(90, 270), (10, 40)])  # se achica o se agranda
def test_preproceso_normaliza_a_menos_uno_y_uno(shape: tuple[int, int]) -> None:
    black = preprocess(np.zeros((*shape, 3), np.uint8), HEIGHT, WIDTH)
    white = preprocess(np.full((*shape, 3), 255, np.uint8), HEIGHT, WIDTH)
    gray = preprocess(np.full((*shape, 3), 128, np.uint8), HEIGHT, WIDTH)

    np.testing.assert_allclose(black, -1.0)
    np.testing.assert_allclose(white, 1.0)
    np.testing.assert_allclose(gray, 128 / 127.5 - 1.0, atol=1e-6)


def test_preproceso_conserva_el_orden_de_los_canales_rgb() -> None:
    red = np.zeros((40, 100, 3), np.uint8)
    red[..., 0] = 255

    tensor = preprocess(red, HEIGHT, WIDTH)

    assert tensor[0, 0].min() == 1.0 and tensor[0, 1].max() == -1.0 and tensor[0, 2].max() == -1.0


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (10, 10, 1), (0, 10, 3)])
def test_preproceso_rechaza_recortes_que_no_son_rgb(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="RGB"):
        preprocess(np.zeros(shape, np.uint8), HEIGHT, WIDTH)


# --- decodificación greedy -----------------------------------------------------------------


def test_el_juego_de_caracteres_es_el_de_94_de_parseq() -> None:
    assert len(CHARSET) == 94 and NUM_CLASSES == 95
    assert CHARSET[:3] == "012" and CHARSET[10] == "a" and CHARSET[36] == "A"
    assert len(set(CHARSET)) == 94


def test_decodifica_hasta_el_fin_de_texto() -> None:
    text, probs = decode_greedy(logits_for("AB123CD"))

    assert text == "AB123CD"
    assert len(probs) == 7
    assert all(p == pytest.approx(0.99, abs=1e-3) for p in probs)


def test_ignora_lo_que_viene_despues_del_fin_de_texto() -> None:
    out = logits_for("AB")
    out[0, 5, token("Z")] = 20.0  # un carácter fuerte después del fin de texto

    assert decode_greedy(out)[0] == "AB"


def test_sin_fin_de_texto_usa_todos_los_pasos() -> None:
    out = np.zeros((1, 4, NUM_CLASSES), np.float32)
    for step, char in enumerate("WXYZ"):
        out[0, step, token(char)] = 9.0

    assert decode_greedy(out)[0] == "WXYZ"


def test_texto_vacio() -> None:
    assert decode_greedy(logits_for("")) == ("", [])


@pytest.mark.parametrize("char", ["0", "9", "a", "z", "A", "Z", "!", "~"])
def test_cada_clase_se_traduce_a_su_caracter(char: str) -> None:
    assert decode_greedy(logits_for(char))[0] == char


def test_la_probabilidad_es_la_del_caracter_elegido_y_no_incluye_el_fin_de_texto() -> None:
    _, probs = decode_greedy(logits_for("AB1", [0.99, 0.70, 0.95]))

    assert probs == pytest.approx([0.99, 0.70, 0.95], abs=2e-3)


@pytest.mark.parametrize(
    "logits",
    [
        np.zeros((1, STEPS, NUM_CLASSES - 1), np.float32),  # faltan clases
        np.zeros((STEPS, NUM_CLASSES), np.float32),  # sin eje de lote
        np.zeros((2, STEPS, NUM_CLASSES), np.float32),  # lote inesperado
        np.zeros((1, 0, NUM_CLASSES), np.float32),  # sin pasos
        np.full((1, STEPS, NUM_CLASSES), np.nan, np.float32),
        np.full((1, STEPS, NUM_CLASSES), np.inf, np.float32),
    ],
)
def test_logits_invalidos(logits: NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="logits|finita"):
        decode_greedy(logits)


# --- extracción de la patente --------------------------------------------------------------


def extract(text: str, prob: float = 0.9) -> tuple[str, float]:
    return extract_plate(text, [prob] * len(text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AB123CD", "AB123CD"),  # Mercosur
        ("ABC123", "ABC123"),  # formato anterior
        ("ARGAB123CD", "AB123CD"),  # con texto antes
        ("AB123CDARG", "AB123CD"),  # con texto después
        ("AB 123 CD", "AB123CD"),  # con separadores
        ("ab-123-cd", "AB123CD"),  # en minúsculas
        ("XABC123", "ABC123"),
        ("ABC123Y", "ABC123"),
        ("ARGA8123CD", "A8123CD"),  # una ventana del formato anterior ("RGA812") no le gana a la patente
        ("ARGENTINAAB123CD", "AB123CD"),
        ("A8123CD", "A8123CD"),  # 8 donde va una B (se devuelve como se leyó)
        ("ABI23CD", "ABI23CD"),  # I donde va un 1
        ("AB1Z3CD", "AB1Z3CD"),  # Z donde va un 2
        ("ABO23CD", "ABO23CD"),  # O donde va un 0
        ("AB123C0", "AB123C0"),  # 0 donde va una O
        ("A0123CD", "A0123CD"),  # 0 donde va una letra
    ],
)
def test_extrae_la_subcadena_con_formato_argentino(text: str, expected: str) -> None:
    assert extract(text)[0] == expected


@pytest.mark.parametrize("text", ["HELLO", "AB12", "12345678", "ABCDEFG", "", "!!!", "AB123", "A1B2C3D"])
def test_si_no_hay_patente_devuelve_el_texto_crudo(text: str) -> None:
    assert extract(text)[0] == text


def test_el_texto_crudo_se_devuelve_tal_cual_sin_canonicalizar() -> None:
    assert extract("ab-c")[0] == "ab-c"


def test_prefiere_la_ventana_que_necesita_menos_reinterpretaciones() -> None:
    # "AB123CD" encaja exacto; "AB1Z3CD" requiere reinterpretar la Z
    assert extract("AB1Z3CDAB123CD")[0] == "AB123CD"


def test_una_ventana_corta_dentro_de_otra_patente_no_gana_por_costar_cero() -> None:
    # En "ARGA8123CD", "RGA812" encaja sin reinterpretar nada, pero deja 4 caracteres afuera;
    # "A8123CD" reinterpreta la B/8 y deja 3
    text, _ = extract("ARGA8123CD")

    assert text == "A8123CD"


def test_ante_un_empate_gana_mercosur_y_luego_la_mas_a_la_izquierda() -> None:
    # "ABC123" y "AB123CD" no se pisan; con dos Mercosur exactos gana el primero
    assert extract("AB123CDXY456ZW")[0] == "AB123CD"
    # "ABC1234" tiene el formato anterior exacto en "ABC123"
    assert extract("ABC1234")[0] == "ABC123"


def test_la_confianza_es_la_de_los_caracteres_de_la_patente_y_no_la_del_resto() -> None:
    probs = [0.2, 0.3, 0.3, 0.99, 0.95, 0.9, 0.97, 0.96, 0.98, 0.99]  # "ARG" tiene baja confianza
    text, conf = extract_plate("ARGAB123CD", probs)

    assert text == "AB123CD"
    assert conf == pytest.approx(0.9)


def test_la_confianza_con_separadores_usa_los_indices_del_texto_original() -> None:
    probs = [0.99, 0.98, 0.1, 0.97, 0.96, 0.95, 0.1, 0.94, 0.93]  # los espacios valen 0.1
    text, conf = extract_plate("AB 123 CD", probs)

    assert text == "AB123CD"
    assert conf == pytest.approx(0.93)


def test_el_texto_crudo_tiene_la_confianza_minima_de_todo_el_texto() -> None:
    assert extract_plate("HELLO", [0.9, 0.8, 0.95, 0.99, 0.6])[1] == pytest.approx(0.6)


def test_texto_vacio_tiene_confianza_cero() -> None:
    assert extract_plate("", []) == ("", 0.0)


def test_lo_extraido_se_compara_bien_con_la_patente_reportada() -> None:
    text, conf = extract("ARGA8123CD")  # el OCR leyó "A8123CD": una B como 8

    assert plate_check("AB123CD", text, conf, 0.6) == PlateCheck.MATCH


# --- adaptador -----------------------------------------------------------------------------


def test_lee_la_patente_y_pasa_el_tensor_al_modelo() -> None:
    adapter, session = reader(logits_for("AB123CD", 0.98))

    result = adapter.read(crop())

    assert result is not None
    assert result[0] == "AB123CD"
    assert result[1] == pytest.approx(0.98, abs=2e-3)
    ((names, feed),) = session.calls
    assert tuple(names or ()) == OUTPUT_NAMES
    assert feed["image"].shape == (1, 3, HEIGHT, WIDTH)


def test_sin_texto_devuelve_vacio_con_confianza_cero_y_no_none() -> None:
    adapter, _ = reader(logits_for(""))

    assert adapter.read(crop()) == ("", 0.0)


def test_expone_la_version_del_modelo() -> None:
    adapter, _ = reader(logits_for(""))

    assert adapter.version == f"{MODEL_NAME}@abc"


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_recorte_invalido_devuelve_none_sin_correr_el_modelo(shape: tuple[int, ...]) -> None:
    adapter, session = reader(logits_for("AB123CD"))

    assert adapter.read(np.zeros(shape, np.uint8)) is None
    assert session.calls == []


def test_salida_invalida_devuelve_none() -> None:
    adapter, _ = reader(np.full((1, STEPS, NUM_CLASSES), np.nan, np.float32))

    assert adapter.read(crop()) is None


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    adapter, _ = reader(RuntimeError("AB123CD detalle interno"))

    with caplog.at_level(logging.WARNING):
        assert adapter.read(crop()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del OCR"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "AB123CD" not in caplog.text


def test_nunca_loguea_la_patente_leida(caplog: pytest.LogCaptureFixture) -> None:
    adapter, _ = reader(logits_for("AB123CD"))

    with caplog.at_level(logging.DEBUG):
        adapter.read(crop())

    assert "AB123CD" not in caplog.text


# --- registro: un ONNX diminuto con la misma interfaz que PARSeq ----------------------------


def tiny_parseq(text: str = "AB123CD", output_name: str = "logits", classes: int = NUM_CLASSES) -> bytes:
    """`(1, 3, 32, 128)` → logits constantes que leen `text`."""
    constant = logits_for(text)[..., :classes]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["const"], [output_name])],
        "tiny_parseq",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(constant.shape))],
        [numpy_helper.from_array(constant, "const")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def wrong_model() -> bytes:
    """Un ONNX válido que no es PARSeq."""
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
        "ports": ["ocr"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ECCV", "year": 2022},
        "source": "generado en el test",
        "commercial": "escalable",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_ocr() -> None:
    assert ("ocr", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_lee(tmp_path: Path) -> None:
    weights = tiny_parseq("ARGAB123CD")
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"ocr": MODEL_NAME}, manifest, models_dir)

    sha = hashlib.sha256(weights).hexdigest()
    assert models.versions()["ocr"] == f"{MODEL_NAME}@{sha[:12]}"
    result = models.plate.read(crop(90, 270))
    assert result is not None and result[0] == "AB123CD"
    assert 0.9 < result[1] <= 1.0


def test_un_archivo_que_no_es_parseq_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, wrong_model())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"ocr": MODEL_NAME}, manifest, models_dir)


def test_una_salida_con_otro_nombre_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_parseq(output_name="otra"))

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"ocr": MODEL_NAME}, manifest, models_dir)


def test_un_modelo_con_otras_clases_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_parseq(classes=NUM_CLASSES - 5))

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"ocr": MODEL_NAME}, manifest, models_dir)


# --- manifest versionado -------------------------------------------------------------------


def test_el_manifest_del_repo_declara_parseq_con_apache_2_y_la_fuente_de_los_pesos() -> None:
    """16/11/2026: pesos de baudm/parseq-small (Hugging Face, Apache-2.0), los mismos de la release v1.0.0."""
    entry = load_manifest(REPO_ROOT / "models" / "manifest.json").get(MODEL_NAME)

    assert entry is not None
    assert entry.ports == ("ocr",)
    assert entry.commercial == Commercial.YES
    assert (entry.paper.venue, entry.paper.year) == ("ECCV", 2022)
    assert entry.sha256 == "7ff186b1f7433e62980d01d647dce74ab338ebe43eefe6ec5e031a2c06315991"
    assert entry.license.startswith("Apache-2.0") and "baudm/parseq-small" in entry.license
    assert "huggingface.co/baudm/parseq-small" in entry.source
    assert "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00" in entry.source


# --- integración con los pesos reales (opt-in: requiere MODELS_DIR) --------------------------


def render_plate(text: str, height: int = 90, width: int = 270) -> NDArray[np.uint8]:
    """Imagen sintética con texto negro sobre gris claro: no hay ninguna patente real."""
    img = np.full((height, width, 3), 235, np.uint8)
    scale = height / 60.0
    thickness = max(1, int(2 * scale))
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


@pytest.mark.models
@pytest.mark.parametrize("text", ["AB123CD", "ABC123", "AE456FG"])
def test_pesos_reales_leen_texto_sintetico_con_formato_de_patente(text: str) -> None:
    models = build_models(
        {"ocr": MODEL_NAME}, REPO_ROOT / "models" / "manifest.json", Path(os.environ["MODELS_DIR"])
    )

    result = models.plate.read(render_plate(text))

    assert result is not None
    assert result[0] == text
    assert result[1] > 0.9
    assert models.versions()["ocr"].startswith(f"{MODEL_NAME}@")


def test_extract_plate_chars_devuelve_la_probabilidad_de_cada_caracter_de_la_patente() -> None:
    probs = [0.99, 0.98, 0.1, 0.97, 0.96, 0.95, 0.1, 0.94, 0.93]  # los espacios valen 0.1

    text, chars = extract_plate_chars("AB 123 CD", probs)

    assert text == "AB123CD"
    assert chars == [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]


def test_extract_plate_chars_sin_patente_devuelve_el_texto_crudo_con_todas_sus_probabilidades() -> None:
    assert extract_plate_chars("HELLO", [0.9, 0.8, 0.95, 0.99, 0.6]) == ("HELLO", [0.9, 0.8, 0.95, 0.99, 0.6])
