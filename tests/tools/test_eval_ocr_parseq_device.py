"""tools/eval/ocr_parseq_device.py: PARSeq en el dispositivo (datos sintéticos, sin pesos ni red)."""

import ast
import csv
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from vvi_ia.adapters.models.parseq import CHARSET, NUM_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "eval" / "ocr_parseq_device.py"
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_ocr_parseq_device", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_ocr_parseq_device"] = module
    spec.loader.exec_module(module)
    return module


pd = load()
STEPS = 26
LABEL_COLUMNS = ["id", "archivo", "recorte", "texto", "formato", "luz", "estado", "revisar_moto"]


def logits_for(text: str, conf: float = 0.99) -> np.ndarray:
    """Logits `(1, 26, 95)` cuyo texto greedy es `text` y cuyos caracteres tienen probabilidad `conf`."""
    logits = np.zeros((1, STEPS, NUM_CLASSES), dtype=np.float32)
    peak = math.log(conf * (NUM_CLASSES - 1) / (1 - conf))  # softmax con una clase en `peak` y 94 en 0
    for step, char in enumerate(text):
        logits[0, step, CHARSET.index(char) + 1] = peak
    logits[0, len(text), 0] = 10.0  # fin de texto
    return logits


class FakeInput:
    name = "image"


class FakeSession:
    """Devuelve, en orden, los logits de cada lectura; cuenta las corridas."""

    def __init__(self, readings: list[tuple[str, float]]) -> None:
        self.readings = readings
        self.runs = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert feed["image"].shape == (1, 3, 32, 128)
        text, conf = self.readings[self.runs % len(self.readings)]
        self.runs += 1
        return [logits_for(text, conf)]


def reading(ident: str, label: str, text: str | None, conf: float, light: str = "dia") -> Any:
    return pd.Reading(ident=ident, light=light, label=label, text=text, conf=conf)


# ---------------------------------------------------------------- normalización por posición y formato


def test_normaliza_por_posicion_en_mercosur_y_corrige_letra_y_digito() -> None:
    probs = [0.9, 0.8, 0.99, 0.95, 0.97, 0.96, 0.9]

    # La B leída como 8 (posición de letra) y la O leída como 0 (posición de letra).
    assert pd.normalize("A8123C0", probs) == ("AB123CO", 0.8)
    # La I leída en posición de dígito se lee 1.
    assert pd.normalize("ABI23CD", probs) == ("AB123CD", 0.8)


def test_normaliza_el_formato_anterior() -> None:
    assert pd.normalize("ABC1I3", [0.9] * 6) == ("ABC113", 0.9)


def test_la_confianza_es_la_menor_de_la_patente_sin_el_ruido_de_alrededor() -> None:
    probs = [0.1, 0.1, 0.9, 0.8, 0.99, 0.95, 0.97, 0.96, 0.9, 0.2]

    assert pd.normalize("XYAB123CDZ", probs) == ("AB123CD", 0.8)


def test_sin_la_estructura_de_una_patente_no_hay_lectura() -> None:
    assert pd.normalize("HOLA", [0.9] * 4) is None
    assert pd.normalize("", []) is None
    assert pd.normalize("12345678", [0.9] * 8) is None


# ---------------------------------------------------------------- exacta, sin lectura, equivocada


def test_un_texto_exacto_equivocado_o_ausente_y_el_umbral_de_confianza() -> None:
    exact = reading("1", "AB123CD", "AB123CD", 0.5)
    wrong = reading("2", "AB123CD", "AB123CE", 0.9)
    none = reading("3", "AB123CD", None, 0.0)

    assert [pd.outcome(r, 0.0) for r in (exact, wrong, none)] == ["exacta", "equivocada", "sin_lectura"]
    # Con umbral, lo que no llega se vuelve "sin lectura", sea exacto o equivocado.
    assert [pd.outcome(r, 0.78) for r in (exact, wrong, none)] == ["sin_lectura", "equivocada", "sin_lectura"]
    assert pd.outcome(wrong, 0.95) == "sin_lectura"


def test_block_cuenta_porcentajes_y_equivocadas_a_un_caracter() -> None:
    rows = [
        reading("1", "AB123CD", "AB123CD", 0.9),
        reading("2", "AB123CD", "AB123CD", 0.9),
        reading("3", "AB123CD", "AB123CE", 0.9),  # a un carácter
        reading("4", "AB123CD", "ZZ999ZZ", 0.9),  # otra patente
        reading("5", "AB123CD", None, 0.0),
    ]

    result = pd.block(rows, 0.0)

    assert (result["exacta"], result["sin_lectura"], result["equivocada"]) == (2, 1, 2)
    assert result["equivocada_a_un_caracter"] == 1
    assert (result["exacta_pct"], result["sin_lectura_pct"], result["equivocada_pct"]) == (40.0, 20.0, 40.0)
    assert result["recortes"] == 5


def test_block_sin_filas_no_divide_por_cero() -> None:
    assert pd.block([], 0.0)["exacta_pct"] == 0.0


def test_summarize_separa_total_dia_y_noche_y_cada_umbral() -> None:
    rows = [
        reading("1", "AB123CD", "AB123CD", 0.95, "dia"),
        reading("2", "AB123CD", "AB123CD", 0.60, "dia"),
        reading("3", "AB123CD", "AB123CE", 0.90, "noche"),
    ]

    result = pd.summarize(rows)

    assert set(result) == {"sin_umbral", "umbral_0.5", "umbral_0.78", "umbral_0.9"}
    assert [result["sin_umbral"][g]["recortes"] for g in ("total", "dia", "noche")] == [3, 2, 1]
    assert result["sin_umbral"]["dia"]["exacta"] == 2 and result["sin_umbral"]["noche"]["equivocada"] == 1
    assert result["umbral_0.78"]["dia"]["exacta"] == 1 and result["umbral_0.78"]["dia"]["sin_lectura"] == 1


# ---------------------------------------------------------------- lectura de recortes con el modelo


def make_set(root: Path, readings_per_id: dict[str, str]) -> Path:
    """Set etiquetado con un recorte por id en `mlkit\\` y `mlkit-sin-margen\\`, más una ilegible."""
    set_dir = root / "set"
    for folder in pd.CROPS.values():
        (set_dir / folder).mkdir(parents=True)
        for ident in readings_per_id:
            ok, buffer = cv2.imencode(".png", np.full((20, 60, 3), 128, np.uint8))
            assert ok
            buffer.tofile(str(set_dir / folder / f"{ident}.png"))
    with (set_dir / "etiquetas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for index, (ident, text) in enumerate(readings_per_id.items()):
            writer.writerow(
                {
                    "id": ident,
                    "archivo": "train/p.jpg",
                    "recorte": f"recortes/{ident}.png",
                    "texto": text,
                    "formato": "mercosur",
                    "luz": "noche" if index == 0 else "dia",
                    "estado": "legible",
                    "revisar_moto": "",
                }
            )
        writer.writerow({**{c: "" for c in LABEL_COLUMNS}, "id": "999", "estado": "ilegible", "luz": "dia"})
    return set_dir


def test_load_labels_solo_las_legibles_de_auto(tmp_path: Path) -> None:
    set_dir = make_set(tmp_path, {"001": "AB123CD", "002": "EF456GH"})

    assert sorted(pd.load_labels(set_dir)) == ["001", "002"]


def test_read_crops_normaliza_y_compara_con_la_etiqueta_canonica(tmp_path: Path) -> None:
    set_dir = make_set(tmp_path, {"001": "AB-123 CD", "002": "EF456GH", "003": "IJ789KL"})
    labels = pd.load_labels(set_dir)
    crops = pd.load_crops(set_dir / "mlkit")
    session = FakeSession([("AB123CD", 0.99), ("EF456GH", 0.50), ("HOLA", 0.99)])

    rows = pd.read_crops(session, crops, labels)

    assert [r.ident for r in rows] == ["001", "002", "003"]
    assert rows[0].label == "AB123CD" and rows[0].text == "AB123CD" and rows[0].light == "noche"
    assert rows[1].text == "EF456GH" and rows[1].conf == pytest.approx(0.5, abs=0.01)
    assert rows[2].text is None and rows[2].conf == 0.0
    assert session.runs == 3


def test_write_readings_escribe_id_luz_lectura_y_confianza(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"

    pd.write_readings(
        path, [reading("1", "AB123CD", "AB123CD", 0.9), reading("2", "AB123CD", None, 0.0, "noche")]
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "id,luz,lectura,confianza",
        "1,dia,AB123CD,0.9000",
        "2,noche,,0.0000",
    ]


def test_evaluate_all_corre_los_cuatro_modelos_sobre_las_dos_variantes_de_recorte(tmp_path: Path) -> None:
    set_dir = make_set(tmp_path, {"001": "AB123CD", "002": "EF456GH"})
    onnx_dir, out_dir = tmp_path / "onnx", tmp_path / "out"
    out_dir.mkdir()
    opened: list[str] = []

    def factory(path: Path) -> FakeSession:
        opened.append(path.name)
        return FakeSession([("AB123CD", 0.99), ("EF456GX", 0.99)])  # la segunda sale equivocada

    results = pd.evaluate_all(set_dir, onnx_dir, out_dir, session_factory=factory)

    assert len(results) == 8 and len(opened) == 8
    assert set(opened) == {f"parseq-{s}-{q}.onnx" for s in ("tiny", "small") for q in ("fp32", "int8")}
    total = results["parseq-small-int8|sin_margen"]["sin_umbral"]["total"]
    assert (total["exacta"], total["equivocada"], total["equivocada_a_un_caracter"]) == (1, 1, 1)
    assert json.loads((out_dir / "results.json").read_text(encoding="utf-8")) == results
    assert len(list(out_dir.glob("rows-*.csv"))) == 8


def test_evaluate_all_falla_si_los_recortes_no_son_los_de_las_etiquetas(tmp_path: Path) -> None:
    set_dir = make_set(tmp_path, {"001": "AB123CD", "002": "EF456GH"})
    (set_dir / "mlkit" / "002.png").unlink()

    with pytest.raises(ValueError, match=r"mlkit\\"):
        pd.evaluate_all(
            set_dir, tmp_path / "onnx", tmp_path, session_factory=lambda _p: FakeSession([("A", 0.9)])
        )


# ---------------------------------------------------------------- latencia


def test_el_percentil_es_la_posicion_floor_q_por_n() -> None:
    assert pd.percentile(list(range(1, 21)), 0.95) == 20  # n = 20: índice 19
    assert pd.percentile(list(range(1, 11)), 0.95) == 10  # n = 10: índice 9 (no 10)
    assert pd.percentile([5.0], 0.95) == 5.0
    assert pd.percentile([3.0, 1.0, 2.0], 0.5) == 2.0


def test_timing_resume_en_ms() -> None:
    result = pd.timing([10.0, 20.0, 30.0, 40.0])

    assert result == {
        "mediana_ms": 25.0,
        "p95_ms": 40.0,
        "media_ms": 25.0,
        "minimo_ms": 10.0,
        "maximo_ms": 40.0,
    }


def test_time_model_calienta_mide_cada_pasada_y_separa_la_decodificacion() -> None:
    session = FakeSession([("AB123CD", 0.9)])
    tensors = [np.zeros((1, 3, 32, 128), np.float32)] * 4

    result = pd.time_model(session, tensors, warmup=2, rounds=3)

    assert session.runs == 2 + 3 * 4
    assert set(result) == {"inferencia", "decodificacion"}
    assert (
        0
        <= result["inferencia"]["minimo_ms"]
        <= result["inferencia"]["mediana_ms"]
        <= result["inferencia"]["p95_ms"]
    )
    assert result["decodificacion"]["mediana_ms"] >= 0


def test_time_preprocess_mide_cada_recorte() -> None:
    result = pd.time_preprocess([np.zeros((20, 60, 3), np.uint8)] * 5)

    assert result["maximo_ms"] >= result["mediana_ms"] >= result["minimo_ms"] >= 0


# ---------------------------------------------------------------- tamaños, AAR y APK


def test_file_sizes_da_el_tamano_y_el_comprimido(tmp_path: Path) -> None:
    zeros, noise = tmp_path / "zeros.onnx", tmp_path / "noise.onnx"
    zeros.write_bytes(bytes(2_000_000))
    noise.write_bytes(np.random.default_rng(0).bytes(2_000_000))

    assert pd.file_sizes(zeros)["mb"] == 2.0 and pd.file_sizes(zeros)["comprimido_mb"] < 0.05
    assert pd.file_sizes(noise)["comprimido_mb"] == pytest.approx(2.0, abs=0.01)  # lo aleatorio no comprime


def make_aar(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("jni/arm64-v8a/libonnxruntime.so", bytes(3_000_000))
        archive.writestr("jni/arm64-v8a/libonnxruntime4j_jni.so", bytes(500_000))
        archive.writestr("jni/x86/libonnxruntime.so", bytes(4_000_000))
        archive.writestr("classes.jar", bytes(100_000))  # no es una biblioteca nativa
        archive.writestr("jni/arm64-v8a/README.txt", b"no es una .so")


def test_aar_report_suma_las_bibliotecas_nativas_por_abi(tmp_path: Path) -> None:
    aar = tmp_path / "ort.aar"
    make_aar(aar)

    report = pd.aar_report(aar)

    assert list(report["abis"]) == ["arm64-v8a", "x86"]
    assert report["abis"]["arm64-v8a"]["mb"] == 3.5 and report["abis"]["x86"]["mb"] == 4.0
    assert report["abis"]["arm64-v8a"]["comprimido_mb"] < 0.1  # ceros: comprimen casi a nada
    assert report["todas_las_abis"]["mb"] == 7.5
    assert report["aar_mb"] == round(aar.stat().st_size / 1e6, 2)


def test_apk_estimate_suma_una_abi_de_ort_y_el_modelo() -> None:
    runtime = {"mb": 34.33, "comprimido_mb": 12.92}
    model = {"mb": 9.91, "comprimido_mb": 6.9}

    assert pd.apk_estimate(runtime, model) == {"en_disco_mb": 44.2, "descarga_mb": 19.8}


# ---------------------------------------------------------------- cuantización INT8


def test_quantize_deja_un_onnx_int8_mas_chico_que_lee_casi_igual(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(256, 256)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 256])],
        [onnx.numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    fp32, int8 = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    onnx.save(model, fp32)

    pd.quantize(fp32, int8)

    assert int8.stat().st_size < fp32.stat().st_size / 2
    ops = {node.op_type for node in onnx.load(int8).graph.node}
    assert "MatMulInteger" in ops  # los pesos y el producto son enteros
    sample = rng.normal(size=(1, 256)).astype(np.float32)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    expected = ort.InferenceSession(str(fp32), options).run(None, {"x": sample})[0]
    actual = ort.InferenceSession(str(int8), options).run(None, {"x": sample})[0]
    assert np.abs(actual - expected).max() < 0.1 * np.abs(expected).max()  # error de cuantización chico


# ---------------------------------------------------------------- tablas y línea de comandos


def table_inputs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    rows = [reading("1", "AB123CD", "AB123CD", 0.95), reading("2", "AB123CD", None, 0.0, "noche")]
    results = {f"{name}|{crop}": pd.summarize(rows) for name in pd.all_models() for crop in pd.CROPS}
    timing = {"mediana_ms": 28.29, "p95_ms": 39.25, "media_ms": 0, "minimo_ms": 0, "maximo_ms": 0}
    latency = {"modelos": {f"{name}|1": {"inferencia": timing} for name in pd.all_models()}}
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    for name in pd.all_models():
        (onnx_dir / f"{name}.onnx").write_bytes(bytes(2_500_000))
    return results, latency, onnx_dir


def test_report_tables_arma_las_dos_variantes_con_coma_decimal(tmp_path: Path) -> None:
    results, latency, onnx_dir = table_inputs(tmp_path)

    text = pd.report_tables(results, latency, onnx_dir)

    assert "**Recorte con margen**" in text and "**Recorte sin margen**" in text
    assert "| parseq-tiny-int8 | 50,0 / 50,0 / 0,0 | 100,0 / 0,0 / 0,0 | 0,0 / 100,0 / 0,0 | 2,5 |  |" in text
    # La latencia es de los recortes sin margen: solo esa tabla la lleva.
    assert text.count("28,3 ms (39)") == 4
    assert "Latencia" in text


def test_report_tables_sin_latencia_ni_onnx_deja_las_celdas_vacias(tmp_path: Path) -> None:
    results, _latency, _onnx = table_inputs(tmp_path)

    text = pd.report_tables(results, None, None)

    assert "ms (" not in text and "| parseq-small-fp32 | 50,0 / 50,0 / 0,0 |" in text


def test_main_report_imprime_las_tablas(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results, latency, onnx_dir = table_inputs(tmp_path)
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (tmp_path / "latency.json").write_text(json.dumps(latency), encoding="utf-8")

    code = pd.main(["report", "--out-dir", str(tmp_path), "--onnx-dir", str(onnx_dir)])

    assert code == 0 and "parseq-small-int8" in capsys.readouterr().out


def test_main_aar_imprime_el_peso_por_abi(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    aar = tmp_path / "ort.aar"
    make_aar(aar)

    assert pd.main(["aar", str(aar)]) == 0
    assert '"arm64-v8a"' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["eval", "latency"])
def test_main_se_niega_a_leer_o_escribir_dentro_del_repo(command: str, tmp_path: Path, capsys: Any) -> None:
    code = pd.main(
        [
            command,
            "--labeled-set",
            str(tmp_path),
            "--onnx-dir",
            str(tmp_path),
            "--out-dir",
            str(REPO_ROOT / "x"),
        ]
    )

    assert code == 1 and "dentro del repositorio" in capsys.readouterr().err


def test_main_avisa_si_falta_algo_en_vez_de_romper(tmp_path: Path, capsys: Any) -> None:
    assert pd.main(["report", "--out-dir", str(tmp_path)]) == 1
    assert "No se pudo" in capsys.readouterr().err


def test_la_herramienta_no_importa_torch_ni_la_cascada_al_cargarse() -> None:
    """CI no instala torch ni `pycocotools`: el export los carga solo al correr."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
    }

    assert not top_level & {"torch", "cascade", "detector", "huggingface_hub", "transformers"}
