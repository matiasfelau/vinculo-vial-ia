"""tools/eval/face.py: decodificación de SCRFD y RetinaFace, métricas y criterio, con datos sintéticos.

No hay pesos ni fotos reales: las salidas de los modelos se fabrican a mano para que den cajas conocidas.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from vvi_ia.domain.model import Box

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_face", REPO_ROOT / "tools" / "eval" / "face.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_face"] = module
    spec.loader.exec_module(module)
    return module


face = load()


def box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, conf=conf, label="face")


# --- utilidades de cajas -------------------------------------------------------------------------------


def test_nms_deja_la_de_mayor_confianza_entre_las_que_se_pisan() -> None:
    dets = np.array(
        [[10, 10, 50, 50, 0.6], [12, 12, 52, 52, 0.9], [100, 100, 140, 140, 0.5]],
        np.float32,
    )

    kept = face.nms(dets)

    assert kept == [1, 2]


def test_nms_sin_cajas() -> None:
    assert face.nms(np.empty((0, 5), np.float32)) == []


def test_el_lienzo_deja_la_foto_arriba_a_la_izquierda_y_devuelve_la_escala() -> None:
    img = np.full((100, 200, 3), 255, np.uint8)

    canvas, scale = face.letterbox(img, 64)

    assert canvas.shape == (64, 64, 3) and scale == pytest.approx(0.32)
    assert (
        canvas[10, 10].tolist() == [255.0] * 3 and canvas[60, 10].tolist() == [0.0] * 3
    )  # abajo, el relleno


def test_el_lienzo_rellena_con_el_valor_pedido() -> None:
    canvas, _ = face.letterbox(np.zeros((10, 40, 3), np.uint8), 16, fill=(7, 8, 9))

    assert canvas[15, 0].tolist() == [7.0, 8.0, 9.0]


def test_to_boxes_recorta_a_la_foto_y_descarta_las_degeneradas() -> None:
    dets = np.array([[-5, -5, 30, 30, 0.7], [90, 90, 120, 120, 0.9], [10, 10, 10, 40, 0.5]], np.float32)

    boxes = face.to_boxes(dets, 100, 100)

    assert [(b.x1, b.y1, b.x2, b.y2) for b in boxes] == [(90.0, 90.0, 100.0, 100.0), (0.0, 0.0, 30.0, 30.0)]
    assert boxes[0].conf == pytest.approx(0.9) and all(b.label == "face" for b in boxes)


# --- SCRFD ---------------------------------------------------------------------------------------------


class FakeSession:
    def __init__(self, outputs: list[Any], input_name: str = "input.1") -> None:
        self.outputs = outputs
        self.input_names = (input_name,)
        self.feeds: list[Any] = []

    def run(self, names: Any, feed: Any) -> list[Any]:
        self.feeds.append(feed)
        return self.outputs


def scrfd(session: Any | None = None, size: int = 640) -> Any:
    detector = face.Scrfd.__new__(face.Scrfd)
    detector._session = session
    detector._size = size
    return detector


def scrfd_outputs(
    hits: dict[tuple[int, int, int, int], tuple[float, tuple[float, float, float, float]]],
    size: int = 640,
    kps: bool = False,
) -> list[Any]:
    """Salidas de un SCRFD: `hits[(stride, fila, columna, ancla)] = (confianza, distancias en píxeles)`."""
    scores, boxes = [], []
    for stride in (8, 16, 32):
        cells = size // stride
        scores.append(np.zeros((cells * cells * 2, 1), np.float32))
        boxes.append(np.zeros((cells * cells * 2, 4), np.float32))
    for (stride, row, column, anchor), (score, distances) in hits.items():
        index = {8: 0, 16: 1, 32: 2}[stride]
        cells = size // stride
        position = (row * cells + column) * 2 + anchor
        scores[index][position, 0] = score
        boxes[index][position] = (
            np.array(distances, np.float32) / stride
        )  # el modelo predice distancias / stride
    extra = [np.zeros((s.shape[0], 10), np.float32) for s in scores] if kps else []
    return [*scores, *boxes, *extra]


def test_scrfd_decodifica_la_caja_desde_el_centro_del_ancla() -> None:
    outputs = scrfd_outputs({(8, 10, 20, 1): (0.9, (16, 16, 16, 16))})  # centro en (160, 80)

    (det,) = scrfd().decode(outputs, scale=1.0)

    assert det[:4].tolist() == [144.0, 64.0, 176.0, 96.0] and det[4] == pytest.approx(0.9)


def test_scrfd_devuelve_las_cajas_en_la_escala_de_la_foto_original() -> None:
    outputs = scrfd_outputs({(16, 5, 5, 0): (0.8, (32, 32, 32, 32))})  # centro en (80, 80)

    (det,) = scrfd().decode(outputs, scale=0.5)

    assert det[:4].tolist() == [96.0, 96.0, 224.0, 224.0]


def test_scrfd_filtra_por_confianza_y_aplica_nms() -> None:
    outputs = scrfd_outputs(
        {
            (8, 10, 20, 0): (0.9, (16, 16, 16, 16)),
            (8, 10, 20, 1): (0.5, (16, 16, 17, 16)),  # casi la misma caja: se descarta
            (32, 3, 3, 0): (0.01, (64, 64, 64, 64)),  # bajo el piso de confianza
            (32, 10, 10, 1): (0.7, (64, 64, 64, 64)),
        }
    )

    dets = scrfd().decode(outputs, scale=1.0)

    assert [round(float(s), 2) for s in dets[:, 4]] == [0.9, 0.7]


def test_scrfd_sin_nada_devuelve_una_matriz_vacia() -> None:
    dets = scrfd().decode(scrfd_outputs({}), scale=1.0)

    assert dets.shape == (0, 5)


def test_scrfd_con_puntos_de_referencia_ignora_las_nueve_salidas_extra() -> None:
    outputs = scrfd_outputs({(8, 1, 1, 0): (0.9, (8, 8, 8, 8))}, kps=True)

    (det,) = scrfd().decode(outputs, scale=1.0)

    assert len(outputs) == 9 and det[:4].tolist() == [0.0, 0.0, 16.0, 16.0]


def test_scrfd_detect_normaliza_en_rgb_y_devuelve_cajas_en_pixeles_de_la_foto() -> None:
    session = FakeSession(scrfd_outputs({(8, 10, 20, 0): (0.9, (16, 16, 16, 16))}))
    detector = scrfd(session)
    img = np.zeros((320, 640, 3), np.uint8)
    img[..., 0] = 255  # rojo puro en RGB

    boxes = detector.detect(img)

    blob = session.feeds[0]["input.1"]
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert blob[0, 0, 0, 0] == pytest.approx((255 - 127.5) / 128) and blob[0, 1, 0, 0] == pytest.approx(
        -127.5 / 128
    )
    assert blob[0, 0, 600, 0] == pytest.approx(-127.5 / 128)  # el relleno de abajo
    assert boxes is not None and len(boxes) == 1 and boxes[0].conf == pytest.approx(0.9)
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (
        144.0,
        64.0,
        176.0,
        96.0,
    )  # escala 1,0 (ancho 640)


def test_scrfd_exige_seis_o_nueve_salidas(tmp_path: Path) -> None:
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "otro.onnx"
    path.write_bytes(model.SerializeToString())

    with pytest.raises(ValueError, match="6 o 9 salidas"):
        face.Scrfd(path)


# --- RetinaFace ----------------------------------------------------------------------------------------


def retina(session: Any | None = None, size: int = 640) -> Any:
    detector = face.RetinaFace.__new__(face.RetinaFace)
    detector._session = session
    detector._size = size
    detector._priors = face.RetinaFace.priors(size)
    return detector


def test_retinaface_tiene_las_16800_anclas_de_640_en_el_orden_del_repositorio() -> None:
    priors = face.RetinaFace.priors(640)

    assert priors.shape == (16800, 4)
    assert priors[0].tolist() == pytest.approx([4 / 640, 4 / 640, 16 / 640, 16 / 640])
    assert priors[1].tolist() == pytest.approx(
        [4 / 640, 4 / 640, 32 / 640, 32 / 640]
    )  # la segunda talla, misma posición
    assert priors[2][:2].tolist() == pytest.approx([12 / 640, 4 / 640])  # la columna siguiente, misma fila
    assert priors[80 * 80 * 2][2] == pytest.approx(64 / 640)  # arranca el stride 16


def test_retinaface_sin_correccion_la_caja_es_el_ancla() -> None:
    detector = retina()
    loc, conf = np.zeros((16800, 4), np.float32), np.zeros((16800, 2), np.float32)
    anchor = 5000
    conf[anchor] = [0.1, 0.8]

    (det,) = detector.decode(loc, conf, scale=1.0)

    cx, cy, w, h = detector._priors[anchor] * 640
    assert det[:4].tolist() == pytest.approx([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], abs=1e-3)
    assert det[4] == pytest.approx(0.8)


def test_retinaface_aplica_las_varianzas_y_la_escala() -> None:
    detector = retina()
    loc, conf = np.zeros((16800, 4), np.float32), np.zeros((16800, 2), np.float32)
    anchor = 100
    loc[anchor] = [
        1.0,
        -1.0,
        np.log(2) / 0.2,
        0.0,
    ]  # corre 0,1 de ancho a la derecha, 0,1 de alto arriba y duplica el ancho
    conf[anchor] = [0.0, 0.9]
    cx, cy, w, h = detector._priors[anchor] * 640

    (det,) = detector.decode(loc, conf, scale=0.5)

    new_cx, new_cy, new_w = cx + 0.1 * w, cy - 0.1 * h, 2 * w
    expected = [new_cx - new_w / 2, new_cy - h / 2, new_cx + new_w / 2, new_cy + h / 2]
    assert det[:4].tolist() == pytest.approx([v / 0.5 for v in expected], abs=1e-2)


def test_retinaface_usa_la_segunda_columna_como_confianza_y_filtra() -> None:
    detector = retina()
    loc = np.zeros((16800, 4), np.float32)
    conf = np.zeros((16800, 2), np.float32)
    conf[:, 0] = 0.99  # la columna 0 es "no cara": no cuenta
    conf[7000] = [0.4, 0.6]

    dets = detector.decode(loc, conf, scale=1.0)

    assert dets.shape == (1, 5) and dets[0, 4] == pytest.approx(0.6)


def test_retinaface_sin_nada_devuelve_una_matriz_vacia() -> None:
    dets = retina().decode(np.zeros((16800, 4), np.float32), np.zeros((16800, 2), np.float32), 1.0)

    assert dets.shape[0] == 0


def test_retinaface_detect_resta_la_media_en_bgr_y_reconoce_las_salidas_por_su_eje() -> None:
    loc, conf, landms = (
        np.zeros((1, 16800, 4), np.float32),
        np.zeros((1, 16800, 2), np.float32),
        np.zeros((1, 16800, 10), np.float32),
    )
    conf[0, 3000] = [0.2, 0.7]
    session = FakeSession([landms, conf, loc], "input0")  # en otro orden: se reconocen por su último eje
    detector = retina(session)
    img = np.zeros((320, 640, 3), np.uint8)
    img[..., 0] = 200  # rojo en RGB → el canal 2 en BGR

    boxes = detector.detect(img)

    blob = session.feeds[0]["input0"]
    assert blob.shape == (1, 3, 640, 640)
    assert blob[0, 2, 0, 0] == pytest.approx(200 - 123) and blob[0, 0, 0, 0] == pytest.approx(-104)
    assert blob[0, 0, 600, 0] == 0.0  # el relleno es la media (cero tras restarla)
    assert boxes is not None and len(boxes) == 1 and boxes[0].conf == pytest.approx(0.7)


def test_retinaface_sin_las_salidas_esperadas() -> None:
    detector = retina(FakeSession([np.zeros((1, 16800, 7), np.float32)]))

    with pytest.raises(ValueError, match="RetinaFace"):
        detector.detect(np.zeros((40, 40, 3), np.uint8))


# --- YuNet ---------------------------------------------------------------------------------------------


def test_yunet_arma_las_cajas_desde_las_filas_de_15_valores(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class FakeYN:
        @staticmethod
        def create(
            model: str, config: str, size: tuple[int, int], score: float, nms: float, top_k: int
        ) -> Any:
            calls["create"] = (model, config, score, nms, top_k)
            return FakeYN()

        def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
            calls["size"] = size

        def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
            calls["image_shape"] = image.shape
            row = np.zeros(15, np.float32)
            row[:4], row[14] = [10, 20, 30, 40], 0.85
            return 1, np.array([row])

    monkeypatch.setattr(face.cv2, "FaceDetectorYN", FakeYN)
    detector = face.YuNet(Path("yunet.onnx"))

    boxes = detector.detect(np.zeros((90, 120, 3), np.uint8))

    assert calls["size"] == (120, 90) and calls["image_shape"] == (90, 120, 3) and calls["create"][1] == ""
    assert boxes is not None and (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (
        10.0,
        20.0,
        40.0,
        60.0,
    )
    assert boxes[0].conf == pytest.approx(0.85)


def test_yunet_sin_caras(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYN:
        @staticmethod
        def create(*args: Any) -> Any:
            return FakeYN()

        def setInputSize(self, size: Any) -> None:  # noqa: N802
            pass

        def detect(self, image: Any) -> tuple[int, None]:
            return 0, None

    monkeypatch.setattr(face.cv2, "FaceDetectorYN", FakeYN)

    assert face.YuNet(Path("y.onnx")).detect(np.zeros((10, 10, 3), np.uint8)) == []


# --- métricas ------------------------------------------------------------------------------------------


def test_iou_y_match() -> None:
    assert face.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert face.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    hits, used = face.match([(0, 0, 10, 10)], [box(0, 0, 10, 10, 0.9), box(1, 1, 10, 10, 0.8)])
    assert hits == [True, False] and used == {0}
    assert face.match([(0, 0, 10, 10)], [box(6, 0, 16, 10)]) == ([False], set())


def test_ap_perfecto_y_con_un_falso_positivo_arriba() -> None:
    truths = {1: [(0, 0, 10, 10)], 2: [(0, 0, 10, 10)]}
    perfect = {1: [box(0, 0, 10, 10, 0.9)], 2: [box(0, 0, 10, 10, 0.8)]}
    worse = {1: [box(0, 0, 10, 10, 0.7)], 2: [box(0, 0, 10, 10, 0.6), box(50, 50, 60, 60, 0.95)]}

    assert face.average_precision(truths, perfect) == 1.0
    assert face.average_precision(truths, worse) < 1.0
    assert face.average_precision(truths, {1: [], 2: []}) == 0.0
    assert face.average_precision({1: []}, {1: []}) is None


def test_puntos_de_operacion() -> None:
    truths = {1: [(0, 0, 10, 10), (50, 50, 60, 62)], 2: [(0, 0, 40, 40)]}  # la primera es chica (10 px)
    predictions = {
        1: [box(0, 0, 10, 10, 0.9), box(200, 200, 220, 220, 0.4)],
        2: [box(0, 0, 40, 40, 0.3)],
    }

    rows = {row["umbral"]: row for row in face.operating_points(truths, predictions)}

    assert rows[0.25]["recall"] == pytest.approx(2 / 3, abs=1e-4) and rows[0.25]["fp_por_foto"] == 0.5
    assert rows[0.5]["recall"] == pytest.approx(1 / 3, abs=1e-4) and rows[0.5]["fp_por_foto"] == 0.0
    assert rows[0.5]["recall_chicas"] == 0.5  # de las dos chicas (10 y 10×12), encontró una
    assert rows[0.5]["precision"] == 1.0


def test_recall_a_un_tope_de_falsos_positivos() -> None:
    rows = [
        {"umbral": 0.1, "recall": 0.95, "fp_por_foto": 2.0},
        {"umbral": 0.3, "recall": 0.90, "fp_por_foto": 0.4},
        {"umbral": 0.5, "recall": 0.80, "fp_por_foto": 0.05},
    ]

    assert face.recall_at(rows, 0.5)["umbral"] == 0.3  # type: ignore[index]
    assert face.recall_at(rows, 0.1)["umbral"] == 0.5  # type: ignore[index]
    assert face.recall_at(rows, 0.01) is None


def report_with(recalls: dict[str, float]) -> dict[str, Any]:
    return {
        name: {"puntos": [{"umbral": 0.3, "recall": recall, "fp_por_foto": 0.05}]}
        for name, recall in recalls.items()
    }


def test_rfdetr_queda_si_su_recall_esta_cerca_del_de_retinaface() -> None:
    result = face.verdict(report_with({"rfdetr": 0.90, "retinaface_r50": 0.92}))

    assert result["rfdetr_queda"] is True and result["diferencia_con_la_referencia"] == pytest.approx(-0.02)


def test_rfdetr_no_queda_si_pierde_demasiado() -> None:
    result = face.verdict(report_with({"rfdetr": 0.80, "retinaface_r50": 0.92}))

    assert result["rfdetr_queda"] is False


def test_sin_la_referencia_no_se_puede_decidir() -> None:
    result = face.verdict(report_with({"rfdetr": 0.9}))

    assert result["rfdetr_queda"] is None and "RetinaFace" in result["motivo"]


def test_si_scrfd_gana_se_avisa_de_su_licencia() -> None:
    result = face.verdict(report_with({"rfdetr": 0.85, "retinaface_r50": 0.9, "scrfd": 0.95}))

    assert result["scrfd_gana"] is True and "escalable" in result["nota_scrfd"]
    assert "scrfd_gana" not in face.verdict(report_with({"rfdetr": 0.9, "scrfd": 0.91}))


def test_las_licencias_de_cada_candidato() -> None:
    info = face.CANDIDATE_INFO

    assert info["rfdetr"]["commercial"] == "yes" and info["rfdetr"]["puede_entrar_al_servicio"] is True
    assert info["scrfd"]["commercial"] == "escalable"
    assert (
        info["retinaface_r50"]["commercial"] == "no"
        and info["retinaface_r50"]["puede_entrar_al_servicio"] is False
    )
    assert info["yunet"]["commercial"] == "no" and info["yunet"]["puede_entrar_al_servicio"] is False


def test_evaluate_junta_licencia_ap_y_recall_a_cada_tope() -> None:
    truths = {1: [(0, 0, 10, 10)]}

    report = face.evaluate(truths, {"yunet": {1: [box(0, 0, 10, 10, 0.9)]}})

    assert report["yunet"]["commercial"] == "no" and report["yunet"]["ap50"] == 1.0
    assert set(report["yunet"]["recall_a_fp"]) == {"0.1", "0.5", "1.0"}


# --- línea de comandos ---------------------------------------------------------------------------------


class FixedDetector:
    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def detect(self, img: np.ndarray) -> list[Box] | None:
        return list(self.boxes)


def make_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    images, annotations = [], []
    for index, name in enumerate(("a.jpg", "b.jpg"), start=1):
        cv2.imwrite(str(folder / name), np.full((60, 100, 3), 100, np.uint8))
        images.append({"id": index, "file_name": name, "width": 100, "height": 60})
        annotations.append({"id": index, "image_id": index, "category_id": 7, "bbox": [10, 10, 20, 20]})
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 7, "name": "rostro"}]}
    (folder / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_main_de_punta_a_punta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    make_dataset(root / "face")
    factory = lambda args: {  # noqa: E731
        "rfdetr": FixedDetector([box(10, 10, 30, 30, 0.9)]),
        "retinaface_r50": FixedDetector([box(10, 10, 30, 30, 0.8)]),
    }

    code = face.main(
        ["--data-dir", str(root), "--out-dir", str(tmp_path / "out"), "--yunet-model", "y.onnx"], factory
    )

    assert code == 0
    report = json.loads((tmp_path / "out" / "face_report.json").read_text(encoding="utf-8"))
    assert report["fotos"] == 2 and report["caras"] == 2 and report["caras_chicas"] == 2  # 20 px < 32
    assert set(report["candidatos"]) == {"rfdetr", "retinaface_r50"}
    assert report["criterio"]["rfdetr_queda"] is True
    assert "latencia_ms_mediana" in report["candidatos"]["rfdetr"]
    shown = capsys.readouterr().out
    assert "Criterio" in shown and "a.jpg" not in shown


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = face.main(
        ["--dataset-dir", str(REPO_ROOT / "x"), "--out-dir", str(REPO_ROOT / "y"), "--yunet-model", "y.onnx"]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_main_sin_anotaciones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = face.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--yunet-model", "y.onnx"]
    )

    assert code == 1 and "Falta _annotations.coco.json" in capsys.readouterr().err


def test_main_si_un_detector_no_se_puede_cargar(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_dataset(tmp_path / "face")

    def broken(args: Any) -> Any:
        raise ValueError("se esperaba un SCRFD de 6 o 9 salidas")

    code = face.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--scrfd-model", "s.onnx"], broken
    )

    assert code == 1 and "No se pudo empezar" in capsys.readouterr().err


def test_hace_falta_al_menos_un_detector(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        face.parse_args(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path)])


def test_rfdetr_necesita_la_carpeta_de_pesos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELS_DIR", raising=False)

    with pytest.raises(SystemExit):
        face.parse_args(
            ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path), "--rfdetr-model", "rfdetr_openimages"]
        )


def test_los_candidatos_se_arman_segun_las_opciones(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: list[str] = []
    monkeypatch.setattr(face, "YuNet", lambda path: created.append(f"yunet:{path.name}") or "yunet")
    monkeypatch.setattr(face, "Scrfd", lambda path, size: created.append(f"scrfd:{size}") or "scrfd")
    monkeypatch.setattr(face, "RetinaFace", lambda path, size: created.append(f"retina:{size}") or "retina")
    args = face.parse_args(
        [
            *("--data-dir", str(tmp_path), "--out-dir", str(tmp_path)),
            *("--yunet-model", "y.onnx", "--scrfd-model", "s.onnx"),
            *("--retinaface-model", "r.onnx", "--size", "320"),
        ]
    )  # fmt: skip

    candidates = face.build_candidates(args)

    assert candidates == {"scrfd": "scrfd", "retinaface_r50": "retina", "yunet": "yunet"}
    assert created == ["scrfd:320", "retina:320", "yunet:y.onnx"]


# --- tamaños y mosaico ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (10, "chicas"),
        (31.9, "chicas"),
        (32, "medianas"),
        (96, "medianas"),
        (96.1, "grandes"),
        (400, "grandes"),
    ],
)
def test_el_tamano_se_mide_por_el_lado_menor(side: float, expected: str) -> None:
    assert face.size_of((0, 0, side, 500)) == expected
    assert face.size_of((0, 0, 500, side)) == expected


def test_los_puntos_de_operacion_dan_el_recall_por_tamano() -> None:
    truths = {1: [(0, 0, 20, 20), (100, 100, 160, 160), (200, 200, 400, 400)]}  # chica, mediana, grande
    predictions = {1: [box(100, 100, 160, 160, 0.9), box(200, 200, 400, 400, 0.8)]}

    row = {r["umbral"]: r for r in face.operating_points(truths, predictions)}[0.5]

    assert row["recall_por_tamano"] == {"chicas": 0.0, "medianas": 1.0, "grandes": 1.0}


def test_el_mosaico_agrega_lo_que_solo_se_ve_en_un_tile() -> None:
    class SmallOnlyInTiles:
        def detect(self, img: np.ndarray) -> list[Box]:
            return [box(5, 5, 25, 25, 0.9)] if img.shape[0] < 1000 else []

    tiled = face.Tiled(SmallOnlyInTiles(), tile=640, overlap=0.25)

    found = tiled.detect(np.zeros((1000, 1000, 3), np.uint8))

    assert found is not None and len(found) >= 1
    assert [b.conf for b in found] == sorted((b.conf for b in found), reverse=True)


def test_el_mosaico_propaga_el_fallo() -> None:
    class Broken:
        def detect(self, img: np.ndarray) -> None:
            return None

    assert face.Tiled(Broken()).detect(np.zeros((1000, 1000, 3), np.uint8)) is None


def test_tiling_suma_el_candidato_con_sufijo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(face, "YuNet", lambda path: FixedDetector([]))
    args = face.parse_args(
        [
            *("--data-dir", str(tmp_path), "--out-dir", str(tmp_path), "--yunet-model", "y.onnx"),
            *("--tiling", "yunet", "--tile-px", "512", "--overlap", "0.3"),
        ]
    )  # fmt: skip

    candidates = face.build_candidates(args)

    assert set(candidates) == {"yunet", "yunet+mosaico"} and isinstance(
        candidates["yunet+mosaico"], face.Tiled
    )
    args.tiling = ["scrfd"]
    with pytest.raises(ValueError, match="no se pasó su modelo"):
        face.build_candidates(args)


def test_evaluate_reconoce_el_candidato_con_mosaico() -> None:
    truths = {1: [(0, 0, 40, 40)]}

    report = face.evaluate(truths, {"rfdetr+mosaico": {1: [box(0, 0, 40, 40, 0.9)]}})

    assert report["rfdetr+mosaico"]["commercial"] == "yes"
    assert report["rfdetr+mosaico"]["nombre"].endswith("+ mosaico")
