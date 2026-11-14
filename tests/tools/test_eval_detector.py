"""tools/eval/detector.py: AP50 y recall combinados de varias clases (micro-promedio), coherentes con COCO."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("pycocotools")

from pycocotools.coco import COCO  # noqa: E402

from vvi_ia.domain.model import Box, BoxLabel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CAR, PLATE = str(BoxLabel.CAR), str(BoxLabel.PLATE)


def load() -> ModuleType:
    tools = str(REPO_ROOT / "tools" / "eval")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "eval_detector", REPO_ROOT / "tools" / "eval" / "detector.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_detector"] = module
    spec.loader.exec_module(module)
    return module


detector = load()
IDS = {CAR: 1, PLATE: 2}
THRESHOLDS = {CAR: 0.4, PLATE: 0.2}


def coco(truths: dict[int, list[tuple[int, tuple[float, float, float, float]]]]) -> COCO:
    """Un COCO en memoria: `imagen -> [(clase, (x1, y1, x2, y2))]`."""
    annotations: list[dict[str, Any]] = []
    for image_id, boxes in truths.items():
        for category_id, (x1, y1, x2, y2) in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    gt = COCO()
    gt.dataset = {
        "images": [{"id": i, "width": 1000, "height": 1000, "file_name": f"{i}.jpg"} for i in truths],
        "annotations": annotations,
        "categories": [{"id": 1, "name": CAR}, {"id": 2, "name": PLATE}],
    }
    gt.createIndex()
    return gt


def box(x1: float, y1: float, x2: float, y2: float, conf: float, label: str) -> Box:
    return Box(x1, y1, x2, y2, conf=conf, label=BoxLabel(label))


def test_average_precision_perfecta_y_sin_aciertos() -> None:
    assert detector.average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert detector.average_precision([(0.9, False)], 2) == 0.0
    assert detector.average_precision([], 2) == 0.0
    assert detector.average_precision([(0.9, True)], 0) == 0.0


def test_average_precision_castiga_el_falso_positivo_mejor_rankeado() -> None:
    # un FP con la mayor confianza y dos aciertos: precisión 1/2 y 2/3, envolvente 2/3 hasta recall 1
    ap = detector.average_precision([(0.9, False), (0.8, True), (0.7, True)], 2)

    assert ap == pytest.approx(2 / 3)


def test_average_precision_coincide_con_cocoeval_en_una_clase() -> None:
    gt = coco({1: [(1, (0, 0, 100, 100)), (1, (200, 200, 300, 300))], 2: [(1, (50, 50, 150, 150))]})
    predictions = {
        1: [box(0, 0, 100, 100, 0.9, CAR), box(600, 600, 700, 700, 0.8, CAR)],
        2: [box(50, 50, 150, 150, 0.7, CAR), box(200, 200, 300, 300, 0.3, CAR)],
    }
    detections = [
        {
            "image_id": i,
            "category_id": 1,
            "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],
            "score": b.conf,
        }
        for i, boxes in predictions.items()
        for b in boxes
    ]

    expected = detector.ap50(gt, detections, 1)
    micro = detector.combined(gt, predictions, THRESHOLDS, IDS, names=(CAR,))

    assert expected is not None
    assert micro["ap50"] == pytest.approx(expected, abs=0.01)


def test_combined_suma_aciertos_de_las_dos_clases_sin_mezclarlas() -> None:
    gt = coco({1: [(1, (0, 0, 100, 100)), (2, (10, 10, 40, 30))], 2: [(1, (0, 0, 100, 100))]})
    predictions = {
        1: [
            box(0, 0, 100, 100, 0.9, CAR),  # acierta
            box(10, 10, 40, 30, 0.9, CAR),  # una patente marcada como auto: no cuenta como patente
            box(10, 10, 40, 30, 0.5, PLATE),  # acierta
        ],
        2: [box(0, 0, 100, 100, 0.3, CAR)],  # bajo el umbral del auto (0,4): no cuenta
    }

    micro = detector.combined(gt, predictions, THRESHOLDS, IDS)

    assert micro["cajas_reales"] == 3
    assert micro["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert micro["precision"] == pytest.approx(2 / 3, abs=1e-4)  # 2 aciertos y 1 falso positivo
    assert micro["falsos_positivos"] == 1
    assert micro["clases"] == [CAR, PLATE]


def test_combined_usa_el_umbral_de_cada_clase_para_recall_y_todas_las_cajas_para_el_ap() -> None:
    gt = coco({1: [(1, (0, 0, 100, 100)), (2, (10, 10, 40, 30))]})
    predictions = {1: [box(0, 0, 100, 100, 0.3, CAR), box(10, 10, 40, 30, 0.25, PLATE)]}

    micro = detector.combined(gt, predictions, THRESHOLDS, IDS)

    assert micro["recall"] == pytest.approx(0.5)  # el auto (0,3) no llega a 0,4; la patente (0,25) sí a 0,2
    assert micro["precision"] == pytest.approx(1.0)
    assert micro["ap50"] == pytest.approx(1.0)  # sin umbral, las dos cajas están bien rankeadas


def test_cpu_name_devuelve_un_texto() -> None:
    assert isinstance(detector.cpu_name(), str)


def test_false_positives_lista_solo_los_falsos_positivos_con_su_iou_y_las_cajas_reales() -> None:
    gt = coco({1: [(1, (0, 0, 100, 100))], 2: [(2, (10, 10, 40, 30))]})
    predictions = {
        1: [
            box(0, 0, 100, 100, 0.9, CAR),  # acierta: no se lista
            box(10, 0, 110, 100, 0.8, CAR),  # repite el auto (el real ya lo usó el de arriba)
            box(500, 500, 600, 600, 0.7, CAR),  # no hay nada anotado ahí
            box(500, 500, 600, 600, 0.3, CAR),  # bajo el umbral del auto (0,4): no cuenta
        ],
        2: [box(60, 60, 90, 80, 0.5, PLATE)],
    }

    found = detector.false_positives(gt, predictions, THRESHOLDS, IDS)

    assert [(f["foto"], f["clase"], f["confianza"]) for f in found] == [
        ("1.jpg", CAR, 0.8),
        ("1.jpg", CAR, 0.7),
        ("2.jpg", PLATE, 0.5),
    ]
    assert found[0]["iou_max"] > 0.8 and found[1]["iou_max"] == 0.0
    assert found[0]["reales"] == [[0.0, 0.0, 100.0, 100.0]] and found[0]["tam"] == [1000, 1000]
    assert len({f["id"] for f in found}) == 3
