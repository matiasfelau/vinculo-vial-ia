"""Herramientas del detector: conversión de Open Images a COCO, armado de splits y métricas de evaluación.

La conversión no necesita FiftyOne ni PyTorch y corre siempre; el resto se omite si no está el grupo `tools`.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "tools" / "training" / "detector"


def load(name: str, path: Path) -> ModuleType:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset = load("dataset", TRAINING / "dataset.py")
Annotation, ImageRecord = dataset.Annotation, dataset.ImageRecord


def box(
    label: str = "Car", x: float = 0.1, y: float = 0.2, w: float = 0.5, h: float = 0.4, **flags: bool
) -> Annotation:
    return Annotation(label, x, y, w, h, **flags)


# --- conversión ------------------------------------------------------------------------------


def test_las_clases_siguen_el_orden_del_adaptador() -> None:
    assert list(dataset.CLASSES) == ["car", "wheel", "plate", "face"]
    assert dataset.OPEN_IMAGES_CLASSES["Vehicle registration plate"] == "plate"


@pytest.mark.parametrize(
    "annotation",
    [
        box("Person"),  # otra clase
        box(group_of=True),  # una caja que cubre un grupo
        box(depiction=True),  # un dibujo
        box(inside=True),  # foto tomada desde adentro
        box(w=0.01),  # 6,4 px de ancho en una foto de 640
        box(h=0.01),
    ],
)
def test_cajas_que_no_se_usan(annotation: Annotation) -> None:
    assert not dataset.is_usable(annotation, 640, 480)


def test_una_caja_normal_se_usa() -> None:
    assert dataset.is_usable(box(), 640, 480)
    assert dataset.is_usable(box("Human face", w=0.02, h=0.03), 640, 480)  # 12,8 × 14,4 px


def test_el_tamano_minimo_es_en_la_foto_original() -> None:
    small = box(w=0.01, h=0.01)

    assert not dataset.is_usable(small, 640, 480)
    assert dataset.is_usable(small, 4000, 3000)  # 40 × 30 px


def test_a_coco_pasa_cajas_normalizadas_a_pixeles() -> None:
    coco = dataset.to_coco(
        [ImageRecord("a.jpg", 200, 100, [box("Vehicle registration plate", 0.25, 0.5, 0.5, 0.25)])]
    )

    (annotation,) = coco["annotations"]
    assert annotation["bbox"] == [50.0, 50.0, 100.0, 25.0]
    assert annotation["area"] == 2500.0
    assert annotation["category_id"] == 3 and annotation["iscrowd"] == 0
    assert coco["images"] == [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}]


def test_las_categorias_salen_en_el_orden_de_las_columnas_del_modelo() -> None:
    coco = dataset.to_coco([])

    assert [(c["id"], c["name"]) for c in coco["categories"]] == [
        (1, "car"),
        (2, "wheel"),
        (3, "plate"),
        (4, "face"),
    ]


def test_las_cajas_se_recortan_a_la_foto() -> None:
    coco = dataset.to_coco([ImageRecord("a.jpg", 100, 100, [box("Car", 0.8, 0.8, 0.5, 0.5)])])

    assert coco["annotations"][0]["bbox"] == [80.0, 80.0, 20.0, 20.0]


def test_las_fotos_sin_cajas_utiles_se_descartan_y_los_ids_son_contiguos() -> None:
    records = [
        ImageRecord("vacia.jpg", 640, 480, [box("Person")]),
        ImageRecord(
            "a.jpg", 640, 480, [box("Car"), box("Wheel", 0.5, 0.5, 0.1, 0.1), box("Car", group_of=True)]
        ),
        ImageRecord("b.jpg", 640, 480, [box("Human face", 0.4, 0.4, 0.1, 0.15)]),
    ]

    coco = dataset.to_coco(records)

    assert [image["file_name"] for image in coco["images"]] == ["a.jpg", "b.jpg"]
    assert [image["id"] for image in coco["images"]] == [1, 2]
    assert [a["id"] for a in coco["annotations"]] == [1, 2, 3]
    assert [a["image_id"] for a in coco["annotations"]] == [1, 1, 2]


def test_conteo_por_clase() -> None:
    coco = dataset.to_coco(
        [ImageRecord("a.jpg", 640, 480, [box("Car"), box("Car", 0.6, 0.6), box("Wheel", 0.5, 0.5, 0.1, 0.1)])]
    )

    assert dataset.class_counts(coco) == {"car": 2, "wheel": 1, "plate": 0, "face": 0}


# --- armado de splits (necesita fiftyone) --------------------------------------------------------


def test_write_split_copia_solo_las_fotos_con_cajas_utiles(tmp_path: Path) -> None:
    pytest.importorskip("fiftyone")
    download = load("download", TRAINING / "download.py")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(source / name), np.zeros((100, 200, 3), np.uint8))
    items = [
        (ImageRecord("a.jpg", 200, 100, [box("Car")]), source / "a.jpg"),
        (ImageRecord("b.jpg", 200, 100, [box("Person")]), source / "b.jpg"),
    ]

    stats = download.write_split(tmp_path / "train", items)

    assert stats == {"fotos": 1, "car": 1, "wheel": 0, "plate": 0, "face": 0}
    assert sorted(p.name for p in (tmp_path / "train").iterdir()) == ["_annotations.coco.json", "a.jpg"]
    assert json.loads((tmp_path / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))["images"]


def test_download_se_niega_a_escribir_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("fiftyone")
    download = load("download", TRAINING / "download.py")

    assert download.main(["--data-dir", str(REPO_ROOT / "datos")]) == 1
    assert "repositorio" in capsys.readouterr().err


def test_download_exige_la_carpeta_de_datos(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fiftyone")
    monkeypatch.delenv("TRAIN_DATA_DIR", raising=False)
    download = load("download", TRAINING / "download.py")

    with pytest.raises(SystemExit):
        download.parse_args([])


# --- evaluación (necesita pycocotools) -----------------------------------------------------------


@pytest.fixture(scope="module")
def evaluation() -> ModuleType:
    pytest.importorskip("pycocotools")
    return load("eval_detector", REPO_ROOT / "tools" / "eval" / "detector.py")


def test_iou(evaluation: ModuleType) -> None:
    assert evaluation.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert evaluation.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert evaluation.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_una_caja_real_se_usa_una_sola_vez(evaluation: ModuleType) -> None:
    truths = [(0, 0, 10, 10)]
    guesses = [(0.9, (0, 0, 10, 10)), (0.8, (1, 1, 10, 10))]  # las dos caen sobre la misma caja

    hits, found = evaluation.match(truths, guesses)

    assert hits == [True, False] and found == {0}


def test_se_empareja_de_mayor_a_menor_confianza(evaluation: ModuleType) -> None:
    truths = [(0, 0, 10, 10)]

    hits, _ = evaluation.match(truths, [(0.2, (0, 0, 10, 10)), (0.9, (0, 0, 10, 10))])

    assert hits == [True, False]  # la de 0,9 va primero y gana


def test_una_prediccion_que_no_llega_a_iou_05_no_acierta(evaluation: ModuleType) -> None:
    hits, found = evaluation.match([(0, 0, 10, 10)], [(0.9, (6, 0, 16, 10))])

    assert hits == [False] and found == set()


def coco_with(tmp_path: Path, boxes: list[tuple[int, list[float]]]) -> object:
    from pycocotools.coco import COCO

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": cat,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            for i, (cat, bbox) in enumerate(boxes)
        ],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(dataset.CLASSES)],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    return COCO(str(path))


def test_evaluate_calcula_recall_precision_y_ap50(evaluation: ModuleType, tmp_path: Path) -> None:
    from vvi_ia.domain.model import Box

    gt = coco_with(tmp_path, [(1, [10, 10, 100, 50]), (4, [150, 10, 20, 20])])  # un auto y un rostro chico
    predictions = {
        1: [
            Box(10, 10, 110, 60, conf=0.9, label="car"),  # acierta
            Box(0, 0, 5, 5, conf=0.8, label="car"),  # falso positivo
            Box(150, 10, 170, 30, conf=0.35, label="face"),  # acierta, sobre el umbral de rostro (0,30)
        ]
    }

    report = evaluation.evaluate(gt, predictions)

    car = report["car"]
    assert (car["cajas_reales"], car["recall"], car["precision"], car["falsos_positivos"]) == (1, 1.0, 0.5, 1)
    assert car["ap50"] == pytest.approx(1.0)
    face = report["face"]
    assert face["recall"] == 1.0 and face["cajas_chicas"] == 1 and face["recall_chicas"] == 1.0
    assert (
        report["wheel"]["cajas_reales"] == 0
        and report["wheel"]["recall"] is None
        and report["wheel"]["ap50"] is None
    )


def test_evaluate_aplica_el_umbral_de_cada_clase(evaluation: ModuleType, tmp_path: Path) -> None:
    from vvi_ia.domain.model import Box

    gt = coco_with(tmp_path, [(4, [150, 10, 20, 20])])
    predictions = {1: [Box(150, 10, 170, 30, conf=0.25, label="face")]}  # bajo el umbral de rostro (0,30)

    report = evaluation.evaluate(gt, predictions)

    assert report["face"]["recall"] == 0.0 and report["face"]["precision"] is None
    assert report["face"]["ap50"] == pytest.approx(1.0)  # el AP no depende del umbral


def test_evaluate_sin_detecciones(evaluation: ModuleType, tmp_path: Path) -> None:
    gt = coco_with(tmp_path, [(1, [10, 10, 100, 50])])

    report = evaluation.evaluate(gt, {})

    assert report["car"]["recall"] == 0.0 and report["car"]["ap50"] == 0.0


def test_la_evaluacion_se_niega_a_trabajar_dentro_del_repo(
    evaluation: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    code = evaluation.main(
        [
            "--dataset-dir",
            str(REPO_ROOT / "x"),
            "--out-dir",
            str(REPO_ROOT / "y"),
            "--models-dir",
            str(REPO_ROOT),
        ]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


# --- módulo común de entrenamiento (necesita torch) ---------------------------------------------


@pytest.fixture(scope="module")
def common() -> ModuleType:
    pytest.importorskip("torch")
    return load("common", REPO_ROOT / "tools" / "training" / "common.py")


def rows(*values: list[float]) -> np.ndarray:
    return np.array([list(values)], np.float32)


def test_las_filas_del_onnx_se_buscan_sin_importar_el_orden(common: ModuleType) -> None:
    boxes = np.array([[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3], [0.9, 0.9, 0.1, 0.1]]], np.float32)
    logits = np.array([[[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0]]], np.float32)
    shuffled = [boxes[:, [2, 0, 1]], logits[:, [2, 0, 1]]]

    assert common.matching_rows(shuffled, [boxes, logits]) == 1.0


def test_cuenta_las_filas_que_no_coinciden(common: ModuleType) -> None:
    boxes = np.zeros((1, 4, 4), np.float32)
    boxes[0, :, 0] = [0.1, 0.3, 0.5, 0.7]
    other = boxes.copy()
    other[0, 3, 0] = 0.95  # una de cuatro cambia

    assert common.matching_rows([other], [boxes]) == 0.75


def test_la_tolerancia_de_las_filas(common: ModuleType) -> None:
    boxes = np.zeros((1, 2, 4), np.float32)
    boxes[0, 1, 0] = 0.5
    close = boxes + 0.005
    far = boxes + 0.05

    assert common.matching_rows([close], [boxes]) == 1.0
    assert common.matching_rows([far], [boxes]) == 0.0


def test_las_mascaras_tambien_se_comparan(common: ModuleType) -> None:
    masks = np.random.default_rng(0).normal(size=(1, 5, 6, 6)).astype(np.float32)

    assert common.matching_rows([masks[:, ::-1].copy()], [masks]) == 1.0


def test_encuentra_el_mejor_checkpoint(common: ModuleType, tmp_path: Path) -> None:
    assert common.find_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint_best_ema.pth").write_bytes(b"x")
    assert common.find_checkpoint(tmp_path).name == "checkpoint_best_ema.pth"  # type: ignore[union-attr]
    (tmp_path / "checkpoint_best_total.pth").write_bytes(b"x")
    assert common.find_checkpoint(tmp_path).name == "checkpoint_best_total.pth"  # type: ignore[union-attr]


def test_lee_las_metricas_de_validacion_y_de_prueba(common: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "metrics.csv").write_text(
        "epoch,step,train/loss,val/mAP_50,val/mAP_50_95,val/recall,val/precision,test/mAP_50,test/mAP_50_95,test/recall,test/precision\n"
        "0,7,1.5,,,,,,,,\n"
        "0,7,,0.50,0.30,0.60,0.40,,,,\n"
        "1,15,,0.70,0.45,0.75,0.55,,,,\n"
        "2,23,,0.65,0.40,0.70,0.50,,,,\n"
        "2,23,,,,,,0.68,0.41,0.72,0.52\n",
        encoding="utf-8",
    )

    summary = common.read_metrics(tmp_path)

    assert summary["epocas_hechas"] == 3 and summary["mejor_epoca"] == 1
    assert summary["validacion_mejor"] == {"mAP50": 0.7, "mAP50_95": 0.45, "recall": 0.75, "precision": 0.55}
    assert summary["prueba"] == {"mAP50": 0.68, "mAP50_95": 0.41, "recall": 0.72, "precision": 0.52}


def test_sin_metricas_no_hay_resumen(common: ModuleType, tmp_path: Path) -> None:
    assert common.read_metrics(tmp_path) == {}


def test_rutas_dentro_del_repo(common: ModuleType, tmp_path: Path) -> None:
    assert common.inside_repo(REPO_ROOT / "datos")
    assert not common.inside_repo(tmp_path)


def test_en_cpu_no_hay_nada_que_revisar(common: ModuleType) -> None:
    assert common.check_device("cpu") == ({}, None)


def test_pedir_cuda_sin_gpu_explica_que_usar_cpu(common: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _, problem = common.check_device("cuda")

    assert problem is not None and "--device cpu" in problem


def test_con_poca_memoria_libre_pide_cerrar_programas(
    common: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1000 * common.MB, 6144 * common.MB))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _=0: "GPU de prueba")

    report, problem = common.check_device("cuda")

    assert report["memoria_libre_mb"] == 1000
    assert problem is not None and "Cerrá los programas" in problem and "1000" in problem


def test_con_memoria_suficiente_se_puede_empezar(common: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (5000 * common.MB, 6144 * common.MB))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _=0: "GPU de prueba")

    assert common.check_device("cuda")[1] is None


def test_las_opciones_de_entrenamiento_por_defecto(common: ModuleType) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    common.add_training_options(parser)

    args = parser.parse_args([])

    assert (args.device, args.epochs, args.batch_size, args.grad_accum) == ("cuda", 20, 2, 8)
    assert args.amp is False  # FP32 por defecto
    assert parser.parse_args(["--amp", "--device", "cpu"]).amp is True
