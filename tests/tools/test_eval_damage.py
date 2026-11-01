"""tools/eval/damage.py: métrica por foto y línea de comandos, con un CarDD y un detector sintéticos."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("pycocotools")

from vvi_ia.domain.model import Box  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL = REPO_ROOT / "tools" / "eval"


def load() -> ModuleType:
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    spec = importlib.util.spec_from_file_location("eval_damage", EVAL / "damage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_damage"] = module
    spec.loader.exec_module(module)
    return module


damage = load()
CATEGORIES = [
    {"id": i + 1, "name": n}
    for i, n in enumerate(["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"])
]


def make_split(folder: Path, photos: dict[str, list[tuple[int, list[float]]]]) -> None:
    folder.mkdir(parents=True)
    images, annotations = [], []
    for image_id, (name, boxes) in enumerate(photos.items(), start=1):
        cv2.imwrite(str(folder / name), np.full((60, 100, 3), 100, np.uint8))
        images.append({"id": image_id, "file_name": name, "width": 100, "height": 60})
        for category, bbox in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
    (folder / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}), encoding="utf-8"
    )


def coco_of(folder: Path) -> Any:
    from pycocotools.coco import COCO

    return COCO(str(folder / "_annotations.coco.json"))


def box(conf: float, label: str = "dent", bbox: tuple[float, float, float, float] = (10, 10, 30, 30)) -> Box:
    return Box(*bbox, conf=conf, label=label)


def test_las_clases_tienen_los_ids_de_cardd() -> None:
    assert damage.CLASS_IDS == {
        "dent": 1,
        "scratch": 2,
        "crack": 3,
        "glass_shatter": 4,
        "lamp_broken": 5,
        "tire_flat": 6,
    }


def test_por_foto_cuenta_las_fotos_con_alguna_caja_confiable(tmp_path: Path) -> None:
    make_split(
        tmp_path / "t",
        {"a.jpg": [(1, [10, 10, 20, 20])], "b.jpg": [(2, [5, 5, 10, 10])], "c.jpg": [(1, [0, 0, 8, 8])]},
    )
    gt = coco_of(tmp_path / "t")
    predictions = {1: [box(0.9)], 2: [box(0.2)], 3: []}  # b solo tiene una caja de confianza baja; c, ninguna

    result = damage.photo_level(gt, predictions, 0.4)

    assert result == {
        "fotos_con_dano": 3,
        "con_alguna_caja": 1,
        "recall_por_foto": pytest.approx(1 / 3, abs=1e-4),
    }


def test_por_foto_con_el_umbral_en_el_limite(tmp_path: Path) -> None:
    make_split(tmp_path / "t", {"a.jpg": [(1, [10, 10, 20, 20])]})
    gt = coco_of(tmp_path / "t")

    assert damage.photo_level(gt, {1: [box(0.4)]}, 0.4)["con_alguna_caja"] == 1
    assert damage.photo_level(gt, {1: [box(0.39)]}, 0.4)["con_alguna_caja"] == 0


def test_una_foto_sin_danos_anotados_no_cuenta(tmp_path: Path) -> None:
    make_split(tmp_path / "t", {"a.jpg": [(1, [10, 10, 20, 20])], "vacia.jpg": []})

    result = damage.photo_level(coco_of(tmp_path / "t"), {1: [box(0.9)], 2: [box(0.9)]}, 0.4)

    assert result["fotos_con_dano"] == 1 and result["recall_por_foto"] == 1.0


def fake_models(boxes: list[Box]) -> Any:
    return SimpleNamespace(
        damage=SimpleNamespace(detect=lambda img: list(boxes)),
        versions=lambda: {"damage": "rfdetr_cardd@abc"},
    )


def test_main_de_punta_a_punta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_split(
        tmp_path / "data" / "cardd-detector" / "test",
        {"a.jpg": [(1, [10, 10, 20, 20])], "b.jpg": [(5, [40, 20, 20, 20])]},
    )
    seen: dict[str, Any] = {}

    def build(selection: Any, manifest: Any, models_dir: Any, params: Any = None) -> Any:
        seen["params"] = params
        return fake_models([box(0.9, "dent", (10, 10, 30, 30))])

    monkeypatch.setattr(damage, "build_models", build)

    code = damage.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            "--models-dir",
            str(tmp_path),
            "--damage-min-conf",
            "0.7",
        ]
    )

    assert code == 0
    assert seen["params"] == {"damage_min_conf": damage.FLOOR}  # el AP se calcula con el umbral casi en cero
    report = json.loads((tmp_path / "out" / "damage_report.json").read_text(encoding="utf-8"))
    assert report["fotos"] == 2 and report["por_foto"]["recall_por_foto"] == 1.0
    assert report["clases"]["dent"]["recall"] == 1.0 and report["clases"]["dent"]["precision"] == 0.5
    assert report["clases"]["lamp_broken"]["recall"] == 0.0
    shown = capsys.readouterr().out
    assert "Por foto" in shown and "a.jpg" not in shown


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = damage.main(
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


def test_main_sin_el_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = damage.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--models-dir", str(tmp_path)]
    )

    assert code == 1 and "Corré antes prepare.py" in capsys.readouterr().err


def test_main_con_un_modelo_que_no_esta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_split(tmp_path / "cardd-detector" / "test", {"a.jpg": [(1, [10, 10, 20, 20])]})

    code = damage.main(
        [
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "o"),
            "--models-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "nada.json"),
            "--model",
            "fantasma",
        ]
    )

    assert code == 1 and "No se pudo empezar" in capsys.readouterr().err


def test_main_exige_las_rutas(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TRAIN_DATA_DIR", "EVAL_OUT", "MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        damage.parse_args([])
