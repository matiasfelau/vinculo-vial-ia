"""tools/training/damage/prepare.py: de CarDD (COCO) al dataset de RF-DETR, con un CarDD sintético."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from vvi_ia.adapters.models.rfdetr_damage import DAMAGE_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "training" / "damage" / "prepare.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_damage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_damage"] = module
    spec.loader.exec_module(module)
    return module


prepare = load()
CATEGORIES = [
    {"id": i + 1, "name": name, "supercategory": "damage"} for i, name in enumerate(prepare.CLASSES)
]


def coco(images: list[str], boxes: list[tuple[int, int, list[float]]]) -> dict[str, Any]:
    """`boxes` = lista de `(índice de foto, categoría, bbox)`."""
    return {
        "info": {},
        "licenses": [],
        "images": [
            {"id": i + 1, "file_name": name, "width": 100, "height": 60, "license": 0}
            for i, name in enumerate(images)
        ],
        "annotations": [
            {
                "id": n + 1,
                "image_id": image + 1,
                "category_id": category,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [[0, 0, 1, 1, 2, 2]],
                "attributes": {"occluded": False},
            }
            for n, (image, category, bbox) in enumerate(boxes)
        ],
        "categories": CATEGORIES,
    }


def make_cardd(root: Path) -> Path:
    """Un `CarDD_COCO` sintético con los tres splits."""
    source = root / "CarDD_release" / "CarDD_COCO"
    (source / "annotations").mkdir(parents=True)
    for source_name in prepare.SPLITS.values():
        images = [f"{source_name[:2]}_{n}.jpg" for n in range(2)]
        (source / source_name).mkdir()
        for name in images:
            cv2.imwrite(str(source / source_name / name), np.zeros((60, 100, 3), np.uint8))
        data = coco(images, [(0, 1, [10, 10, 20, 20]), (0, 2, [30, 5, 10, 10]), (1, 6, [5, 5, 40, 30])])
        (source / "annotations" / f"instances_{source_name}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
    return source


def test_las_clases_son_las_de_cardd_en_el_orden_del_adaptador() -> None:
    assert [name.replace(" ", "_") for name in prepare.CLASSES] == list(DAMAGE_LABELS.values())


def test_convert_deja_solo_cajas() -> None:
    converted = prepare.convert(coco(["a.jpg"], [(0, 1, [10, 10, 20, 20])]))

    (annotation,) = converted["annotations"]
    assert set(annotation) == {"id", "image_id", "category_id", "bbox", "area", "iscrowd"}
    assert annotation["bbox"] == [10, 10, 20, 20] and annotation["iscrowd"] == 0
    assert converted["images"] == [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 60}]
    assert [c["name"] for c in converted["categories"]] == list(prepare.CLASSES)


def test_convert_descarta_cajas_sin_area() -> None:
    converted = prepare.convert(coco(["a.jpg"], [(0, 1, [10, 10, 0, 20]), (0, 2, [10, 10, 5, 5])]))

    assert [a["category_id"] for a in converted["annotations"]] == [2]


@pytest.mark.parametrize(
    "names", [["dent", "scratch"], ["scratch", "dent", "crack", "glass shatter", "lamp broken", "tire flat"]]
)
def test_convert_falla_si_las_categorias_no_son_las_de_cardd(names: list[str]) -> None:
    data = coco(["a.jpg"], [])
    data["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(names)]

    with pytest.raises(ValueError, match="no son las de CarDD"):
        prepare.convert(data)


def test_prepare_arma_los_tres_splits_con_fotos_y_anotaciones(tmp_path: Path) -> None:
    source = make_cardd(tmp_path)

    summary = prepare.prepare(source, tmp_path / "out")

    assert set(summary) == {"train", "valid", "test"}
    assert summary["train"] == {
        "fotos": 2, "dent": 1, "scratch": 1, "crack": 0, "glass shatter": 0, "lamp broken": 0, "tire flat": 1
    }  # fmt: skip
    for split in ("train", "valid", "test"):
        folder = tmp_path / "out" / split
        data = json.loads((folder / prepare.ANNOTATIONS).read_text(encoding="utf-8"))
        assert len(data["images"]) == 2
        assert all((folder / image["file_name"]).is_file() for image in data["images"])


def test_prepare_no_toca_el_dataset_original(tmp_path: Path) -> None:
    source = make_cardd(tmp_path)
    before = sorted(str(p.relative_to(source)) for p in source.rglob("*"))
    annotations = (source / "annotations" / "instances_train2017.json").read_bytes()

    prepare.prepare(source, tmp_path / "out")

    assert sorted(str(p.relative_to(source)) for p in source.rglob("*")) == before
    assert (source / "annotations" / "instances_train2017.json").read_bytes() == annotations


def test_prepare_se_puede_repetir(tmp_path: Path) -> None:
    source = make_cardd(tmp_path)
    first = prepare.prepare(source, tmp_path / "out")

    assert prepare.prepare(source, tmp_path / "out") == first


def test_si_no_se_puede_enlazar_copia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_file = tmp_path / "a.jpg"
    source_file.write_bytes(b"datos")

    def refuse(*_: Any) -> None:
        raise OSError("otro disco")

    monkeypatch.setattr(prepare.os, "link", refuse)
    prepare.place(source_file, tmp_path / "b.jpg")

    assert (tmp_path / "b.jpg").read_bytes() == b"datos"


def test_main_arma_el_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_cardd(tmp_path)

    assert prepare.main(["--data-dir", str(tmp_path)]) == 0

    assert "Listo" in capsys.readouterr().out
    assert (tmp_path / prepare.DATASET_NAME / "train" / prepare.ANNOTATIONS).is_file()


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare.main(["--data-dir", str(REPO_ROOT / "datos")]) == 1
    assert "repositorio" in capsys.readouterr().err


def test_main_avisa_si_no_estan_las_anotaciones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare.main(["--data-dir", str(tmp_path)]) == 1
    assert "No encuentro las anotaciones" in capsys.readouterr().err


def test_main_explica_un_json_que_no_es_de_cardd(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = make_cardd(tmp_path)
    path = source / "annotations" / "instances_val2017.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["categories"] = data["categories"][:3]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert prepare.main(["--data-dir", str(tmp_path)]) == 1
    assert "No se pudo armar el dataset" in capsys.readouterr().err


def test_main_exige_la_carpeta_de_datos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIN_DATA_DIR", raising=False)

    with pytest.raises(SystemExit):
        prepare.parse_args([])
