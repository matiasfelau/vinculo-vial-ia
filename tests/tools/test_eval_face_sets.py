"""tools/eval/face_sets.py: el split de prueba de Open Images con solo la clase face, sin descargas."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_face_sets", REPO_ROOT / "tools" / "eval" / "face_sets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_face_sets"] = module
    spec.loader.exec_module(module)
    return module


face_sets = load()


def jpeg(value: int = 100) -> bytes:
    return bytes(cv2.imencode(".jpg", np.full((40, 60, 3), value, np.uint8))[1].tobytes())


def test_build_open_images_deja_solo_la_clase_face(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.jpg").write_bytes(jpeg())
    (source / "b.jpg").write_bytes(jpeg(50))
    coco = {
        "categories": [{"id": 1, "name": "car"}, {"id": 4, "name": "face"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 60, "height": 40},
            {"id": 2, "file_name": "b.jpg", "width": 60, "height": 40},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 30, 30]},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [5, 5, 10, 10], "area": 100},
        ],
    }
    (source / face_sets.ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")

    photos, faces = face_sets.build_open_images(source, tmp_path / "out")

    assert (photos, faces) == (2, 1)  # las fotos sin caras quedan: sirven para contar falsos positivos
    result = json.loads((tmp_path / "out" / face_sets.ANNOTATIONS).read_text(encoding="utf-8"))
    assert [a["category_id"] for a in result["annotations"]] == [1] and result["images"][1][
        "file_name"
    ] == "images/b.jpg"


def test_main_se_niega_a_escribir_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = face_sets.main(["--data-dir", "x", "--eval-dir", str(REPO_ROOT / "caras"), "openimages"])

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_main_openimages_sin_el_subset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = face_sets.main(
        ["--data-dir", str(tmp_path / "train"), "--eval-dir", str(tmp_path / "eval"), "openimages"]
    )

    assert code == 1 and "Falta" in capsys.readouterr().err
