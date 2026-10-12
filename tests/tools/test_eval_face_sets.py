"""tools/eval/face_sets.py: filtro de caras de WIDER, muestra al azar y Open Images, sin descargas."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

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


def write_parquet(path: Path, photos: list[dict[str, list]]) -> None:
    rows = [
        {"image": {"bytes": jpeg(10 * i), "path": f"{i}.jpg"}, "faces": faces}
        for i, faces in enumerate(photos)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


PHOTOS = [
    {"bbox": [[1, 1, 20, 20], [30, 5, 4, 4], [2, 2, 30, 30]], "invalid": [False, False, True]},
    {"bbox": [[0, 0, 10, 12]], "invalid": [True]},  # solo caras inválidas: la foto no entra
    {"bbox": [[5, 5, 9, 9], [20, 20, 40, 40]], "invalid": [False, False]},
    {"bbox": [[1, 1, 16, 16]], "invalid": [False]},
]


def test_valid_boxes_descarta_invalidas_y_chicas() -> None:
    boxes = face_sets.valid_boxes(PHOTOS[0], min_side=8)

    assert boxes == [[1.0, 1.0, 20.0, 20.0]]  # la de 4 px es chica y la tercera es inválida
    assert face_sets.valid_boxes(PHOTOS[0], min_side=1) == [[1.0, 1.0, 20.0, 20.0], [30.0, 5.0, 4.0, 4.0]]


def test_build_wider_arma_el_coco_con_las_fotos_que_tienen_caras_validas(tmp_path: Path) -> None:
    write_parquet(tmp_path / "validation.parquet", PHOTOS)

    photos, faces = face_sets.build_wider(tmp_path / "validation.parquet", tmp_path / "out", 100, 0, 8)

    assert (photos, faces) == (3, 4)  # la foto 1 se descarta; cuenta 1 + 2 + 1 caras
    coco = json.loads((tmp_path / "out" / face_sets.ANNOTATIONS).read_text(encoding="utf-8"))
    assert [c["name"] for c in coco["categories"]] == ["face"]
    assert all((tmp_path / "out" / i["file_name"]).is_file() for i in coco["images"])
    assert coco["images"][0]["width"] == 60 and coco["images"][0]["height"] == 40
    assert {a["image_id"] for a in coco["annotations"]} == {1, 2, 3}


def test_la_muestra_es_al_azar_pero_reproducible(tmp_path: Path) -> None:
    write_parquet(tmp_path / "v.parquet", [PHOTOS[0], PHOTOS[2], PHOTOS[3]] * 4)

    first = face_sets.build_wider(tmp_path / "v.parquet", tmp_path / "a", 5, 0, 8)
    again = face_sets.build_wider(tmp_path / "v.parquet", tmp_path / "b", 5, 0, 8)

    assert first == again and first[0] == 5
    names = lambda folder: sorted(p.name for p in (tmp_path / folder / "images").iterdir())  # noqa: E731
    assert names("a") == names("b")


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


def test_main_wider_usa_el_parquet_ya_bajado(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "eval" / face_sets.WIDER_FOLDER).mkdir(parents=True)
    write_parquet(tmp_path / "eval" / face_sets.WIDER_FOLDER / "validation.parquet", PHOTOS)

    code = face_sets.main(["--eval-dir", str(tmp_path / "eval"), "wider", "--sample", "10"])

    assert code == 0 and "3 fotos y 4 caras" in capsys.readouterr().out
