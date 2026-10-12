"""tools/eval/damage_fp.py: elección de fotos, barrido de umbrales, umbrales por clase y hoja de contacto."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

pytest.importorskip("pycocotools")

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL = REPO_ROOT / "tools" / "eval"


def load() -> ModuleType:
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    spec = importlib.util.spec_from_file_location("eval_damage_fp", EVAL / "damage_fp.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_damage_fp"] = module
    spec.loader.exec_module(module)
    return module


fp = load()


def coco(*cars: tuple[str, int, int, list[float]]) -> dict[str, object]:
    """COCO con una foto de 100x100 por nombre y las cajas dadas como `(foto, id, categoría, bbox)`."""
    names = sorted({name for name, *_ in cars})
    return {
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "wheel"}],
        "images": [{"id": i + 1, "file_name": n, "width": 100, "height": 100} for i, n in enumerate(names)],
        "annotations": [
            {"id": i, "image_id": names.index(name) + 1, "category_id": cat, "bbox": bbox}
            for name, i, cat, bbox in cars
        ],
    }


def test_closeups_elige_las_fotos_con_un_auto_grande() -> None:
    data = coco(
        ("grande.jpg", 1, 1, [0, 0, 60, 60]),
        ("chico.jpg", 2, 1, [0, 0, 20, 20]),
        ("rueda.jpg", 3, 2, [0, 0, 90, 90]),
        ("dos.jpg", 4, 1, [0, 0, 10, 10]),
        ("dos.jpg", 5, 1, [0, 0, 50, 60]),
    )

    assert fp.closeups(data, 0.30) == ["dos.jpg", "grande.jpg"]


def test_photo_rate_cuenta_las_fotos_que_llegan_al_umbral() -> None:
    assert fp.photo_rate([0.1, 0.5, 0.9], 0.5) == pytest.approx(2 / 3)
    assert fp.photo_rate([], 0.5) == 0.0


def test_sweep_cruza_falsos_positivos_y_recall() -> None:
    rows = fp.sweep([0.0, 0.3, 0.6, 0.9], [0.5, 0.8, 0.9, 0.95], thresholds=(0.4, 0.7))

    assert rows[0] == {
        "umbral": 0.4,
        "fotos_sin_dano_con_caja": 2,
        "falsos_positivos_por_foto": 0.5,
        "recall_por_foto": 1.0,
    }
    assert rows[1]["falsos_positivos_por_foto"] == 0.25 and rows[1]["recall_por_foto"] == 0.75


def test_propose_maximiza_recall_menos_falsos_positivos_y_desempata_por_el_umbral_bajo() -> None:
    rows = fp.sweep([0.2, 0.2, 0.9, 0.9], [0.3, 0.8, 0.95, 0.95], thresholds=(0.1, 0.5, 0.9))

    assert fp.propose(rows)["DAMAGE_MIN_CONF"] == 0.5
    flat = fp.sweep([0.0], [1.0], thresholds=(0.2, 0.6))
    assert fp.propose(flat)["DAMAGE_MIN_CONF"] == 0.2


def test_by_class_mide_cada_clase_con_sus_propias_fotos() -> None:
    clean = [{"dent": 0.9}, {"dent": 0.2, "scratch": 0.6}, {}]
    damaged = [{"dent": 0.8}, {"scratch": 0.7}, {"dent": 0.1}]
    labels = [{"dent"}, {"scratch"}, {"dent"}]

    result = fp.by_class(clean, damaged, labels, thresholds=(0.5,))

    assert result["dent"] == [{"umbral": 0.5, "falsos_positivos_por_foto": 0.3333, "recall_por_foto": 0.5}]
    assert result["scratch"][0]["recall_por_foto"] == 1.0
    assert result["crack"][0]["recall_por_foto"] == 0.0  # ninguna foto con esa clase


def test_class_thresholds_y_combined() -> None:
    per_class = {
        "dent": [
            {"umbral": 0.3, "falsos_positivos_por_foto": 0.5, "recall_por_foto": 1.0},
            {"umbral": 0.6, "falsos_positivos_por_foto": 0.0, "recall_por_foto": 0.5},
        ],
        "crack": [{"umbral": 0.3, "falsos_positivos_por_foto": 0.5, "recall_por_foto": 1.0}],
    }

    chosen = fp.class_thresholds(per_class, 0.05)

    assert chosen == {"dent": 0.6, "crack": 1.0}
    result = fp.combined([{"dent": 0.7}, {"crack": 0.9}], [{"dent": 0.65}, {"dent": 0.2}], chosen)
    assert result == {"falsos_positivos_por_foto": 0.5, "recall_por_foto": 0.5}


def test_read_excluded_ignora_lineas_vacias(tmp_path: Path) -> None:
    assert fp.read_excluded(tmp_path) == set()
    (tmp_path / fp.EXCLUDED).write_text("a.jpg\n\n b.jpg \n", encoding="utf-8")

    assert fp.read_excluded(tmp_path) == {"a.jpg", "b.jpg"}


def test_contact_sheet_arma_una_grilla_aunque_una_foto_no_se_decodifique(tmp_path: Path) -> None:
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(tmp_path / name), np.full((50, 80, 3), 128, np.uint8))
        paths.append(tmp_path / name)
    (tmp_path / "rota.jpg").write_bytes(b"no es una imagen")
    paths.append(tmp_path / "rota.jpg")

    sheet = fp.contact_sheet(paths, ["a", "b", "c", "d"], columns=2)

    cell_w, cell_h = fp.CELL[0], fp.CELL[1] + 18
    assert sheet.shape == (2 * cell_h, 2 * cell_w, 3)


def test_la_salida_dentro_del_repo_se_rechaza(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = fp.main(["--data-dir", str(tmp_path), "--out-dir", str(REPO_ROOT / "salida"), "select"])

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_select_copia_las_fotos_con_un_auto_grande(tmp_path: Path) -> None:
    source = tmp_path / fp.OPEN_IMAGES_NAME / "train"
    source.mkdir(parents=True)
    cv2.imwrite(str(source / "grande.jpg"), np.zeros((10, 10, 3), np.uint8))
    (source / fp.ANNOTATIONS).write_text(
        json.dumps(coco(("grande.jpg", 1, 1, [0, 0, 60, 60]))), encoding="utf-8"
    )

    assert fp.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "select"]) == 0
    assert (tmp_path / fp.FOLDER / "images" / "train_grande.jpg").is_file()
