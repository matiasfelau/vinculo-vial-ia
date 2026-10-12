"""tools/eval/presence.py: reglas de presencia, tasas por conjunto, recortes de patentes y fotos negativas."""

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
        "eval_presence", REPO_ROOT / "tools" / "eval" / "presence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_presence"] = module
    spec.loader.exec_module(module)
    return module


presence = load()
T = presence.DEFAULT_THRESHOLDS


def test_passes_si_alguna_senal_de_la_regla_llega_al_umbral() -> None:
    assert presence.passes({"wheel": [[0.5, 0.1]]}, presence.RULES["R1"], T)
    assert not presence.passes({"wheel": [[0.3, 0.1]], "seg": [[0.9, 0.5]]}, presence.RULES["R1"], T)
    assert presence.passes({"seg": [[0.9, 0.5]]}, presence.RULES["R2"], T)
    assert presence.passes({"damage": [[0.8, 0.1]]}, presence.RULES["R3"], T)
    assert not presence.passes({"damage": [[0.8, 0.1]]}, presence.RULES["R2"], T)
    assert not presence.passes({}, presence.RULES["R3"], T)


def test_passes_exige_el_area_minima_en_el_mismo_hallazgo() -> None:
    signals = {"car": [[0.9, 0.01], [0.2, 0.5]]}  # uno confiable pero chico y otro grande pero poco confiable

    assert presence.passes(signals, ("car",), T, min_area=0.0)
    assert not presence.passes(signals, ("car",), T, min_area=0.05)


def test_rates_cuenta_las_fotos_que_pasan_por_regla() -> None:
    scores = {
        "plate": [{"plate": [[0.9, 0.2]]}, {"seg": [[0.9, 0.2]]}, {}],
        "negativas": [{"car": [[0.1, 0.2]]}],
    }

    table = presence.rates(scores)

    assert table["plate"] == {"fotos": 3, "solo_car": 0, "R1": 1, "R2": 2, "R3": 2}
    assert table["negativas"]["R3"] == 0
    assert presence.rates(scores, min_area=0.5)["plate"]["R2"] == 0


def test_verdict_marca_que_reglas_cumplen_a_y_c() -> None:
    table = {
        "general": {"fotos": 10, "solo_car": 10, "R1": 10, "R2": 10, "R3": 10},
        "plate": {"fotos": 5, "solo_car": 0, "R1": 5, "R2": 5, "R3": 5},
        "detail": {"fotos": 8, "solo_car": 0, "R1": 6, "R2": 8, "R3": 8},
        "negativas": {"fotos": 20, "solo_car": 0, "R1": 0, "R2": 1, "R3": 3},
    }

    result = presence.verdict(table)

    assert result["solo_car"]["cumple_A"] and result["solo_car"]["cumple_C"]
    assert result["R1"]["cumple_A"] and not result["R1"]["cumple_C"]
    assert result["R1"]["validas_rechazadas_por_rol"] == {"plate": 0, "detail": 2}
    assert not result["R2"]["cumple_A"] and result["R2"]["cumple_C"]
    assert result["R3"]["negativas_que_pasan"] == 3


def test_la_cota_es_3_sobre_n() -> None:
    assert presence.bound(60) == 0.05
    assert presence.bound(1) == 1.0
    assert presence.bound(0) is None


def make_coco(folder: Path, images: dict[str, list[tuple[str, tuple[float, float, float, float]]]]) -> None:
    folder.mkdir(parents=True)
    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "wheel"},
        {"id": 3, "name": "plate"},
        {"id": 4, "name": "face"},
    ]
    ids = {c["name"]: c["id"] for c in categories}
    coco_images, annotations = [], []
    for index, (name, boxes) in enumerate(images.items(), start=1):
        cv2.imwrite(str(folder / name), np.full((100, 200, 3), 90, np.uint8))
        coco_images.append({"id": index, "file_name": name, "width": 200, "height": 100})
        for label, bbox in boxes:
            annotations.append(
                {"id": len(annotations) + 1, "image_id": index, "category_id": ids[label], "bbox": list(bbox)}
            )
    (folder / presence.ANNOTATIONS).write_text(
        json.dumps({"images": coco_images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )


def test_negative_photos_deja_las_que_no_tienen_car_plate_ni_wheel(tmp_path: Path) -> None:
    make_coco(
        tmp_path / presence.OPEN_IMAGES / "test",
        {
            "auto.jpg": [("car", (0, 0, 50, 50))],
            "rueda.jpg": [("wheel", (0, 0, 20, 20))],
            "patente.jpg": [("plate", (0, 0, 20, 10))],
            "cara.jpg": [("face", (0, 0, 20, 20))],
            "nada.jpg": [],
        },
    )

    names = sorted(p.name for p in presence.negative_photos(tmp_path))

    assert names == ["cara.jpg", "nada.jpg"]


def test_plate_crops_agrega_margen_y_descarta_las_cajas_chicas(tmp_path: Path) -> None:
    make_coco(
        tmp_path / "train",
        {"a.jpg": [("plate", (50, 40, 40, 20)), ("plate", (0, 0, 8, 8))]},
    )

    crops = presence.plate_crops(tmp_path)

    assert len(crops) == 1
    assert crops[0].shape[:2] == (
        40,
        80,
    )  # 20 px de alto con 50 % de margen a cada lado → 40; 40 de ancho → 80


def test_plate_crops_respeta_el_limite(tmp_path: Path) -> None:
    make_coco(tmp_path / "train", {"a.jpg": [("plate", (10, 10, 40, 20)), ("plate", (100, 10, 40, 20))]})

    assert len(presence.plate_crops(tmp_path, limit=1)) == 1


def test_la_salida_dentro_del_repo_se_rechaza(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = presence.main(
        [
            *("--data-dir", str(tmp_path), "--eval-dir", str(tmp_path)),
            *("--out-dir", str(REPO_ROOT / "salida"), "--models-dir", str(tmp_path)),
        ]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err
