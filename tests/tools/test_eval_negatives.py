"""tools/eval/negatives.py: qué fotos de Open Images valen como negativas nuevas."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_negatives", REPO_ROOT / "tools" / "eval" / "negatives.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_negatives"] = module
    spec.loader.exec_module(module)
    return module


negatives = load()


def test_una_foto_con_auto_patente_o_rueda_no_es_negativa() -> None:
    assert negatives.is_negative(["Human face", "Person"])
    assert negatives.is_negative([])
    for label in ("Car", "Vehicle registration plate", "Wheel"):
        assert not negatives.is_negative(["Person", label])


def test_pick_descarta_vistas_repetidas_y_con_autos_y_respeta_el_limite() -> None:
    candidates = [
        ("a.jpg", ["Person"]),
        ("b.jpg", ["Car"]),  # tiene auto
        ("c.jpg", ["Dog"]),  # la vio el detector
        ("d.jpg", ["Building"]),  # ya elegida en otra clase
        ("e.jpg", ["Tree"]),
        ("f.jpg", []),
    ]

    chosen = negatives.pick(candidates, seen={"c.jpg"}, taken={"d.jpg"}, limit=2)

    assert chosen == ["a.jpg", "e.jpg"]


def test_seen_files_junta_las_tres_particiones(tmp_path: Path) -> None:
    for split, name in (("train", "x.jpg"), ("valid", "y.jpg"), ("test", "z.jpg")):
        folder = tmp_path / negatives.SUBSET / split
        folder.mkdir(parents=True)
        (folder / "_annotations.coco.json").write_text(json.dumps({"images": [{"file_name": name}]}))

    assert negatives.seen_files(tmp_path) == {"x.jpg", "y.jpg", "z.jpg"}
    assert negatives.seen_files(tmp_path / "no-existe") == set()


def test_las_clases_pedidas_no_son_de_vehiculos() -> None:
    assert not set(negatives.DEFAULT_CLASSES) & negatives.EXCLUDED_LABELS
