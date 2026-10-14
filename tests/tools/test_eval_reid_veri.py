"""tools/eval/reid_veri.py: recortes, pares y calibración de REID_T, con embeddings sintéticos."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL = REPO_ROOT / "tools" / "eval"


def load() -> ModuleType:
    sys.path.insert(0, str(EVAL))  # `reid_veri` importa `reid` como los demás scripts de tools/eval
    try:
        spec = importlib.util.spec_from_file_location("eval_reid_veri", EVAL / "reid_veri.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["eval_reid_veri"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EVAL))
    return module


veri = load()


def crop(name: str, vehicle: str, camera: str, color: str = "1", kind: str = "1", query: bool = False):
    return veri.Crop(name, vehicle, camera, color, kind, query)


LABELS_XML = (
    '<?xml version="1.0" encoding="gb2312" ?>\n<TrainingImages Version="1.0">\n<Items number="3">\n'
    '<Item imageName="0002_c002_00030600_0.jpg" vehicleID="0002" cameraID="c002" colorID="1" typeID="4" />\n'
    '<Item imageName="0002_c003_00030605_1.jpg" vehicleID="0002" cameraID="c003" colorID="1" typeID="4" />\n'
    '<Item imageName="0007_c001_00030615_1.jpg" vehicleID="0007" cameraID="c001" colorID="10" typeID="2" />\n'
    "</Items></TrainingImages>\n"
)


def test_load_labels_lee_el_xml_en_gb2312(tmp_path: Path) -> None:
    path = tmp_path / "test_label.xml"
    path.write_bytes(LABELS_XML.encode("gb2312"))

    labels = veri.load_labels(path)

    assert labels["0002_c003_00030605_1.jpg"] == ("0002", "c003", "1", "4")
    assert len(labels) == 3


def test_load_labels_sin_archivo_o_sin_items_falla(tmp_path: Path) -> None:
    with pytest.raises(veri.DataError):
        veri.load_labels(tmp_path / "no_existe.xml")
    (tmp_path / "vacio.xml").write_bytes(b"<TrainingImages/>")
    with pytest.raises(veri.DataError):
        veri.load_labels(tmp_path / "vacio.xml")


def test_select_crops_limita_por_auto_y_camara_y_prefiere_la_consulta() -> None:
    names = [f"0001_c001_{i:08d}_0.jpg" for i in range(5)] + ["0001_c002_00000001_0.jpg"]
    labels = {n: (n[:4], n[5:9], "1", "1") for n in names}

    crops = veri.select_crops(labels, names, [names[4]], per_track=2, seed=0)

    by_camera = Counter(c.camera for c in crops)
    assert by_camera == {"c001": 2, "c002": 1}
    assert names[4] in {c.name for c in crops}  # la de la consulta entra primero


def test_pair_similarities_separa_mismo_auto_distintos_y_dificiles() -> None:
    crops = [
        crop("a1", "A", "c1", "1", "1"),
        crop("a2", "A", "c2", "1", "1"),
        crop("a3", "A", "c2", "1", "1"),  # misma cámara que a2: no es un par del mismo auto
        crop("b1", "B", "c1", "1", "1"),  # mismo color y tipo que A: difícil
        crop("c1", "C", "c3", "2", "5"),
    ]
    matrix = np.eye(5, dtype=np.float32)

    same, different, hard, ids = veri.pair_similarities(crops, matrix)

    assert same.size == 2 and set(ids.tolist()) == {"A"}  # a1-a2 y a1-a3
    assert different.size == 7  # A-B (3), A-C (3) y B-C (1)
    assert hard.size == 3  # b1 contra a1, a2 y a3


def test_calibrate_con_el_exacto_util_lo_usa() -> None:
    same = np.array([0.70, 0.80, 0.90], np.float32)
    different = np.array([0.10, 0.20, 0.30, 0.95], np.float32)

    out = veri.calibrate(same, different, different)

    assert out["umbral_exacto"] == 0.7 and out["el_exacto_sirve"]
    assert out["umbral_propuesto"] == 0.7
    assert out["con_el_propuesto"]["falsos_rojos"] == 0.0
    assert out["detecta_con_el_exacto"]["autos_distintos"] == 0.75
    assert out["cota_95_del_exacto"] == 1.0


def test_calibrate_si_el_exacto_no_detecta_nada_propone_un_valor_con_falsos_rojos_acotados() -> None:
    rng = np.random.default_rng(0)
    same = np.concatenate([[0.01], rng.uniform(0.5, 1.0, 4_999)]).astype(np.float32)  # un par muy bajo
    different = rng.uniform(0.0, 1.0, 10_000).astype(np.float32)

    out = veri.calibrate(same, different, different, max_false_red=0.01)

    assert out["umbral_exacto"] == 0.01 and not out["el_exacto_sirve"]
    assert out["umbral_propuesto"] > out["umbral_exacto"]
    assert out["con_el_propuesto"]["falsos_rojos"] <= 0.01
    assert out["con_el_propuesto"]["autos_distintos_detectados"] > 0.4


def test_calibrate_sin_pares_no_es_posible() -> None:
    empty = np.zeros(0, np.float32)

    assert not veri.calibrate(empty, np.array([0.5], np.float32), empty)["posible"]


def test_curve_cuenta_los_pares_debajo_de_cada_umbral() -> None:
    same = np.array([0.2, 0.6, 0.9], np.float32)
    different = np.array([0.1, 0.3, 0.5, 0.7], np.float32)

    rows = veri.curve(same, different, different[:2], [0.25, 0.65])

    assert rows[0]["mismo_auto_debajo"] == 1 and rows[0]["autos_distintos_detectados"] == 0.25
    assert rows[1]["mismo_auto_debajo"] == 2 and rows[1]["dificiles_detectados"] == 1.0


def test_vehicles_with_low_pair_cuenta_autos_no_pares() -> None:
    same = np.array([0.1, 0.2, 0.9, 0.3], np.float32)
    ids = np.array(["A", "A", "B", "C"])

    assert veri.vehicles_with_low_pair(same, ids, 0.5) == 2


def test_la_salida_dentro_del_repo_se_rechaza(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = veri.main(
        ["--veri-dir", str(tmp_path), "--out-dir", str(REPO_ROOT / "salida"), "--models-dir", str(tmp_path)]
    )

    assert code == 1 and "dentro del repositorio" in capsys.readouterr().err
