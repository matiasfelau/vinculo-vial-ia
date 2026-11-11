"""tools/eval/fp_audit.py: veredictos de los falsos positivos del detector y precisión corregida."""

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

pytest.importorskip("pycocotools")  # solo en la máquina de evaluación: CI no instala los extras de tools

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))
import fp_audit  # noqa: E402


def write_photo(path: Path, width: int = 200, height: int = 120) -> None:
    pixels = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), pixels)


def fp(
    fp_id: str, photo: str, name: str = "car", box: tuple[float, ...] = (10, 10, 60, 50)
) -> dict[str, Any]:
    return {
        "id": fp_id,
        "foto": photo,
        "clase": name,
        "tam": [120, 200],
        "confianza": 0.6,
        "caja": list(box),
        "iou_max": 0.0,
        "reales": [[100, 10, 150, 50]],
    }


def test_inherit_copia_el_veredicto_de_la_misma_caja_en_la_misma_foto_y_clase() -> None:
    old = [fp("o1", "a.jpg"), fp("o2", "b.jpg")]
    new = [
        fp("n1", "a.jpg", box=(12, 12, 62, 52)),  # IoU alto con o1
        fp("n2", "a.jpg", box=(100, 100, 120, 120)),  # otra caja
        fp("n3", "b.jpg", name="plate"),  # otra clase
        fp("n4", "c.jpg"),  # otra foto
    ]

    assert fp_audit.inherit(new, old, {"o1": fp_audit.REAL, "o2": fp_audit.ERROR}) == {"n1": fp_audit.REAL}


def test_resolve_verdicts_valida_categorias_y_numeros() -> None:
    index = {"1": "x", "2": "y", "3": "z"}

    assert fp_audit.resolve_verdicts({"real_sin_anotar": [1, 3], "error": [2]}, index) == {
        "x": "real_sin_anotar",
        "z": "real_sin_anotar",
        "y": "error",
    }
    with pytest.raises(ValueError, match="desconocida"):
        fp_audit.resolve_verdicts({"otra": [1]}, index)
    with pytest.raises(ValueError, match="dos categorías"):
        fp_audit.resolve_verdicts({"error": [1], "dudoso": [1]}, index)


def test_summarize_corrige_la_precision_sin_reemplazar_la_oficial() -> None:
    report = {"falsos_positivos": 4, "precision": 0.6}  # 6 aciertos y 4 falsos positivos
    fps = [fp(f"f{i}", f"{i}.jpg") for i in range(4)]
    verdicts = {"f0": fp_audit.REAL, "f1": fp_audit.REAL, "f2": fp_audit.DOUBT, "f3": fp_audit.ERROR}

    row = fp_audit.summarize(report, fps, verdicts, "car")

    assert row["aciertos"] == 6 and row["precision_oficial"] == 0.6
    assert row["precision_corregida"] == 0.8  # (6 + 2) / 10
    assert row["precision_corregida_con_dudosos"] == 0.9
    assert row["por_categoria"][fp_audit.ERROR] == 1
    with pytest.raises(ValueError, match="faltan"):
        fp_audit.summarize(report, fps, {"f0": fp_audit.REAL}, "car")
    with pytest.raises(ValueError, match="en el informe"):
        fp_audit.summarize({"falsos_positivos": 3, "precision": 0.5}, fps, verdicts, "car")


def test_combine_es_el_micro_promedio_de_las_clases() -> None:
    rows = [
        {"aciertos": 6, "falsos_positivos": 4, "por_categoria": {fp_audit.REAL: 2, fp_audit.DOUBT: 1}},
        {"aciertos": 2, "falsos_positivos": 2, "por_categoria": {fp_audit.REAL: 1, fp_audit.DOUBT: 0}},
    ]

    out = fp_audit.combine(rows)

    assert out["precision_oficial"] == round(8 / 14, 4)
    assert out["precision_corregida"] == round(11 / 14, 4)
    assert out["precision_corregida_con_dudosos"] == round(12 / 14, 4)


def test_sheets_y_corrected_de_punta_a_punta_con_herencia(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    for name in ("a.jpg", "b.jpg"):
        write_photo(images / name)
    first = [fp("p1", "a.jpg"), fp("p2", "b.jpg", name="plate", box=(20, 20, 60, 40))]
    (tmp_path / "first.json").write_text(json.dumps({"dataset": str(images), "falsos_positivos": first}))

    assert (
        fp_audit.main(["sheets", "--dump", str(tmp_path / "first.json"), "--out", str(tmp_path / "sin")]) == 0
    )
    assert (tmp_path / "sin" / "fp-01.jpg").is_file()
    index = json.loads((tmp_path / "sin" / "indice.json").read_text())
    assert index == {"1": "p1", "2": "p2"}
    (tmp_path / "sin" / "veredictos.json").write_text(json.dumps({"real_sin_anotar": [1], "error": [2]}))
    report = {
        "mosaico": None,
        "clases": {
            "car": {"falsos_positivos": 1, "precision": 0.5},
            "plate": {"falsos_positivos": 1, "precision": 0.5},
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report))

    out = tmp_path / "corregida.json"
    assert (
        fp_audit.main(
            [
                "corrected",
                "--report",
                str(tmp_path / "report.json"),
                "--audit",
                str(tmp_path / "sin"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["por_clase"]["car"]["precision_corregida"] == 1.0  # 1 acierto + 1 real de 2
    assert result["por_clase"]["plate"]["precision_corregida"] == 0.5
    assert result["auto_y_patente"]["precision_corregida"] == 0.75

    second = [fp("q1", "a.jpg", box=(11, 11, 61, 51)), fp("q2", "a.jpg", box=(100, 80, 140, 110))]
    (tmp_path / "second.json").write_text(json.dumps({"dataset": str(images), "falsos_positivos": second}))
    args = [
        "sheets",
        "--dump",
        str(tmp_path / "second.json"),
        "--out",
        str(tmp_path / "mos"),
        "--inherit-from",
        str(tmp_path / "sin" / "falsos_positivos.json"),
    ]
    assert fp_audit.main(args) == 0
    assert json.loads((tmp_path / "mos" / "heredados.json").read_text()) == {"q1": "real_sin_anotar"}
    assert json.loads((tmp_path / "mos" / "indice.json").read_text()) == {
        "1": "q2"
    }  # solo lo nuevo se revisa


def test_la_salida_de_fp_audit_no_puede_estar_en_el_repo(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text(json.dumps({"dataset": str(tmp_path), "falsos_positivos": []}))

    assert (
        fp_audit.main(["sheets", "--dump", str(tmp_path / "d.json"), "--out", str(REPO_ROOT / "tmp-hojas")])
        == 1
    )
