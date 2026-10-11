"""tools/eval/ocr.py: carga del COCO, lectura, métricas, propuesta de OCR_MIN_CONF y auditoría.

El dataset es sintético: fotos grises con un recuadro de un tono distinto por caja, y un lector falso que
responde según el tono. No hay patentes ni fotos reales.
"""

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

from tests.adapters.test_parseq import tiny_parseq, write_env
from vvi_ia.domain.model import Box
from vvi_ia.domain.plate import PlateCheck

REPO_ROOT = Path(__file__).resolve().parents[2]
TONE_STEP = 40


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_ocr", REPO_ROOT / "tools" / "eval" / "ocr.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_ocr"] = module
    spec.loader.exec_module(module)
    return module


ocr = load_script()


class FakeReader:
    """Lee según el tono del recuadro: `reads[tono]` es `(texto, confianza)` o `None`."""

    def __init__(self, reads: dict[int, tuple[str, float] | None]) -> None:
        self.reads = reads
        self.calls = 0

    def read(self, crop: Any) -> tuple[str, float] | None:
        self.calls += 1
        return self.reads[int(crop.mean() // TONE_STEP)]


def make_dataset(
    root: Path, splits: dict[str, list[tuple[str, list[tuple[str, tuple[int, int, int, int], int]]]]]
) -> Path:
    """`splits[split]` = lista de `(archivo, [(clase, (x, y, w, h), tono)])`."""
    for split, photos in splits.items():
        folder = root / split
        folder.mkdir(parents=True)
        categories: dict[str, int] = {}
        images, annotations = [], []
        for image_id, (file, boxes) in enumerate(photos):
            pixels = np.zeros((120, 240, 3), np.uint8)
            for label, (x, y, w, h), tone in boxes:
                pixels[y : y + h, x : x + w] = tone * TONE_STEP + TONE_STEP // 2
                category = categories.setdefault(label, len(categories))
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                    }
                )
            cv2.imwrite(str(folder / file), pixels)
            images.append({"id": image_id, "file_name": file, "width": 240, "height": 120})
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": cid, "name": name} for name, cid in categories.items()],
        }
        (folder / ocr.ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")
    return root


def reading(label: str, text: str, conf: float, split: str = "train") -> Any:
    return ocr.Reading(ocr.Sample(split, "f.jpg", Box(0, 0, 10, 10), label), text, conf, failed=False)


# --- etiquetas -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [("AB123CD", "LLDDDLL"), ("ABC123", "LLLDDD"), ("123ABC", "DDDLLL"), ("A B", "L L")],
)
def test_forma_de_la_etiqueta(label: str, expected: str) -> None:
    assert ocr.shape(label) == expected


@pytest.mark.parametrize("label", ["AA000AA", "ORT000", "ZZ000AS", "000"])
def test_marcadores_de_posicion(label: str) -> None:
    assert ocr.is_placeholder(label)


@pytest.mark.parametrize("label", ["AB123CD", "ABC100", "A0B1C", "ABCDEF", ""])
def test_no_son_marcadores_de_posicion(label: str) -> None:
    assert not ocr.is_placeholder(label)


def test_formatos_de_patente() -> None:
    assert ocr.label_format("AB123CD") == "mercosur"
    assert ocr.label_format("ABC123") == "anterior"
    assert ocr.label_format("123ABC") == "otro"
    assert ocr.label_format("ABC12") == "otro"


# --- carga del dataset ---------------------------------------------------------------------


def test_carga_normaliza_etiquetas_y_convierte_cajas(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {
            "train": [
                ("a.jpg", [("ab 123 cd", (10, 20, 100, 40), 1)]),
                ("b.jpg", [("XY-999-ZW", (0, 0, 50, 30), 2), ("AA000AA", (60, 0, 50, 30), 3)]),
            ]
        },
    )

    samples, discarded = ocr.load_samples(root, ["train"])

    assert [s.label for s in samples] == ["AB123CD", "XY999ZW"]
    first = samples[0]
    assert (first.split, first.file) == ("train", "a.jpg")
    assert (first.box.x1, first.box.y1, first.box.x2, first.box.y2) == (10, 20, 110, 60)
    assert discarded == Counter({"marcador de posición": 1})


def test_carga_descarta_etiquetas_vacias(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("---", (0, 0, 50, 30), 1)])]})

    samples, discarded = ocr.load_samples(root, ["train"])

    assert samples == []
    assert discarded == Counter({"etiqueta vacía": 1})


def test_carga_solo_los_splits_pedidos(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {
            "train": [("a.jpg", [("AB123CD", (0, 0, 50, 30), 1)])],
            "valid": [("b.jpg", [("EF456GH", (0, 0, 50, 30), 1)])],
        },
    )

    samples, _ = ocr.load_samples(root, ["valid"])

    assert [(s.split, s.label) for s in samples] == [("valid", "EF456GH")]


def test_falta_el_json_de_anotaciones(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        ocr.load_samples(tmp_path, ["train"])


# --- lectura -------------------------------------------------------------------------------


def test_lee_cada_recorte_con_el_lector_y_reusa_la_foto(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1), ("EF456GH", (100, 50, 60, 40), 2)])]},
    )
    samples, _ = ocr.load_samples(root, ["train"])
    reader = FakeReader({1: ("AB123CD", 0.9), 2: ("EF456GH", 0.8)})

    items, discarded = ocr.read_samples(samples, root, reader)

    assert [(r.text, r.conf, r.failed) for r, _ in items] == [
        ("AB123CD", 0.9, False),
        ("EF456GH", 0.8, False),
    ]
    assert all(crop.shape == (40, 60, 3) for _, crop in items)
    assert not discarded


def test_si_el_lector_falla_la_lectura_queda_vacia_y_marcada(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1)])]})
    samples, _ = ocr.load_samples(root, ["train"])

    ((result, _),), _ = ocr.read_samples(samples, root, FakeReader({1: None}))

    assert (result.text, result.conf, result.failed) == ("", 0.0, True)
    assert result.check(0.0) == PlateCheck.UNREADABLE


def test_descarta_cajas_vacias_y_fotos_ilegibles(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {
            "train": [
                ("a.jpg", [("AB123CD", (300, 10, 50, 30), 1)]),  # fuera de la foto
                ("b.jpg", [("EF456GH", (0, 0, 50, 30), 1)]),
            ]
        },
    )
    (root / "train" / "b.jpg").write_bytes(b"no es un jpeg")
    samples, _ = ocr.load_samples(root, ["train"])
    reader = FakeReader({})

    items, discarded = ocr.read_samples(samples, root, reader)

    assert items == []
    assert discarded == Counter({"caja vacía": 1, "foto ilegible": 1})
    assert reader.calls == 0


# --- una lectura ---------------------------------------------------------------------------


def test_exacta_no_incluye_las_correcciones_por_confusion() -> None:
    corrected = reading("AB123CD", "A8123CD", 0.9)  # una B leída como 8

    assert not corrected.exact
    assert corrected.content == PlateCheck.MATCH
    assert reading("AB123CD", "ab-123-cd", 0.9).exact


def test_el_umbral_solo_cambia_el_resultado_y_no_el_contenido() -> None:
    read = reading("AB123CD", "AB123CD", 0.5)

    assert read.check(0.6) == PlateCheck.UNREADABLE
    assert read.check(0.5) == PlateCheck.MATCH
    assert read.content == PlateCheck.MATCH


def test_contenido_fuzzy_mismatch_y_vacio() -> None:
    assert reading("AB123CD", "AB129CD", 0.9).content == PlateCheck.FUZZY
    assert reading("AB123CD", "ZZ999ZZ", 0.9).content == PlateCheck.MISMATCH
    assert reading("AB123CD", "", 0.9).content == PlateCheck.UNREADABLE


# --- métricas ------------------------------------------------------------------------------


def test_la_distribucion_tiene_siempre_las_cuatro_clases() -> None:
    reads = [reading("AB123CD", "AB123CD", 0.9), reading("AB123CD", "AB123CD", 0.4)]

    assert ocr.distribution(reads, 0.6) == {
        "PLATE_MATCH": 1,
        "PLATE_FUZZY": 0,
        "PLATE_MISMATCH": 0,
        "PLATE_UNREADABLE": 1,
    }


def test_resumen_con_cero_lecturas() -> None:
    summary = ocr.summarize([], 0.6)

    assert summary["lecturas"] == 0
    assert summary["coincidencia_exacta_pct"] is None


def test_el_barrido_acepta_cada_vez_menos_lecturas() -> None:
    reads = [reading("AB123CD", "AB123CD", conf) for conf in (0.95, 0.75, 0.55, 0.35)]

    rows = ocr.sweep(reads, [0.0, 0.5, 0.7, 0.9, 0.99])

    assert [row["aceptadas"] for row in rows] == [4, 3, 2, 1, 0]
    assert [row["match"] for row in rows] == [4, 3, 2, 1, 0]


# --- propuesta de OCR_MIN_CONF -------------------------------------------------------------


def test_sin_mismatch_se_propone_el_piso() -> None:
    proposal = ocr.propose_min_conf([reading("AB123CD", "AB123CD", 0.9)])

    assert proposal["OCR_MIN_CONF"] == ocr.PROPOSAL_FLOOR
    assert proposal["mayor_confianza_de_un_mismatch"] is None


def test_el_umbral_queda_por_encima_del_peor_mismatch() -> None:
    reads = [
        reading("AB123CD", "ZZ999ZZ", 0.7636),
        reading("AB123CD", "ZZ999ZZ", 0.30),
        reading("AB123CD", "AB123CD", 0.95),
        reading("AB123CD", "AB129CD", 0.99),  # un FUZZY no sube el umbral
    ]

    proposal = ocr.propose_min_conf(reads)

    assert proposal["OCR_MIN_CONF"] == 0.78
    assert proposal["mismatch_totales"] == 2
    assert proposal["lecturas_aceptadas"] == 2
    assert proposal["mayor_confianza_de_un_fuzzy"] == 0.99


def test_el_umbral_propuesto_tiene_tope_y_piso() -> None:
    assert ocr.propose_min_conf([reading("AB123CD", "ZZ999ZZ", 0.9999)])["OCR_MIN_CONF"] == ocr.PROPOSAL_CEIL
    assert ocr.propose_min_conf([reading("AB123CD", "ZZ999ZZ", 0.10)])["OCR_MIN_CONF"] == ocr.PROPOSAL_FLOOR


def test_las_lecturas_vacias_no_cuentan_como_mismatch() -> None:
    proposal = ocr.propose_min_conf([reading("AB123CD", "", 0.0)])

    assert proposal["mismatch_totales"] == 0
    assert proposal["lecturas_aceptadas"] == 0


def test_el_reporte_arma_la_propuesta_sin_los_formatos_fuera_de_alcance() -> None:
    reads = [
        reading("AB123CD", "AB123CD", 0.9),
        reading("ABC123", "ABC124", 0.7),  # FUZZY
        reading("123ABC", "123", 0.97, split="valid"),  # una moto: lee solo los dígitos
    ]

    report = ocr.build_report(reads, 0.6, "parseq@abc", Counter({"marcador de posición": 1}), 4)

    assert report["propuesta"]["OCR_MIN_CONF"] == ocr.PROPOSAL_FLOOR
    assert report["propuesta_con_todos_los_formatos"]["OCR_MIN_CONF"] == 0.98
    assert report["lecturas_fuera_de_alcance"] == 1
    assert report["cajas_del_dataset"] == 4 and report["descartadas"] == {"marcador de posición": 1}
    assert set(report["por_formato"]) == {"mercosur", "anterior", "otro"}
    assert set(report["por_split"]) == {"train", "valid"}
    assert report["total"]["lecturas"] == 3
    json.dumps(report)  # serializable


def test_el_reporte_no_contiene_patentes() -> None:
    reads = [reading("AB123CD", "AB123CD", 0.9), reading("EF456GH", "EF456GH", 0.8)]

    text = json.dumps(ocr.build_report(reads, 0.6, "parseq@abc", Counter(), 2))

    assert "AB123CD" not in text and "EF456GH" not in text


# --- auditoría -----------------------------------------------------------------------------


def items(count_right: int, count_wrong: int) -> list[tuple[Any, Any]]:
    crop = np.full((20, 40, 3), 200, np.uint8)
    right = [(reading("AB123CD", "AB123CD", 0.9), crop) for _ in range(count_right)]
    wrong = [(reading("AB123CD", "AB129CD", 0.9), crop) for _ in range(count_wrong)]
    return right + wrong


def test_la_auditoria_incluye_todo_lo_que_falla_y_una_muestra_de_lo_que_acierta() -> None:
    selection = ocr.audit_selection(items(10, 3), 0.6, match_samples=4, seed=0)

    assert len(selection) == 7
    assert sum(1 for read, _ in selection if not read.exact) == 3


def test_un_acierto_con_confianza_baja_tambien_se_audita() -> None:
    low = [(reading("AB123CD", "AB123CD", 0.4), np.zeros((20, 40, 3), np.uint8))]

    assert len(ocr.audit_selection(low, 0.6, match_samples=0, seed=0)) == 1


def test_una_correccion_por_confusion_se_audita() -> None:
    corrected = [(reading("AB123CD", "A8123CD", 0.9), np.zeros((20, 40, 3), np.uint8))]

    assert len(ocr.audit_selection(corrected, 0.6, match_samples=0, seed=0)) == 1


def test_la_muestra_es_reproducible_y_acotada() -> None:
    a = ocr.audit_selection(items(10, 0), 0.6, match_samples=4, seed=7)
    b = ocr.audit_selection(items(10, 0), 0.6, match_samples=4, seed=7)

    assert len(a) == len(b) == 4
    assert len(ocr.audit_selection(items(2, 0), 0.6, match_samples=50, seed=0)) == 2


def test_escribe_los_recortes_el_csv_y_el_html_escapado(tmp_path: Path) -> None:
    crop = np.full((20, 40, 3), 200, np.uint8)
    nasty = reading("AB123CD", "<script>alert(1)</script>", 0.5)

    ocr.write_audit([(nasty, crop)], tmp_path, 0.6)

    assert (tmp_path / "audit" / "001.jpg").is_file()
    (row,) = list(csv.DictReader((tmp_path / "audit.csv").open(encoding="utf-8")))
    assert row["recorte"] == "audit/001.jpg" and row["etiqueta"] == "AB123CD"
    assert row["resultado"] == "PLATE_UNREADABLE" and row["exacta"] == "no"
    page = (tmp_path / "audit.html").read_text(encoding="utf-8")
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_reescribir_la_auditoria_borra_los_recortes_viejos(tmp_path: Path) -> None:
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "099.jpg").write_bytes(b"viejo")

    ocr.write_audit([], tmp_path, 0.6)

    assert not (tmp_path / "audit" / "099.jpg").exists()
    assert (tmp_path / "audit.html").is_file()


# --- línea de comandos ---------------------------------------------------------------------


def run_main(tmp_path: Path, extra: list[str] | None = None) -> tuple[int, Path]:
    dataset = make_dataset(
        tmp_path / "dataset",
        {
            "train": [
                ("a.jpg", [("AB123CD", (0, 0, 60, 40), 1), ("XY999ZW", (100, 50, 60, 40), 2)]),
                ("b.jpg", [("AA000AA", (0, 0, 60, 40), 3)]),
            ]
        },
    )
    manifest, models_dir = write_env(tmp_path / "env", tiny_parseq("AB123CD"))
    out = tmp_path / "out"
    args = [
        "--data-dir", str(dataset), "--out-dir", str(out), "--models-dir", str(models_dir),
        "--manifest", str(manifest), "--match-samples", "5", *(extra or []),
    ]  # fmt: skip
    return ocr.main(args), out


def test_main_de_punta_a_punta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "env").mkdir()

    code, out = run_main(tmp_path)

    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["cajas_del_dataset"] == 3 and report["descartadas"] == {"marcador de posición": 1}
    assert report["total"]["lecturas"] == 2
    assert report["total"]["distribucion"]["PLATE_MATCH"] == 1
    assert report["total"]["distribucion"]["PLATE_MISMATCH"] == 1
    assert report["propuesta"]["OCR_MIN_CONF"] >= ocr.PROPOSAL_FLOOR
    assert report["modelo"].startswith("parseq@")
    assert (out / "audit.html").is_file() and (out / "audit.csv").is_file()
    shown = capsys.readouterr().out
    assert "OCR_MIN_CONF" in shown
    assert "AB123CD" not in shown and "XY999ZW" not in shown  # la consola no muestra patentes


def test_main_se_niega_a_escribir_ni_leer_dentro_del_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for flag in ("--data-dir", "--out-dir"):
        args = {"--data-dir": str(tmp_path), "--out-dir": str(tmp_path), "--models-dir": str(tmp_path)}
        args[flag] = str(REPO_ROOT / "dato")
        argv = [item for pair in args.items() for item in pair]

        assert ocr.main(argv) == 1
        assert "repositorio" in capsys.readouterr().err


def test_main_con_un_modelo_que_no_esta_en_el_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "env").mkdir()

    code, _ = run_main(tmp_path, ["--model", "fantasma"])

    assert code == 1
    assert "No se pudo empezar" in capsys.readouterr().err


def test_main_exige_las_rutas(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EVAL_DATA_DIR", "EVAL_OUT", "MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        ocr.main([])


def test_min_conf_fuera_de_rango(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ocr.parse_args(
            [
                "--data-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--models-dir",
                str(tmp_path),
                "--min-conf",
                "2",
            ]
        )
