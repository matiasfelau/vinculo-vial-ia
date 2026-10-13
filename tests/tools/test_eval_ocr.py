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

    loaded = ocr.load_samples(root, ["train"])

    assert [s.label for s in loaded.samples] == ["AB123CD", "XY999ZW"]
    first = loaded.samples[0]
    assert (first.split, first.file) == ("train", "a.jpg")
    assert (first.box.x1, first.box.y1, first.box.x2, first.box.y2) == (10, 20, 110, 60)
    assert loaded.discarded == Counter({"marcador de posición": 1})


def test_carga_descarta_etiquetas_vacias(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("---", (0, 0, 50, 30), 1)])]})

    loaded = ocr.load_samples(root, ["train"])

    assert loaded.samples == []
    assert loaded.discarded == Counter({"etiqueta vacía": 1})


def test_carga_solo_los_splits_pedidos(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {
            "train": [("a.jpg", [("AB123CD", (0, 0, 50, 30), 1)])],
            "valid": [("b.jpg", [("EF456GH", (0, 0, 50, 30), 1)])],
        },
    )

    samples = ocr.load_samples(root, ["valid"]).samples

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
    samples = ocr.load_samples(root, ["train"]).samples
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
    samples = ocr.load_samples(root, ["train"]).samples

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
    samples = ocr.load_samples(root, ["train"]).samples
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


def run_main(
    tmp_path: Path, extra: list[str] | None = None, overrides: str | None = None
) -> tuple[int, Path]:
    root = tmp_path / "data"
    make_dataset(
        root / ocr.DATASET_DIR,
        {
            "train": [
                ("a.jpg", [("AB123CD", (0, 0, 60, 40), 1), ("XY999ZW", (100, 50, 60, 40), 2)]),
                ("b.jpg", [("AA000AA", (0, 0, 60, 40), 3)]),
            ]
        },
    )
    if overrides is not None:
        (root / ocr.OVERRIDES_PATH).parent.mkdir(parents=True)
        (root / ocr.OVERRIDES_PATH).write_text(overrides, encoding="utf-8")
    manifest, models_dir = write_env(tmp_path / "env", tiny_parseq("AB123CD"))
    out = tmp_path / "out"
    args = [
        "--data-dir", str(root), "--out-dir", str(out), "--models-dir", str(models_dir),
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


# --- correcciones de etiquetas -----------------------------------------------------------------


def write_overrides(path: Path, rows: list[str], header: str = "archivo,etiqueta,motivo") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_sin_archivo_de_correcciones_no_hay_correcciones(tmp_path: Path) -> None:
    assert ocr.load_overrides(tmp_path / "no-existe.csv") == {}


def test_carga_las_correcciones_y_normaliza_la_etiqueta(tmp_path: Path) -> None:
    path = write_overrides(
        tmp_path / "o.csv", ["a.jpg,hsc 691,etiqueta incorrecta", "b.jpg,excluir,patente ilegible"]
    )

    assert ocr.load_overrides(path) == {"a.jpg": "HSC691", "b.jpg": ocr.EXCLUDE}


@pytest.mark.parametrize(
    ("rows", "header", "message"),
    [
        (["a.jpg,AB123CD,x", "a.jpg,EF456GH,x"], "archivo,etiqueta,motivo", "repetido"),
        ([",AB123CD,x"], "archivo,etiqueta,motivo", "faltan"),
        (["a.jpg,,x"], "archivo,etiqueta,motivo", "faltan"),
        (["a.jpg,---,x"], "archivo,etiqueta,motivo", "faltan"),
        (["a.jpg,AB123CD"], "archivo,etiqueta", "columnas"),
    ],
)
def test_correcciones_invalidas(tmp_path: Path, rows: list[str], header: str, message: str) -> None:
    path = write_overrides(tmp_path / "o.csv", rows, header)

    with pytest.raises(ocr.OverridesError, match=message):
        ocr.load_overrides(path)


def test_una_correccion_cambia_la_etiqueta_sin_tocar_el_dataset(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1)])]})
    before = (root / "train" / ocr.ANNOTATIONS).read_bytes()

    loaded = ocr.load_samples(root, ["train"], {"a.jpg": "EF456GH"})

    assert [(s.label, s.corrected) for s in loaded.samples] == [("EF456GH", True)]
    assert (root / "train" / ocr.ANNOTATIONS).read_bytes() == before


def test_una_foto_excluida_se_cuenta_como_ilegible_y_no_como_descartada(tmp_path: Path) -> None:
    root = make_dataset(
        tmp_path,
        {
            "train": [
                ("a.jpg", [("AB123CD", (0, 0, 60, 40), 1), ("EF456GH", (100, 50, 60, 40), 2)]),
                ("b.jpg", [("IJ789KL", (0, 0, 60, 40), 1)]),
            ]
        },
    )

    loaded = ocr.load_samples(root, ["train"], {"a.jpg": ocr.EXCLUDE})

    assert loaded.unreadable == 2  # una corrección vale para todas las cajas de la foto
    assert [s.file for s in loaded.samples] == ["b.jpg"]
    assert not loaded.discarded


def test_una_correccion_a_un_marcador_de_posicion_se_descarta(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1)])]})

    loaded = ocr.load_samples(root, ["train"], {"a.jpg": "AA000AA"})

    assert loaded.discarded == Counter({"marcador de posición": 1})


def test_una_correccion_a_una_foto_que_no_existe_es_un_error(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1)])]})

    with pytest.raises(ocr.OverridesError, match="no están en el dataset: b.jpg"):
        ocr.load_samples(root, ["train"], {"b.jpg": "EF456GH"})


def test_main_aplica_las_correcciones_y_las_informa_aparte(tmp_path: Path) -> None:
    (tmp_path / "env").mkdir()
    overrides = "archivo,etiqueta,motivo\na.jpg,EXCLUIR,patente ilegible\n"

    code, out = run_main(tmp_path, overrides=overrides)

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0
    assert report["ilegibles"] == 2  # las dos cajas de a.jpg
    assert report["total"]["lecturas"] == 0
    assert report["total"]["fallas_del_modelo"] == 0  # no es una falla del modelo
    assert report["cajas_del_dataset"] == 3


def test_main_corrige_una_etiqueta_y_lo_marca_en_la_auditoria(tmp_path: Path) -> None:
    (tmp_path / "env").mkdir()
    overrides = "archivo,etiqueta,motivo\na.jpg,AB123CD,etiqueta incorrecta\n"

    code, out = run_main(tmp_path, overrides=overrides)

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["etiquetas_corregidas"] == 2
    rows = list(csv.DictReader((out / "audit.csv").open(encoding="utf-8")))
    assert {row["etiqueta_corregida"] for row in rows} == {"si"}


def test_main_con_correcciones_invalidas_no_empieza(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "env").mkdir()

    code, _ = run_main(tmp_path, overrides="archivo,etiqueta,motivo\nfantasma.jpg,AB123CD,x\n")

    assert code == 1
    assert "No se pudo empezar" in capsys.readouterr().err


# --- causas ----------------------------------------------------------------------------------


def plate_image(text: str = "AB123CD", height: int = 100, width: int = 330) -> Any:
    """Texto negro sobre blanco: una "patente" sintética."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(
        img, text, (8, int(height * 0.72)), cv2.FONT_HERSHEY_DUPLEX, height / 42, (0, 0, 0), 3, cv2.LINE_AA
    )
    return img


def rotated(img: Any, degrees: float) -> Any:
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (width, height), borderValue=(255, 255, 255))


def sample_of(label: str, box: tuple[float, float, float, float] = (20, 20, 120, 60), size=(640, 480)) -> Any:
    return ocr.Sample("train", "f.jpg", Box(*box), label, size)


def reading_of(label: str, text: str, conf: float = 0.5, raw: str | None = None, **sample_kwargs: Any) -> Any:
    return ocr.Reading(
        sample_of(label, **sample_kwargs), text, conf, failed=False, raw=text if raw is None else raw
    )


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("AB123CD", "A8123CD"),  # B/8
        ("AE015TG", "AEG15TG"),  # 0 leído G
        ("AE622RW", "AE622RM"),  # W leída M
        ("AG101OA", "AG1010A"),  # O/0
        ("AB123CD", "ABI23CD"),  # I/1
        ("ABC123", "ABC1Z3"),  # Z/2
    ],
)
def test_confusiones_de_caracteres_parecidos(label: str, text: str) -> None:
    assert ocr.only_confusions(label, text)


@pytest.mark.parametrize(
    ("label", "text"),
    [("AB123CD", "AB123CD"), ("AB123CD", "AB123C"), ("AB123CD", "AE123CD"), ("AB123CD", "ZZ999ZZ")],
)
def test_no_son_solo_confusiones(label: str, text: str) -> None:
    assert not ocr.only_confusions(label, text)


@pytest.mark.parametrize("raw", ["REPUBLICAAE622RW", "ARGENTINA AE622RW", "mercosur", "AE622RWARGEN"])
def test_detecta_fragmentos_de_la_franja(raw: str) -> None:
    assert ocr.has_stripe(raw)


@pytest.mark.parametrize("raw", ["AE622RW", "ABC123", "", "AB 123 CD"])
def test_sin_franja(raw: str) -> None:
    assert not ocr.has_stripe(raw)


def test_la_inclinacion_del_texto() -> None:
    assert abs(ocr.text_tilt(plate_image())) < 3
    assert abs(abs(ocr.text_tilt(rotated(plate_image(), 20))) - 20) < 4
    assert abs(abs(ocr.text_tilt(rotated(plate_image(), -15))) - 15) < 4


def test_sin_texto_claro_no_hay_inclinacion() -> None:
    assert ocr.text_tilt(np.full((100, 300, 3), 255, np.uint8)) is None
    noise = np.random.default_rng(0).integers(0, 256, (100, 300, 3), dtype=np.uint8)
    assert ocr.text_tilt(noise) is None


def test_la_nitidez_baja_con_el_desenfoque() -> None:
    sharp = plate_image()

    assert ocr.sharpness(cv2.GaussianBlur(sharp, (0, 0), 6)) < ocr.MIN_SHARPNESS < ocr.sharpness(sharp)


def test_una_lectura_con_solo_una_confusion() -> None:
    result = ocr.diagnose(reading_of("AB123CD", "A8123CD"), plate_image())

    assert result.cause == ocr.CONFUSION and result.factors == (ocr.CONFUSION,)


def test_una_moto_gana_a_las_demas_causas_pero_los_factores_se_conservan() -> None:
    small = plate_image(height=30, width=90)

    result = ocr.diagnose(reading_of("123ABC", "123"), small)

    assert result.cause == ocr.TWO_LINES
    assert {ocr.TWO_LINES, ocr.BLUR, ocr.CUT} <= set(result.factors)


def test_baja_resolucion() -> None:
    result = ocr.diagnose(reading_of("AB123CD", "AB129CD"), plate_image(height=30, width=90))

    assert result.cause == ocr.BLUR


def test_placa_inclinada() -> None:
    result = ocr.diagnose(reading_of("AB123CD", "AB129CD"), rotated(plate_image(), 20))

    assert result.cause == ocr.ANGLE


def test_lee_la_franja() -> None:
    result = ocr.diagnose(reading_of("AB123CD", "AB129CD", raw="REPUBLICAARGENTINAAB129CD"), plate_image())

    assert result.cause == ocr.STRIPE


def test_recorte_cortado_por_el_borde_de_la_foto() -> None:
    result = ocr.diagnose(
        reading_of("AB123CD", "AB123", box=(600, 20, 40, 60), size=(640, 480)), plate_image()
    )

    assert result.cause == ocr.CUT


def test_un_solo_caracter_de_menos_no_es_un_recorte_cortado() -> None:
    result = ocr.diagnose(reading_of("UUN013", "UN013"), plate_image())

    assert ocr.CUT not in result.factors and result.cause == ocr.OTHER_CAUSE


def test_etiqueta_dudosa_solo_con_lectura_muy_segura_de_una_patente_valida() -> None:
    sure = ocr.diagnose(reading_of("AB123CD", "ZZ999ZZ", conf=0.97), plate_image())
    unsure = ocr.diagnose(reading_of("AB123CD", "ZZ999ZZ", conf=0.6), plate_image())

    assert sure.cause == ocr.DOUBTFUL_LABEL
    assert unsure.cause == ocr.OTHER_CAUSE


def test_reflejo() -> None:
    glare = np.full((100, 300, 3), 255, np.uint8)

    assert ocr.GLARE in ocr.diagnose(reading_of("AB123CD", "ZZ999ZZ"), glare).factors


def test_el_reporte_de_causas_cuenta_la_causa_principal_y_los_factores() -> None:
    diagnoses = [
        ocr.Diagnosis(ocr.BLUR, (ocr.BLUR, ocr.CONFUSION)),
        ocr.Diagnosis(ocr.CONFUSION, (ocr.CONFUSION,)),
        ocr.Diagnosis(ocr.OTHER_CAUSE, ()),
    ]

    report = ocr.causes_report(diagnoses)

    assert report["lecturas_sin_coincidencia_exacta"] == 3
    assert report["causa_principal"][ocr.BLUR] == 1 and report["causa_principal"][ocr.OTHER_CAUSE] == 1
    assert set(report["causa_principal"]) == set(ocr.CAUSES)
    assert report["factores_presentes"][ocr.CONFUSION] == 2
    assert ocr.OTHER_CAUSE not in report["factores_presentes"]


def test_el_reporte_de_causas_no_contiene_patentes() -> None:
    reads = [reading_of("AB123CD", "ZZ999ZZ", 0.97)]
    diagnoses = [ocr.diagnose(reads[0], plate_image())]

    text = json.dumps(ocr.build_report(reads, 0.6, "parseq@abc", Counter(), 1, 0, diagnoses))

    assert "AB123CD" not in text and "ZZ999ZZ" not in text


def test_read_samples_guarda_el_texto_crudo_si_el_lector_lo_ofrece(tmp_path: Path) -> None:
    class RawReader(FakeReader):
        def raw_text(self, crop: Any) -> str:
            return "REPUBLICAAB123CD"

    root = make_dataset(tmp_path, {"train": [("a.jpg", [("AB123CD", (0, 0, 60, 40), 1)])]})
    samples = ocr.load_samples(root, ["train"]).samples

    ((with_raw, _),), _ = ocr.read_samples(samples, root, RawReader({1: ("AB123CD", 0.9)}))
    ((without_raw, _),), _ = ocr.read_samples(samples, root, FakeReader({1: ("AB123CD", 0.9)}))

    assert with_raw.raw == "REPUBLICAAB123CD"
    assert without_raw.raw == "AB123CD"  # sin raw_text, el texto crudo es el texto leído


def test_main_escribe_la_causa_en_la_auditoria_y_en_el_reporte(tmp_path: Path) -> None:
    (tmp_path / "env").mkdir()

    code, out = run_main(tmp_path)

    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["causas"]["lecturas_sin_coincidencia_exacta"] == 1
    rows = list(csv.DictReader((out / "audit.csv").open(encoding="utf-8")))
    wrong = [row for row in rows if row["exacta"] == "no"]
    assert [row["causa"] for row in wrong] == [ocr.BLUR]  # los recortes sintéticos son diminutos
    assert ocr.DOUBTFUL_LABEL in wrong[0]["factores"]


def test_pares_cruzados_cuenta_que_parte_de_las_patentes_ajenas_sale_mismatch() -> None:
    readings = [
        reading("AB123CD", "AB123CD", 0.95),
        reading("EF456GH", "EF456GH", 0.90),
        reading("IJ789KL", "IJ789KL", 0.40),  # lectura poco segura: se escapa como ilegible
    ]

    result = ocr.cross_pairs(readings, 0.78)

    assert result["todos"]["pares"] == 6  # cada lectura contra las 2 etiquetas ajenas
    assert result["todos"]["distribucion"]["PLATE_MISMATCH"] == 4
    assert result["todos"]["distribucion"]["PLATE_UNREADABLE"] == 2
    assert result["todos"]["mismatch_pct"] == pytest.approx(66.67)
    assert result["con_lectura_segura"] == {
        "pares": 4,
        "mismatch": 4,
        "mismatch_pct": 100.0,
        "distribucion": {"PLATE_MATCH": 0, "PLATE_FUZZY": 0, "PLATE_MISMATCH": 4, "PLATE_UNREADABLE": 0},
    }


def test_pares_cruzados_no_cuenta_la_etiqueta_propia_ni_las_lecturas_vacias() -> None:
    readings = [reading("AB123CD", "AB123CD", 0.95), reading("EF456GH", "", 0.0)]

    result = ocr.cross_pairs(readings, 0.78)

    assert result["todos"]["pares"] == 1 and result["todos"]["distribucion"]["PLATE_MATCH"] == 0


def test_pares_cruzados_con_etiquetas_casi_iguales_se_escapan_como_fuzzy() -> None:
    readings = [reading("AB123CD", "AB123CD", 0.95), reading("AB123CE", "AB123CE", 0.95)]

    result = ocr.cross_pairs(readings, 0.78)

    assert result["todos"]["distribucion"]["PLATE_FUZZY"] == 2 and result["todos"]["mismatch"] == 0
