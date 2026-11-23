"""tools/eval/ocr_labels.py y el set etiquetado en `ocr.py`: el xlsx de etiquetas, sin datos reales ni OCR."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("PIL")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))
import ocr_labels  # noqa: E402


def load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / "eval" / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ocr = load("eval_ocr_for_labels", "ocr.py")
ocr_set = load("eval_ocr_set_for_labels", "ocr_set.py")

LABEL_COLUMNS = ["id", "archivo", "recorte", "texto", "formato", "luz", "estado", "revisar_moto"]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_set(root: Path, count: int = 4) -> tuple[Path, Path]:
    """Un set como el de `ocr_set.py`: recortes, etiquetas.csv, cajas.csv y las fotos enteras."""
    set_dir, photos = root / "set", root / "photos"
    (set_dir / "recortes").mkdir(parents=True)
    (photos / "train").mkdir(parents=True)
    labels, boxes = [], []
    for number in range(1, count + 1):
        ident = f"{number:03d}"
        photo = np.full((200, 300, 3), 90 + number * 20, np.uint8)
        cv2.rectangle(photo, (60, 50), (200, 100), (240, 240, 240), -1)
        assert cv2.imwrite(str(photos / "train" / f"p{number}.jpg"), photo)
        assert cv2.imwrite(str(set_dir / "recortes" / f"{ident}.png"), photo[40:110, 50:210])
        labels.append(
            {
                "id": ident,
                "archivo": f"train/p{number}.jpg",
                "recorte": f"recortes/{ident}.png",
                "texto": "",
                "formato": "",
                "luz": "noche" if number % 2 == 0 else "dia",
                "estado": "",
                "revisar_moto": "sí" if number == 2 else "",
            }
        )
        boxes.append(
            {
                "id": ident,
                "archivo": f"train/p{number}.jpg",
                "caja": "60 50 140 50",
                "relacion_aspecto": "2.80",
            }
        )
    write_csv(set_dir / "etiquetas.csv", LABEL_COLUMNS, labels)
    write_csv(set_dir / "cajas.csv", ["id", "archivo", "caja", "relacion_aspecto"], boxes)
    return set_dir, photos


def fill(set_dir: Path, values: dict[str, dict[str, Any]]) -> None:
    """Carga etiquetas en el xlsx como lo haría una persona: `values` mapea el id a {columna: valor}."""
    path = set_dir / "etiquetas.xlsx"
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["etiquetas"]
    header = {cell.value: cell.column for cell in sheet[1]}
    for row in range(2, sheet.max_row + 1):
        for column, value in values.get(sheet.cell(row=row, column=1).value, {}).items():
            sheet.cell(row=row, column=header[column], value=value)
    workbook.save(path)


def test_el_xlsx_tiene_una_fila_por_id_con_miniatura_encabezado_congelado_y_filtros(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    path = ocr_labels.build_xlsx(set_dir, photos)
    sheet = openpyxl.load_workbook(path)["etiquetas"]
    assert [cell.value for cell in sheet[1]] == list(ocr_labels.COLUMNS)
    assert [sheet.cell(row=r, column=1).value for r in range(2, 6)] == ["001", "002", "003", "004"]
    assert len(sheet._images) == 4  # una miniatura por fila
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref == "A1:G5"
    assert sheet.column_dimensions["B"].width == 60
    assert sheet.row_dimensions[2].height == ocr_labels.ROW_HEIGHT_PT
    assert [sheet.cell(row=r, column=5).value for r in range(2, 6)] == ["dia", "noche", "dia", "noche"]  # luz
    assert [sheet.cell(row=r, column=7).value for r in range(2, 6)] == [None, "sí", None, None]
    assert [sheet.cell(row=r, column=3).value for r in range(2, 6)] == [None] * 4  # texto vacío


def test_las_columnas_de_solo_lectura_quedan_bloqueadas_y_en_gris_y_las_demas_editables(
    tmp_path: Path,
) -> None:
    set_dir, photos = make_set(tmp_path)
    sheet = openpyxl.load_workbook(ocr_labels.build_xlsx(set_dir, photos))["etiquetas"]
    assert sheet.protection.sheet is True and sheet.protection.password is None
    assert sheet.protection.objects is True  # las miniaturas no se mueven
    assert (
        sheet.protection.autoFilter is False and sheet.protection.sort is False
    )  # filtrar y ordenar, permitido
    for letter in "ABG":  # id, miniatura, revisar_moto
        cell = sheet[f"{letter}2"]
        assert cell.protection.locked is True
        assert cell.fill.start_color.rgb.endswith(ocr_labels.COLOR_LOCKED)
    for letter in "CDEF":  # texto, formato, luz, estado
        assert sheet[f"{letter}2"].protection.locked is False


def test_los_desplegables_y_la_validacion_del_texto(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    sheet = openpyxl.load_workbook(ocr_labels.build_xlsx(set_dir, photos))["etiquetas"]
    by_range = {str(v.sqref): v for v in sheet.data_validations.dataValidation}
    assert set(by_range) == {"C2:C5", "D2:D5", "E2:E5", "F2:F5"}
    assert (by_range["D2:D5"].type, by_range["D2:D5"].formula1) == ("list", '"anterior,mercosur,moto,otro"')
    assert (by_range["E2:E5"].type, by_range["E2:E5"].formula1) == ("list", '"dia,noche"')
    assert (by_range["F2:F5"].type, by_range["F2:F5"].formula1) == (
        "list",
        '"legible,ilegible,fuera_de_alcance"',
    )
    text = by_range["C2:C5"]
    assert text.type == "custom" and text.formula1 == ocr_labels.TEXT_FORMULA
    assert text.allow_blank is True and text.showErrorMessage is True and text.errorStyle == "stop"
    assert "6 o 7 caracteres" in text.error and "MAYÚSCULAS" in text.error and "guiones" in text.error
    for validation in by_range.values():
        assert validation.allow_blank is True and validation.showErrorMessage is True


def test_resalta_las_filas_a_revisar_y_las_legibles_sin_texto(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    sheet = openpyxl.load_workbook(ocr_labels.build_xlsx(set_dir, photos))["etiquetas"]
    ranges = list(sheet.conditional_formatting)
    assert [str(r.sqref) for r in ranges] == ["A2:G5"]
    rules = sorted(ranges[0].rules, key=lambda rule: rule.priority)
    assert rules[0].formula == ['AND($F2="legible",LEN($C2)=0)'] and rules[0].stopIfTrue is True
    assert rules[1].formula == ['$G2="sí"']
    assert rules[0].dxf.fill.start_color.rgb.endswith(ocr_labels.COLOR_MISSING_TEXT)
    assert rules[1].dxf.fill.start_color.rgb.endswith(ocr_labels.COLOR_REVIEW_MOTO)


def test_el_xlsx_trae_una_hoja_de_instrucciones(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    workbook = openpyxl.load_workbook(ocr_labels.build_xlsx(set_dir, photos))
    assert workbook.sheetnames == ["etiquetas", "instrucciones"]


def test_armar_el_xlsx_no_toca_el_csv_ni_pisa_uno_con_etiquetas(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    before = (set_dir / "etiquetas.csv").read_bytes()
    path = ocr_labels.build_xlsx(set_dir, photos)
    assert (set_dir / "etiquetas.csv").read_bytes() == before

    ocr_labels.build_xlsx(set_dir, photos)  # sin etiquetas: se puede regenerar
    fill(set_dir, {"001": {"estado": "legible", "texto": "AB123CD"}})
    labeled = path.read_bytes()
    with pytest.raises(ocr_labels.LabelsError, match="ya tiene etiquetas"):
        ocr_labels.build_xlsx(set_dir, photos)
    assert path.read_bytes() == labeled


def test_las_etiquetas_del_xlsx_se_leen_junto_con_los_datos_del_csv(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, photos)
    fill(
        set_dir,
        {
            "001": {"texto": "AB123CD", "formato": "mercosur", "estado": "legible"},
            "002": {"texto": 654321, "estado": "legible", "luz": "dia"},  # un número, sin texto de Excel
            "003": {"estado": "ilegible"},
        },
    )
    rows = {row["id"]: row for row in ocr_labels.read_label_rows(set_dir)}
    assert rows["001"]["texto"] == "AB123CD" and rows["001"]["formato"] == "mercosur"
    assert rows["002"]["texto"] == "654321" and rows["002"]["luz"] == "dia"  # lo que cambió la persona
    assert rows["003"]["estado"] == "ilegible" and rows["004"]["estado"] == ""
    assert rows["002"]["archivo"] == "train/p2.jpg" and rows["002"]["revisar_moto"] == "sí"  # del csv
    assert ocr_labels.labels_loaded(set_dir) is True


def test_sin_xlsx_se_lee_el_csv_y_la_luz_precargada_no_cuenta_como_etiqueta(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    assert [row["luz"] for row in ocr_labels.read_label_rows(set_dir)] == ["dia", "noche", "dia", "noche"]
    assert ocr_labels.labels_loaded(set_dir) is False
    ocr_labels.build_xlsx(set_dir, photos)
    assert ocr_labels.labels_loaded(set_dir) is False  # el xlsx recién armado solo trae la luz


def test_un_xlsx_con_otros_ids_que_el_csv_es_un_error(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, photos)
    write_csv(
        set_dir / "etiquetas.csv", LABEL_COLUMNS, ocr_labels.read_csv_rows(set_dir / "etiquetas.csv")[:3]
    )
    with pytest.raises(ocr_labels.LabelsError, match="no tiene los mismos ids"):
        ocr_labels.read_label_rows(set_dir)


def test_un_archivo_que_no_es_un_xlsx_da_un_error_claro(tmp_path: Path) -> None:
    set_dir, _ = make_set(tmp_path)
    (set_dir / "etiquetas.xlsx").write_bytes(b"no es un zip")
    with pytest.raises(ocr_labels.LabelsError, match="no se pudo abrir"):
        ocr_labels.read_label_rows(set_dir)


def test_convertir_a_csv_copia_el_csv_anterior_y_es_idempotente(tmp_path: Path) -> None:
    set_dir, photos = make_set(tmp_path)
    original = (set_dir / "etiquetas.csv").read_bytes()
    ocr_labels.build_xlsx(set_dir, photos)
    assert ocr_labels.to_csv(set_dir) is None  # sin cambios, no hay nada que hacer
    fill(set_dir, {"001": {"texto": "AB123CD", "formato": "mercosur", "estado": "legible"}})

    backup = ocr_labels.to_csv(set_dir)

    assert backup is not None and backup.name == "etiquetas-previo-1.csv"
    assert backup.read_bytes() == original
    rows = ocr_labels.read_csv_rows(set_dir / "etiquetas.csv")
    assert list(rows[0]) == LABEL_COLUMNS
    assert (rows[0]["texto"], rows[0]["formato"], rows[0]["estado"]) == ("AB123CD", "mercosur", "legible")
    assert rows[1]["revisar_moto"] == "sí"
    assert ocr_labels.to_csv(set_dir) is None  # ya está al día: no hay una segunda copia
    assert not (set_dir / "etiquetas-previo-2.csv").exists()


def test_el_avance_y_los_problemas_se_informan_sin_mostrar_patentes() -> None:
    rows = [
        {"id": "001", "texto": "AB123CD", "formato": "mercosur", "luz": "dia", "estado": "legible"},
        {"id": "002", "texto": "", "formato": "", "luz": "noche", "estado": "legible"},
        {"id": "003", "texto": "", "formato": "", "luz": "dia", "estado": "ilegible"},
        {"id": "004", "texto": "ab-123", "formato": "viejo", "luz": "tarde", "estado": "dudoso"},
        {"id": "005", "texto": "", "formato": "", "luz": "dia", "estado": ""},
    ]
    report = ocr_labels.check_rows(rows)
    assert report["filas"] == 5 and report["legibles"] == 2 and report["ilegibles"] == 1
    assert report["sin_estado"] == 1 and report["legibles_sin_texto"] == 1
    assert report["con_texto_sin_estado_legible"] == 1  # la 004: tiene texto y su estado no es legible
    assert report["ids_con_problemas"] == ["004"] and report["filas_con_problemas"] == 1
    assert "AB123CD" not in json.dumps(report)
    assert ocr_labels.row_problems(rows[3]) == [
        "formato 'viejo'",
        "luz 'tarde'",
        "estado 'dudoso'",
        "texto 'ab-123'",
    ]


@pytest.mark.parametrize(
    ("text", "valid"),
    [("AB123CD", True), ("ABC123", True), ("AB123C", True), ("123456", True)]
    + [("abc123", False), ("AB 123CD", False), ("AB-123CD", False), ("ABCDE", False), ("ABC12345", False)]
    + [("AB1.23C", False), ("AB123CD ", False)],
)
def test_el_patron_del_texto_coincide_con_lo_que_pide_la_validacion(text: str, valid: bool) -> None:
    # La fórmula de Excel (`TEXT_FORMULA`) se evaluó con LibreOffice sobre estos mismos casos: da lo mismo.
    assert bool(ocr_labels.TEXT_PATTERN.match(text)) is valid


def test_el_comando_xlsx_y_check_funcionan_y_se_niegan_dentro_del_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, photos = make_set(tmp_path)
    assert ocr_labels.main(["xlsx", "--set", str(set_dir), "--photos", str(photos)]) == 0
    assert (set_dir / "etiquetas.xlsx").is_file()
    assert ocr_labels.main(["check", "--set", str(set_dir)]) == 0
    assert "sin_estado: 4" in capsys.readouterr().out
    assert ocr_labels.main(["xlsx", "--set", str(REPO_ROOT / "set-de-prueba")]) == 1
    assert "dentro del repositorio" in capsys.readouterr().out


# --- ocr.py: el set etiquetado ----------------------------------------------------------------------


class FixedReader:
    """Un lector de OCR falso que siempre lee lo mismo."""

    def __init__(self, text: str = "AB123CD") -> None:
        self.text = text

    def read(self, crop: Any) -> tuple[str, float]:
        return self.text, 0.95


def labeled_set(tmp_path: Path) -> tuple[Path, Path]:
    set_dir, photos = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, photos)
    fill(
        set_dir,
        {
            "001": {"texto": "AB123CD", "formato": "mercosur", "estado": "legible"},
            "002": {"texto": "AB999ZZ", "formato": "mercosur", "estado": "legible"},
            "003": {"estado": "ilegible"},
            "004": {"texto": "ABC123", "formato": "moto", "estado": "legible", "luz": "noche"},
        },
    )
    return set_dir, photos


def test_la_evaluacion_lee_el_xlsx_sin_convertirlo(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    csv_before = (set_dir / "etiquetas.csv").read_bytes()

    loaded = ocr.load_set_samples(set_dir)

    assert (set_dir / "etiquetas.csv").read_bytes() == csv_before  # no hizo falta convertir nada
    assert [(s.file, s.label, s.format, s.light) for s in loaded.samples] == [
        ("p1.jpg", "AB123CD", "mercosur", "dia"),
        ("p2.jpg", "AB999ZZ", "mercosur", "noche"),
        ("p4.jpg", "ABC123", "moto", "noche"),
    ]
    assert loaded.unreadable == 1 and not loaded.discarded
    assert all(sample.split == "train" for sample in loaded.samples)
    assert (loaded.samples[0].box.x1, loaded.samples[0].box.x2, loaded.samples[0].box.y2) == (60, 200, 100)

    items, skipped = ocr.read_samples(loaded.samples, photos, FixedReader())
    assert not skipped and len(items) == 3
    assert items[0][0].sample.image_size == (300, 200)  # se completa con la foto entera
    report = ocr.build_report([r for r, _ in items], 0.78, "fake", loaded.discarded, 4, loaded.unreadable)
    assert set(report["por_luz"]) == {"dia", "noche"} and report["ilegibles"] == 1
    assert set(report["por_formato"]) == {"mercosur", "moto"}
    assert report["lecturas_fuera_de_alcance"] == 1  # la moto etiquetada a mano queda fuera de alcance
    assert report["total"]["lecturas"] == 3


def test_la_evaluacion_tambien_lee_el_csv_si_no_hay_xlsx(tmp_path: Path) -> None:
    set_dir, _ = make_set(tmp_path)
    rows = ocr_labels.read_csv_rows(set_dir / "etiquetas.csv")
    rows[0].update(texto="AB123CD", estado="legible")
    rows[1].update(estado="fuera_de_alcance")
    write_csv(set_dir / "etiquetas.csv", LABEL_COLUMNS, rows)
    loaded = ocr.load_set_samples(set_dir, incomplete=True)
    assert [s.label for s in loaded.samples] == ["AB123CD"]
    assert loaded.discarded == {"fuera de alcance": 1, "sin etiquetar": 2}


def test_la_evaluacion_falla_si_faltan_etiquetas_y_con_incomplete_las_salta(tmp_path: Path) -> None:
    set_dir, _ = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, None)
    fill(set_dir, {"001": {"texto": "AB123CD", "estado": "legible"}, "002": {"estado": "legible"}})
    with pytest.raises(ocr_labels.LabelsError, match=r"faltan etiquetas en 3 filas \(ids 002, 003, 004\)"):
        ocr.load_set_samples(set_dir)
    loaded = ocr.load_set_samples(set_dir, incomplete=True)
    assert [s.label for s in loaded.samples] == ["AB123CD"] and loaded.discarded == {"sin etiquetar": 3}


def test_la_evaluacion_rechaza_valores_no_validos_pegados_en_el_xlsx(tmp_path: Path) -> None:
    set_dir, _ = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, None)
    fill(set_dir, {"001": {"texto": "ab-123", "estado": "legible", "formato": "viejo"}})
    with pytest.raises(ocr_labels.LabelsError, match=r"valores no válidos en 1 filas \(ids 001\)"):
        ocr.load_set_samples(set_dir, incomplete=True)


def test_ocr_py_acepta_labeled_set_y_se_niega_dentro_del_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ocr.parse_args(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--models-dir", str(tmp_path)]
        + ["--labeled-set", str(tmp_path / "ocr-ar-set"), "--incomplete"]
    )
    assert args.labeled_set == tmp_path / "ocr-ar-set" and args.incomplete is True
    assert args.dataset_dir == tmp_path / "ocr-ar"  # las fotos enteras del set
    inside = REPO_ROOT / "set-de-prueba"
    code = ocr.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--models-dir", str(tmp_path)]
        + ["--labeled-set", str(inside)]
    )
    assert code == 1
    assert "dentro del repositorio" in capsys.readouterr().err


# --- ocr_set.py: no pisa etiquetas que están solo en el xlsx ----------------------------------------


def test_ocr_set_no_pisa_un_set_cuyas_etiquetas_estan_solo_en_el_xlsx(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, photos = make_set(tmp_path)
    ocr_labels.build_xlsx(set_dir, photos)
    fill(set_dir, {"001": {"texto": "AB123CD", "estado": "legible"}})
    assert (set_dir / "etiquetas.csv").read_bytes().count(b"AB123CD") == 0  # el csv no las tiene
    with pytest.raises(ValueError, match="etiquetas cargadas"):
        ocr_set.refresh(tmp_path / "x", tmp_path / "y", set_dir)
    unlabeled, _ = make_set(tmp_path / "otro")
    with pytest.raises(ValueError, match="de --out"):
        ocr_set.refresh(tmp_path / "x", tmp_path / "y", set_dir, original=unlabeled)

    only_xlsx = tmp_path / "solo-xlsx"
    only_xlsx.mkdir()
    (only_xlsx / "etiquetas.xlsx").write_bytes(b"cualquier cosa")
    code = ocr_set.main(["--src", str(tmp_path), "--reference", str(tmp_path), "--out", str(only_xlsx)])
    assert code == 1
    assert "etiquetas.xlsx ya existe" in capsys.readouterr().out


def test_la_forma_del_texto_tiene_que_ser_la_del_formato() -> None:
    ok = {"texto": "ABC123", "formato": "anterior", "luz": "dia", "estado": "legible"}
    assert ocr_labels.row_problems(ok) == []
    assert ocr_labels.row_problems({**ok, "texto": "AB123CD", "formato": "mercosur"}) == []
    assert ocr_labels.row_problems({**ok, "texto": "AB123CD"}) == [
        "el texto no tiene la forma del formato anterior"
    ]
    assert ocr_labels.row_problems({**ok, "texto": "ABC1234"}) == [
        "el texto no tiene la forma del formato anterior"
    ]
    assert ocr_labels.row_problems({**ok, "texto": "ABC123", "formato": "mercosur"}) == [
        "el texto no tiene la forma del formato mercosur"
    ]
    assert (
        ocr_labels.row_problems({**ok, "texto": "AB12345", "formato": "moto"}) == []
    )  # moto y otro: sin forma fija


def test_el_check_lista_ids_sin_estado_formato_roto_pares_y_cuenta_legibles_de_auto_y_de_noche() -> None:
    rows = [
        {"id": "001", "texto": "ABC123", "formato": "anterior", "luz": "noche", "estado": "legible"},
        {"id": "002", "texto": "ABC124", "formato": "anterior", "luz": "dia", "estado": "legible"},
        {"id": "003", "texto": "AB123CD", "formato": "anterior", "luz": "dia", "estado": "legible"},
        {"id": "004", "texto": "", "formato": "", "luz": "dia", "estado": ""},
        {"id": "005", "texto": "XYZ999", "formato": "", "luz": "dia", "estado": "legible"},
        {"id": "006", "texto": "KLM456", "formato": "moto", "luz": "noche", "estado": "legible"},
    ]

    report = ocr_labels.check_rows(rows)

    assert report["ids_sin_estado"] == ["004"] and report["legibles_sin_formato"] == ["005"]
    assert report["ids_con_problemas"] == ["003"]  # el texto no tiene la forma del formato
    assert report["legibles_de_auto"] == 3 and report["legibles_de_auto_de_noche"] == 1  # 001, 002 y 003
    assert report["pares_a_un_caracter"] == [["001", "002"]]
    assert "ABC123" not in json.dumps(report)
