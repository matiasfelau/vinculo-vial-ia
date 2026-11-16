"""tools/eval/ocr_mlkit.py: ML Kit contra PARSeq sobre el set etiquetado (datos sintéticos, sin modelos)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_ocr_mlkit", REPO_ROOT / "tools" / "eval" / "ocr_mlkit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_ocr_mlkit"] = module
    spec.loader.exec_module(module)
    return module


mk = load()
LABEL_COLUMNS = ["id", "archivo", "recorte", "texto", "formato", "luz", "estado", "revisar_moto"]

# id, texto, formato, luz, ML Kit (lectura, candidatos, confianza, ms), PARSeq (lectura, confianza)
CASES = [
    ("001", "ABC123", "anterior", "dia", ("ABC123", "ABC 123", "0.9", "100"), ("ABC123", "0.95")),  # exacta
    ("002", "AB123CD", "mercosur", "noche", ("", "REPUBLICA|AB 123 CD", "", "120"), ("AB123CD", "0.90")),
    ("003", "DEF456", "anterior", "dia", ("DEF458", "DEF 458|DEF 456", "0.5", "110"), ("DEF456", "0.90")),
    ("004", "GH789IJ", "mercosur", "noche", ("GH789IK", "GH 789 IK", "0.6", "130"), ("GH789IJ", "0.50")),
    ("005", "KLM012", "anterior", "dia", ("", "", "", "140"), ("KLM012", "0.95")),
    ("006", "NO345PQ", "mercosur", "dia", ("NO345PQ", "NO 345 PQ", "0.8", "105"), ("XY999ZW", "0.90")),
    ("007", "PQR678", "anterior", "dia", ("PQR999", "PQR 999", "0.7", "115"), ("PQR678", "0.95")),
]


def make_files(root: Path) -> tuple[Path, Path, Path]:
    set_dir = root / "set"
    (set_dir / "resultados").mkdir(parents=True)
    with (set_dir / "etiquetas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for ident, text, fmt, light, *_ in CASES:
            writer.writerow(
                {
                    "id": ident,
                    "archivo": "train/p.jpg",
                    "recorte": f"recortes/{ident}.png",
                    "texto": text,
                    "formato": fmt,
                    "luz": light,
                    "estado": "legible",
                    "revisar_moto": "",
                }
            )
        writer.writerow({**{c: "" for c in LABEL_COLUMNS}, "id": "008", "estado": "ilegible", "luz": "dia"})
    results = set_dir / "resultados" / "ocr-results.csv"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "formato", "luz", "lectura", "candidatos", "confianza", "tiempo_ms"])
        for ident, _, fmt, light, (reading, candidates, conf, ms), _ in CASES:
            writer.writerow([ident, fmt, light, reading, candidates, conf, ms])
    readings = root / "readings.csv"
    with readings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "lectura", "confianza", "falla"])
        for ident, *_, (text, conf) in CASES:
            writer.writerow([ident, text, conf, "no"])
    return set_dir, results, readings


def rows_of(tmp_path: Path) -> list[mk.Row]:
    set_dir, results, readings = make_files(tmp_path)
    return mk.build_rows(set_dir, mk.read_results(results), mk.read_parseq(readings))


def test_cruza_por_id_solo_las_legibles_de_auto(tmp_path: Path) -> None:
    rows = rows_of(tmp_path)

    assert [row.ident for row in rows] == [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
    ]  # sin la ilegible 008
    assert rows[1].candidates == ("REPUBLICA", "AB123CD") and rows[1].mlkit == ""


def test_si_los_ids_no_coinciden_falla_y_dice_cuantos(tmp_path: Path) -> None:
    set_dir, results, readings = make_files(tmp_path)
    parsed = mk.read_results(results)
    parsed.pop("003")

    with pytest.raises(mk.ComparisonError, match="ML Kit.*faltan 1, sobran 0"):
        mk.build_rows(set_dir, parsed, mk.read_parseq(readings))


def test_ml_kit_cuenta_exacta_sin_lectura_y_equivocada_en_total_y_por_luz(tmp_path: Path) -> None:
    report = mk.build_report(rows_of(tmp_path), 0.78)

    total, day, night = (report["mlkit"][name] for name in ("total", "dia", "noche"))
    assert (total["lectura_exacta"], total["sin_lectura"], total["lectura_equivocada"]) == (2, 2, 3)
    assert total["lectura_exacta_pct"] == 28.6 and total["recortes"] == 7
    assert (day["recortes"], day["lectura_exacta"], day["sin_lectura"], day["lectura_equivocada"]) == (
        5,
        2,
        1,
        2,
    )
    assert (
        night["recortes"],
        night["lectura_exacta"],
        night["sin_lectura"],
        night["lectura_equivocada"],
    ) == (
        2,
        0,
        1,
        1,
    )


def test_los_candidatos_dicen_si_la_correcta_aparecia_aunque_la_lectura_fuera_otra(tmp_path: Path) -> None:
    candidates = mk.build_report(rows_of(tmp_path), 0.78)["mlkit"]["total"]["candidatos"]

    # no exactas: 002 (sin lectura, la correcta aparece), 003 (equivocada, aparece), 004, 005 (sin
    # candidatos) y 007
    assert candidates["no_exactas"] == 5 and candidates["no_exactas_con_la_correcta"] == 2
    assert candidates["sin_lectura_con_la_correcta"] == 1 and candidates["equivocadas_con_la_correcta"] == 1
    assert candidates["la_correcta_aparece"] == 4  # las 2 exactas más 002 y 003
    assert candidates["sin_candidatos"] == 1
    # 004 (GH789IK contra GH789IJ) está a un carácter en los candidatos, pero no es la correcta
    assert candidates["no_exactas_con_la_correcta_o_a_un_caracter"] == 3  # 002, 003 y 004


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (("REPUBLICA", "AB123CD"), "exacta"),
        (("XXAB123CDXX",), "exacta"),  # la patente adentro de un texto más largo
        (("AB123CE",), "a_un_caracter"),
        (("AB12",), "no"),
        ((), "no"),
    ],
)
def test_aparicion_en_candidatos(candidates: tuple[str, ...], expected: str) -> None:
    assert mk.candidate_match("AB123CD", candidates) == expected


def test_que_tan_equivocada_es_una_lectura() -> None:
    assert mk.wrong_kind("AB123CD", "AB123CE") == "difiere en un carácter"
    assert (
        mk.wrong_kind("AB123CD", "AB1Z3CD") == "igual tras normalizar letra/dígito"
    )  # Z en posición de dígito
    assert mk.wrong_kind("AB123CD", "XY999ZW") == "patente distinta"
    assert mk.wrong_kind("AB123CD", "AB12") == "sin la estructura de una patente"


def test_parseq_con_la_regla_del_servicio_la_poca_confianza_es_sin_lectura(tmp_path: Path) -> None:
    parseq = mk.build_report(rows_of(tmp_path), 0.78)["parseq"]["total"]

    # 004 es exacta pero con confianza 0,5: ilegible; 006 se acepta y es otra patente: equivocada
    assert (parseq["lectura_exacta"], parseq["sin_lectura"], parseq["lectura_equivocada"]) == (5, 1, 1)
    assert parseq["texto_exacto_sin_umbral"] == 6  # el texto de 004 es exacto aunque no se acepte
    assert parseq["equivocada_detalle"] == {"patente distinta": 1}


def test_error_silencioso_simula_r3_con_la_lectura_equivocada_como_denunciada(tmp_path: Path) -> None:
    block = mk.build_report(rows_of(tmp_path), 0.78)["error_silencioso"]["total"]

    # 003: DEF458 contra DEF456 (un carácter, GREEN); 004: PARSeq con 0,5 (ilegible, GREEN);
    # 007: PQR999 contra PQR678 (RED)
    assert block["lecturas_equivocadas"] == 3
    assert block["patente_distinta_RED"] == 1 and block["difiere_en_un_caracter_GREEN"] == 1
    assert block["ilegible_GREEN"] == 1 and block["coincide_GREEN"] == 0
    assert block["patente_distinta_RED_pct"] == 33.3
    assert block["de_las_cuales_parseq_lee_la_correcta"] == 2  # 003 y 007; la 004 no llega al umbral


def test_el_error_silencioso_va_tambien_por_luz(tmp_path: Path) -> None:
    report = mk.build_report(rows_of(tmp_path), 0.78)["error_silencioso"]

    assert report["noche"]["lecturas_equivocadas"] == 1 and report["noche"]["ilegible_GREEN"] == 1
    assert report["dia"]["lecturas_equivocadas"] == 2 and report["dia"]["patente_distinta_RED"] == 1


def test_el_tiempo_usa_mediana_y_p95_por_posicion() -> None:
    rows = [
        mk.Row(str(i), "ABC123", "dia", "", (), None, float(i), "", 0.0) for i in range(1, 21)
    ]  # 1..20 ms

    result = mk.timing(rows)

    assert result["mediana_ms"] == 10.5 and result["p95_ms"] == 20.0  # índice floor(0,95 x 20) = 19
    assert result["minimo_ms"] == 1.0 and result["maximo_ms"] == 20.0
    assert mk.timing([]) == {"recortes": 0}


def test_u5_compara_la_lectura_exacta_con_85_de_dia_y_70_de_noche(tmp_path: Path) -> None:
    report = mk.build_report(rows_of(tmp_path), 0.78)

    assert report["u5"]["umbral_pct"] == {"dia": 85.0, "noche": 70.0}
    assert report["u5"]["mlkit"]["dia"] == {"lectura_exacta_pct": 40.0, "cumple": False}
    assert report["u5"]["parseq"]["dia"]["cumple"] is False  # 4 de 5 = 80 %
    assert report["u5"]["parseq"]["noche"] == {"lectura_exacta_pct": 50.0, "cumple": False}


def test_las_filas_y_el_informe_no_llevan_el_texto_de_ninguna_patente(tmp_path: Path) -> None:
    rows = rows_of(tmp_path)
    out = tmp_path / "rows.csv"

    mk.write_rows(out, rows, 0.78)

    content = out.read_text(encoding="utf-8")
    assert not any(text in content for _, text, *_ in CASES)
    assert "ABC123" not in json.dumps(mk.build_report(rows, 0.78))
    with out.open(encoding="utf-8", newline="") as handle:
        by_id = {row["id"]: row for row in csv.DictReader(handle)}
    assert by_id["007"]["r3"] == "PLATE_MISMATCH" and by_id["003"]["r3"] == "PLATE_FUZZY"
    assert by_id["002"]["mlkit"] == "sin_lectura" and by_id["002"]["candidato"] == "exacta"


def test_la_linea_de_comandos_escribe_el_informe_y_la_tabla(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, results, readings = make_files(tmp_path)
    out = tmp_path / "out"

    code = mk.main(["--labeled-set", str(set_dir), "--parseq-readings", str(readings), "--out-dir", str(out)])

    assert code == 0
    assert (out / "mlkit.json").is_file() and (out / "mlkit_rows.csv").is_file()
    printed = capsys.readouterr().out
    assert "lectura exacta ML Kit" in printed and "lectura exacta PARSeq" in printed
    assert "Error silencioso (total): 3 lecturas equivocadas" in printed


def test_se_niega_dentro_del_repo_y_falla_claro_si_faltan_archivos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, _, readings = make_files(tmp_path)

    assert (
        mk.main(
            [
                "--labeled-set",
                str(set_dir),
                "--parseq-readings",
                str(readings),
                "--out-dir",
                str(REPO_ROOT / "x"),
            ]
        )
        == 1
    )
    assert "dentro del repositorio" in capsys.readouterr().err
    assert (
        mk.main(
            [
                "--labeled-set",
                str(set_dir),
                "--parseq-readings",
                str(tmp_path / "no.csv"),
                "--out-dir",
                str(tmp_path / "o"),
            ]
        )
        == 1
    )
    assert "No se pudo cruzar" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        mk.parse_args(["--labeled-set", str(set_dir)])
