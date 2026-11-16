"""tools/eval/ocr_mlkit.py: ML Kit contra PARSeq sobre el set etiquetado (datos sintéticos, sin modelos)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


def report_of(tmp_path: Path, min_conf: float = 0.78) -> dict[str, Any]:
    return mk.build_report({"v1": rows_of(tmp_path)}, min_conf)


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
    report = report_of(tmp_path)

    total, day, night = (report["mlkit"]["v1"][name] for name in ("total", "dia", "noche"))
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
    candidates = report_of(tmp_path)["mlkit"]["v1"]["total"]["candidatos"]

    # no exactas: 002 (sin lectura, la correcta aparece), 003 (equivocada, aparece), 004, 005 (sin
    # candidatos) y 007
    assert candidates["no_exactas"] == 5 and candidates["no_exactas_con_la_correcta"] == 2
    assert candidates["sin_lectura_con_la_correcta"] == 1 and candidates["equivocadas_con_la_correcta"] == 1
    assert candidates["la_correcta_aparece"] == 4  # las 2 exactas más 002 y 003
    assert candidates["sin_candidatos"] == 1
    assert (
        candidates["la_correcta_aparece_normalizada"] == 4
    )  # ningún candidato de este caso necesita normalizar
    # 004 (GH789IK contra GH789IJ) está a un carácter en los candidatos, pero no es la correcta
    assert candidates["no_exactas_con_la_correcta_o_a_un_caracter"] == 3  # 002, 003 y 004


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (("REPUBLICA", "AB123CD"), "exacta"),
        (("XXAB123CDXX",), "exacta"),  # la patente adentro de un texto más largo
        (("AB123CE",), "a_un_caracter"),
        (("A8123CD",), "normalizada"),  # el 8 en posición de letra se lee B
        (("ABI23CD",), "normalizada"),  # la I en posición de dígito se lee 1
        (("ZZA8123CDZZ",), "normalizada"),  # adentro de un texto más largo
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
    parseq = report_of(tmp_path)["parseq"]["total"]

    # 004 es exacta pero con confianza 0,5: ilegible; 006 se acepta y es otra patente: equivocada
    assert (parseq["lectura_exacta"], parseq["sin_lectura"], parseq["lectura_equivocada"]) == (5, 1, 1)
    assert parseq["texto_exacto_sin_umbral"] == 6  # el texto de 004 es exacto aunque no se acepte
    assert parseq["equivocada_detalle"] == {"patente distinta": 1}


def test_error_silencioso_simula_r3_con_la_lectura_equivocada_como_denunciada(tmp_path: Path) -> None:
    block = report_of(tmp_path)["error_silencioso"]["v1"]["total"]

    # 003: DEF458 contra DEF456 (un carácter, GREEN); 004: PARSeq con 0,5 (ilegible, GREEN);
    # 007: PQR999 contra PQR678 (RED)
    assert block["lecturas_equivocadas"] == 3
    assert block["patente_distinta_RED"] == 1 and block["difiere_en_un_caracter_GREEN"] == 1
    assert block["ilegible_GREEN"] == 1 and block["coincide_GREEN"] == 0
    assert block["patente_distinta_RED_pct"] == 33.3
    assert block["de_las_cuales_parseq_lee_la_correcta"] == 2  # 003 y 007; la 004 no llega al umbral


def test_el_error_silencioso_va_tambien_por_luz(tmp_path: Path) -> None:
    report = report_of(tmp_path)["error_silencioso"]["v1"]

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
    report = report_of(tmp_path)

    assert report["u5"]["umbral_pct"] == {"dia": 85.0, "noche": 70.0}
    assert report["u5"]["mlkit"]["v1"]["dia"] == {"lectura_exacta_pct": 40.0, "cumple": False}
    assert report["u5"]["parseq"]["dia"]["cumple"] is False  # 4 de 5 = 80 %
    assert report["u5"]["parseq"]["noche"] == {"lectura_exacta_pct": 50.0, "cumple": False}


def test_las_filas_y_el_informe_no_llevan_el_texto_de_ninguna_patente(tmp_path: Path) -> None:
    rows = rows_of(tmp_path)
    out = tmp_path / "rows.csv"

    mk.write_rows(out, {"v1": rows}, 0.78)

    content = out.read_text(encoding="utf-8")
    assert not any(text in content for _, text, *_ in CASES)
    assert "ABC123" not in json.dumps(mk.build_report({"v1": rows}, 0.78))
    with out.open(encoding="utf-8", newline="") as handle:
        by_id = {row["id"]: row for row in csv.DictReader(handle)}
    assert by_id["007"]["r3_v1"] == "PLATE_MISMATCH" and by_id["003"]["r3_v1"] == "PLATE_FUZZY"
    assert by_id["002"]["mlkit_v1"] == "sin_lectura" and by_id["002"]["candidato"] == "exacta"


def test_la_linea_de_comandos_escribe_el_informe_y_la_tabla(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, results, readings = make_files(tmp_path)
    out = tmp_path / "out"

    code = mk.main(["--labeled-set", str(set_dir), "--parseq-readings", str(readings), "--out-dir", str(out)])

    assert code == 0
    assert (out / "mlkit.json").is_file() and (out / "mlkit_rows.csv").is_file()
    printed = capsys.readouterr().out
    assert "ML Kit v1" in printed and "PARSeq" in printed and "R3: patente distinta (RED)" in printed
    assert "Tiempo por recorte (v1)" in printed


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


# --- dos versiones de ML Kit ------------------------------------------------------------------------------

# Cambios de v2: 002 ahora lee "AB123CD" (exacta) y 005 lee "KLM013" (equivocada, un carácter).
V2_READINGS = {"002": "AB123CD", "005": "KLM013"}


def write_v2(set_dir: Path, readings: dict[str, str] | None = None, extra_column: bool = True) -> Path:
    readings = V2_READINGS if readings is None else readings
    path = set_dir / "resultados" / "ocr-results-v2.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "formato",
                "luz",
                "lectura",
                "candidatos",
                "confianza",
                "tiempo_ms",
                *(["linea_elegida"] if extra_column else []),
            ]
        )
        for ident, _, fmt, light, (reading, candidates, conf, ms), _ in CASES:
            writer.writerow(
                [
                    ident,
                    fmt,
                    light,
                    readings.get(ident, reading),
                    candidates,
                    conf,
                    str(float(ms) + 1),
                    *(["x"] if extra_column else []),
                ]
            )
    return path


def two_versions(tmp_path: Path) -> dict[str, list[mk.Row]]:
    set_dir, results, readings = make_files(tmp_path)
    v2 = write_v2(set_dir)
    parseq = mk.read_parseq(readings)
    return {
        "v1": mk.build_rows(set_dir, mk.read_results(results), parseq),
        "v2": mk.build_rows(set_dir, mk.read_results(v2), parseq),
    }


def test_el_informe_trae_cada_version_de_ml_kit_y_parseq_lado_a_lado(tmp_path: Path) -> None:
    report = mk.build_report(two_versions(tmp_path), 0.78)

    assert report["versiones"] == ["v1", "v2"] and report["candidatos_iguales_entre_versiones"] is True
    v1, v2 = report["mlkit"]["v1"]["total"], report["mlkit"]["v2"]["total"]
    assert (v1["lectura_exacta"], v1["sin_lectura"], v1["lectura_equivocada"]) == (2, 2, 3)
    assert (v2["lectura_exacta"], v2["sin_lectura"], v2["lectura_equivocada"]) == (3, 0, 4)
    assert report["parseq"]["total"]["lectura_exacta"] == 5  # no depende de la versión
    assert report["tiempo"]["v2"]["mediana_ms"] == report["tiempo"]["v1"]["mediana_ms"] + 1
    assert report["u5"]["mlkit"]["v2"]["dia"]["lectura_exacta_pct"] == 40.0


def test_el_r3_se_simula_por_version(tmp_path: Path) -> None:
    report = mk.build_report(two_versions(tmp_path), 0.78)["error_silencioso"]

    assert report["v1"]["total"]["lecturas_equivocadas"] == 3
    # v2 suma 005 (KLM013 contra la lectura de PARSeq KLM012: un carácter, GREEN)
    assert report["v2"]["total"]["lecturas_equivocadas"] == 4
    assert report["v2"]["total"]["difiere_en_un_caracter_GREEN"] == 2
    assert report["v2"]["total"]["patente_distinta_RED"] == 1


def test_las_lecturas_nuevas_de_v2_se_desglosan_en_exactas_y_equivocadas(tmp_path: Path) -> None:
    delta = mk.build_report(two_versions(tmp_path), 0.78)["lecturas_nuevas"]["v2_contra_v1"]

    assert delta["lecturas_nuevas"] == 2 and delta["exactas"] == 1 and delta["equivocadas"] == 1
    assert delta["equivocada_detalle"] == {"difiere en un carácter": 1}
    assert delta["r3_de_las_equivocadas"]["difiere_en_un_caracter_GREEN"] == 1
    assert delta["perdidas"] == 0 and delta["cambiadas"] == 0
    assert delta["por_luz"]["dia"]["lecturas_nuevas"] == 1  # 005 es de día
    assert delta["por_luz"]["noche"]["exactas"] == 1  # 002 es de noche


def test_las_lecturas_perdidas_y_cambiadas_se_cuentan_aparte(tmp_path: Path) -> None:
    set_dir, results, readings = make_files(tmp_path)
    v2 = write_v2(
        set_dir, {"001": "", "003": "DEF459"}
    )  # 001 pierde la lectura y 003 cambia de equivocada a otra
    parseq = mk.read_parseq(readings)
    versions = {
        "v1": mk.build_rows(set_dir, mk.read_results(results), parseq),
        "v2": mk.build_rows(set_dir, mk.read_results(v2), parseq),
    }

    delta = mk.new_readings(versions["v1"], versions["v2"], 0.78)

    assert delta["lecturas_nuevas"] == 0 and delta["perdidas"] == 1 and delta["cambiadas"] == 1


def test_si_los_candidatos_difieren_entre_versiones_el_informe_lo_dice(tmp_path: Path) -> None:
    versions = two_versions(tmp_path)
    changed = versions["v2"][0]
    versions["v2"][0] = mk.Row(
        changed.ident,
        changed.label,
        changed.light,
        changed.mlkit,
        ("OTRO",),
        changed.confidence,
        changed.time_ms,
        changed.parseq,
        changed.parseq_conf,
    )

    assert mk.build_report(versions, 0.78)["candidatos_iguales_entre_versiones"] is False


def test_el_csv_de_filas_trae_las_columnas_de_cada_version_sin_texto(tmp_path: Path) -> None:
    versions = two_versions(tmp_path)
    out = tmp_path / "rows.csv"

    mk.write_rows(out, versions, 0.78)

    with out.open(encoding="utf-8", newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert rows["002"]["mlkit_v1"] == "sin_lectura" and rows["002"]["mlkit_v2"] == "exacta"
    assert rows["005"]["mlkit_v2"] == "equivocada" and rows["005"]["r3_v2"] == "PLATE_FUZZY"
    assert "KLM013" not in out.read_text(encoding="utf-8")


def test_results_acepta_nombre_ruta_y_solo_la_ruta(tmp_path: Path) -> None:
    default = mk.parse_results(None, tmp_path)
    named = mk.parse_results(["v1=a.csv", "v2=D:\\x\\b.csv"], tmp_path)
    stem = mk.parse_results(["resultados/ocr-results-v2.csv"], tmp_path)

    assert list(default) == ["v1"] and default["v1"] == tmp_path / "resultados" / "ocr-results.csv"
    assert list(named) == ["v1", "v2"] and named["v2"] == Path("D:\\x\\b.csv")
    assert list(stem) == ["ocr-results-v2"]
    with pytest.raises(mk.ComparisonError, match="repetida"):
        mk.parse_results(["v1=a.csv", "v1=b.csv"], tmp_path)


def test_la_linea_de_comandos_compara_dos_versiones_y_acepta_la_columna_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, results, readings = make_files(tmp_path)
    v2 = write_v2(set_dir)
    out = tmp_path / "out"

    code = mk.main(
        ["--labeled-set", str(set_dir), "--parseq-readings", str(readings), "--out-dir", str(out)]
        + ["--results", f"v1={results}", f"v2={v2}"]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert (
        "ML Kit v1" in printed and "ML Kit v2" in printed and "Lecturas nuevas (v2_contra_v1): 2" in printed
    )
    assert "Candidatos iguales entre versiones: True" in printed
    report = json.loads((out / "mlkit.json").read_text(encoding="utf-8"))
    assert report["versiones"] == ["v1", "v2"] and "KLM013" not in json.dumps(report)
