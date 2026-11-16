"""tools/eval/ocr_set.py --topup: una lectura por patente repetida y reposición hasta N legibles de auto."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from tests.tools.test_eval_ocr_set import TEXTS, ocr_set, photo, write_export

pytest.importorskip("openpyxl")
pytest.importorskip("PIL")

LABELS: list[dict[str, str]] = [
    {"texto": "AAA111", "formato": "anterior", "luz": "dia", "estado": "legible"},  # 001: repetida, se va
    {
        "texto": "AAA111",
        "formato": "anterior",
        "luz": "noche",
        "estado": "legible",
    },  # 002: repetida, se queda
    {"texto": "BB222CC", "formato": "mercosur", "luz": "dia", "estado": "legible"},  # 003
    {"texto": "", "formato": "anterior", "luz": "dia", "estado": "ilegible"},  # 004
    {
        "texto": "DDD333",
        "formato": "moto",
        "luz": "dia",
        "estado": "legible",
    },  # 005: legible, pero no de auto
    {"texto": "EEE444", "formato": "anterior", "luz": "dia", "estado": "legible"},  # 006
]


def labeled_set(tmp_path: Path, photos: int = 10) -> tuple[Path, Path, Path]:
    """Un set de 6 filas etiquetadas a mano (en el csv) con `photos - 6` candidatas más en el export."""
    src, reference, out = tmp_path / "src", tmp_path / "reference", tmp_path / "out"
    write_export(src, {f"p{n}.jpg": (photo(TEXTS[n], night=n < 2, seed=n), 1.0) for n in range(photos)})
    write_export(reference, {"r.jpg": (photo(TEXTS[11], seed=77), 1.0)})
    built = ocr_set.build_pool(src, reference)
    first = ocr_set.select(built.items, 6, 1, 5)
    ocr_set.build_outputs(src, [(f"{n:03d}", item) for n, item in enumerate(first, start=1)], out)
    rows = ocr_set.ocr_labels.read_csv_rows(out / "etiquetas.csv")
    for row, labels in zip(rows, LABELS, strict=True):
        row.update(labels)
    ocr_set.write_csv(out / "etiquetas.csv", ocr_set.LABEL_COLUMNS, rows)
    return src, reference, out


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def topup(src: Path, reference: Path, out: Path, count: int = 5, **kwargs: Any) -> dict[str, Any]:
    return ocr_set.topup(src, reference, out, count=count, seed=5, **kwargs)


def test_deja_una_lectura_por_patente_repetida_la_de_noche(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)

    summary = topup(src, reference, out)

    assert summary["repetidas_quitadas"] == 1
    rows = ocr_set.ocr_labels.read_label_rows(out)
    ids = [row["id"] for row in rows]
    assert "001" not in ids and "002" in ids
    assert [row["id"] for row in read(out / "repetidas.csv")] == ["001"]
    assert read(out / "repetidas.csv")[0]["conservada"] == "002"
    discards = [row for row in read(out / "descartes.csv") if row["motivo"] == "texto_repetido"]
    assert [row["detalle"] for row in discards] == ["se conserva el id 002"]


def test_repone_hasta_el_total_de_legibles_de_auto_con_ids_nuevos_y_hojas(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)

    summary = topup(src, reference, out, count=5)

    # quedan legibles de auto: 002, 003 y 006 (la 004 es ilegible y la 005 es de moto): faltan 2
    assert summary["legibles_de_auto_sin_repetidas"] == 3
    assert summary["faltan"] == 2 and summary["agregadas"] == 2 and summary["faltan_tras_reponer"] == 0
    assert summary["ids_nuevos"] == "007-008"
    rows = ocr_set.ocr_labels.read_label_rows(out)
    assert [row["id"] for row in rows] == ["002", "003", "004", "005", "006", "007", "008"]
    for new in rows[-2:]:
        assert not (new["texto"] or new["formato"] or new["estado"])  # sin etiquetar; la luz viene estimada
        assert new["luz"] in ("dia", "noche")
        assert (out / new["recorte"]).is_file()
    assert (out / "hojas" / "reposicion-007-01.jpg").is_file()
    assert len(read(out / "cajas.csv")) == 7


def test_las_filas_conservadas_mantienen_sus_etiquetas_y_el_xlsx_sale_con_ellas(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)

    topup(src, reference, out)

    kept = {row["id"]: row for row in ocr_set.ocr_labels.read_label_rows(out)}
    for number, labels in enumerate(LABELS, start=1):
        if number == 1:
            continue
        assert {key: kept[f"{number:03d}"][key] for key in labels} == labels
    xlsx = ocr_set.ocr_labels.read_xlsx_rows(out / "etiquetas.xlsx")
    assert sorted(xlsx) == sorted(kept)
    assert xlsx["003"]["texto"] == "BB222CC" and xlsx["007"]["estado"] == ""


def test_las_repuestas_no_son_ninguna_de_las_que_estaban_ni_la_repetida_quitada(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    before = {row["archivo"] for row in read(out / "etiquetas.csv")}

    topup(src, reference, out, count=7)  # faltan 4 y hay 4 candidatas

    new = [row["archivo"] for row in read(out / "etiquetas.csv") if row["id"] > "006"]
    assert len(new) == len(set(new)) == 4 and not set(new) & before


def test_si_el_export_se_agota_no_escribe_nada_y_lo_dice(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    before = (out / "etiquetas.csv").read_bytes()

    with pytest.raises(ValueError, match="el export se agotó: faltan 5 y hay 4"):
        topup(src, reference, out, count=8)

    assert (out / "etiquetas.csv").read_bytes() == before
    assert not out.with_name(out.name + "-v1").exists() and not (out / "repetidas.csv").exists()


def test_en_seco_informa_las_candidatas_que_hay_y_las_que_quedan(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)

    summary = topup(src, reference, out, count=5, dry_run=True)
    short = topup(src, reference, out, count=8, dry_run=True)

    assert summary["candidatas_en_el_export"] == 4 and summary["candidatas_que_quedan"] == 2
    assert summary["legibles_de_auto_de_noche"] == 1  # la 002, la que se conserva
    assert short["faltan"] == 5 and short["agregadas"] == 4 and short["faltan_tras_reponer"] == 1


def test_con_menos_patentes_de_las_pedidas_no_hace_falta_reponer_pero_igual_quita_repetidas(
    tmp_path: Path,
) -> None:
    src, reference, out = labeled_set(tmp_path)

    summary = topup(src, reference, out, count=3)

    assert summary["faltan"] == 0 and summary["agregadas"] == 0 and summary["repetidas_quitadas"] == 1
    assert [row["id"] for row in read(out / "etiquetas.csv")] == ["002", "003", "004", "005", "006"]


def test_sin_repetidas_ni_faltantes_no_toca_nada(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    rows = ocr_set.ocr_labels.read_csv_rows(out / "etiquetas.csv")
    rows[0]["texto"] = "ZZZ999"
    ocr_set.write_csv(out / "etiquetas.csv", ocr_set.LABEL_COLUMNS, rows)
    before = (out / "etiquetas.csv").read_bytes()

    summary = topup(src, reference, out, count=3)

    assert summary["agregadas"] == 0 and summary["repetidas_quitadas"] == 0
    assert (out / "etiquetas.csv").read_bytes() == before
    assert not out.with_name(out.name + "-v1").exists()


def test_en_seco_no_escribe_nada(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    before = (out / "etiquetas.csv").read_bytes()

    summary = topup(src, reference, out, dry_run=True)

    assert summary["agregadas"] == 2 and summary["ids_nuevos"] == "007-008"
    assert (out / "etiquetas.csv").read_bytes() == before
    assert not (out / "repetidas.csv").exists() and not (out / "etiquetas.xlsx").exists()
    assert not out.with_name(out.name + "-v1").exists()


def test_respalda_el_set_antes_de_escribir(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    before = (out / "etiquetas.csv").read_bytes()

    topup(src, reference, out)

    assert (out.with_name(out.name + "-v1") / "etiquetas.csv").read_bytes() == before


def test_una_segunda_pasada_con_todo_etiquetado_no_repite_las_quitadas(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)
    topup(src, reference, out, count=5)
    removed_file = read(out.with_name(out.name + "-v1") / "etiquetas.csv")[0]["archivo"]
    rows = ocr_set.ocr_labels.read_label_rows(out)
    texts = iter(["FFF555", "GGG666"])
    for row in rows[-2:]:
        row.update({"texto": next(texts), "formato": "anterior", "luz": "dia", "estado": "legible"})
    ocr_set.write_csv(out / "etiquetas.csv", ocr_set.LABEL_COLUMNS, rows)
    (out / "etiquetas.xlsx").unlink()  # las etiquetas nuevas están en el csv

    summary = topup(src, reference, out, count=7)

    assert summary["legibles_de_auto_sin_repetidas"] == 5
    assert summary["faltan"] == 2
    assert removed_file not in {row["archivo"] for row in read(out / "etiquetas.csv") if row["id"] > "008"}
    assert [row["id"] for row in read(out / "etiquetas.csv")][-2:] == ["009", "010"]
    assert (
        len(json.loads((out / "informe.json").read_text(encoding="utf-8"))["reposiciones"]) == 2
    )  # una por pasada


def test_no_se_mezclan_los_ids_de_una_reposicion_con_los_de_una_fila_quitada(tmp_path: Path) -> None:
    src, reference, out = labeled_set(tmp_path)

    topup(src, reference, out)

    assert "001" not in {row["id"] for row in read(out / "etiquetas.csv")}  # el id quitado no se reutiliza


def test_la_referencia_vacia_falla_en_vez_de_dejar_pasar_las_fotos_de_calibracion(tmp_path: Path) -> None:
    src, _, out = labeled_set(tmp_path)

    with pytest.raises(ValueError, match="dataset de calibración"):
        topup(src, tmp_path / "no-existe", out)

    assert not out.with_name(out.name + "-v1").exists()


def test_repeated_texts_conserva_la_de_noche_despues_la_de_mayor_recorte_despues_el_menor_id() -> None:
    rows = [
        {"id": "001", "texto": "AAA111", "estado": "legible", "luz": "dia"},
        {"id": "002", "texto": "AAA111", "estado": "legible", "luz": "dia"},
        {"id": "003", "texto": "AAA111", "estado": "legible", "luz": "dia"},
        {"id": "004", "texto": "BBB222", "estado": "legible", "luz": "dia"},
        {"id": "005", "texto": "BBB222", "estado": "legible", "luz": "noche"},
        {"id": "006", "texto": "CCC333", "estado": "ilegible", "luz": "dia"},
        {"id": "007", "texto": "CCC333", "estado": "ilegible", "luz": "dia"},
        {"id": "008", "texto": "", "estado": "", "luz": "dia"},
        {"id": "009", "texto": "", "estado": "", "luz": "dia"},
    ]
    areas = {"001": 10, "002": 50, "003": 50}

    pairs = ocr_set.ocr_labels.repeated_texts(rows, lambda row: areas.get(row["id"], 1))

    assert pairs == [("001", "002"), ("003", "002"), ("004", "005")]  # solo las legibles con texto


def test_near_texts_marca_los_pares_a_un_caracter_sin_mostrar_el_texto() -> None:
    rows = [
        {"id": "001", "texto": "ABC123", "estado": "legible", "formato": "anterior"},
        {"id": "002", "texto": "ABC128", "estado": "legible", "formato": "anterior"},
        {"id": "003", "texto": "XYZ999", "estado": "legible", "formato": "anterior"},
        {"id": "004", "texto": "ABC124", "estado": "legible", "formato": "moto"},  # no es de auto
    ]

    assert ocr_set.ocr_labels.near_texts(rows) == [("001", "002")]


def test_la_linea_de_comandos_acepta_topup_en_seco_y_no_se_combina_con_refresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, reference, out = labeled_set(tmp_path)

    code = ocr_set.main(
        [
            "--topup",
            "--dry-run",
            "--src",
            str(src),
            "--reference",
            str(reference),
            "--out",
            str(out),
            "--n",
            "5",
        ]
    )

    assert code == 0 and '"agregadas": 2' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ocr_set.parse_args(["--topup", "--refresh", "--src", "a", "--reference", "b", "--out", "c"])
    with pytest.raises(SystemExit):
        ocr_set.parse_args(["--dry-run", "--src", "a", "--reference", "b", "--out", "c"])
