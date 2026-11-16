"""tools/eval/ocr.py sobre el set etiquetado: `en_alcance` (U2 con cota, por luz) y exportación sin texto."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.tools.test_eval_ocr_labels import REPO_ROOT, FixedReader, labeled_set, ocr, ocr_labels
from vvi_ia.domain.model import Box

pytest.importorskip("openpyxl")


def reading(label: str, text: str, conf: float, light: str, fmt: str = "mercosur") -> object:
    sample = ocr.Sample("train", "f.jpg", Box(0, 0, 10, 10), label, format=fmt, light=light)
    return ocr.Reading(sample, text, conf, failed=False)


def test_u2_cuenta_las_patentes_correctas_marcadas_como_distintas_con_su_cota() -> None:
    readings = [
        reading("AB123CD", "AB123CD", 0.95, "dia"),
        reading("AB123CE", "XY999ZW", 0.95, "dia"),  # lectura segura de otra patente: MISMATCH
        reading("AB123CF", "XY999ZW", 0.30, "dia"),  # poco segura: ilegible, no marca
    ]

    result = ocr.u2(readings, 0.78)  # type: ignore[arg-type]

    assert (result["marcadas"], result["lecturas"], result["tasa"]) == (1, 3, 0.3333)
    assert result["cota_95"] is not None and result["cota_95"] > 0.3333


def test_sin_marcadas_la_cota_es_la_regla_del_3() -> None:
    readings = [reading("AB123CD", "AB123CD", 0.95, "dia") for _ in range(30)]

    result = ocr.u2(readings, 0.78)  # type: ignore[arg-type]

    assert (result["marcadas"], result["lecturas"], result["cota_95"]) == (0, 30, 0.1)


def test_el_bloque_en_alcance_da_lectura_exacta_distribucion_y_u2_en_total_y_por_luz() -> None:
    readings = [
        reading("AB123CD", "AB123CD", 0.95, "dia"),  # MATCH y exacta
        reading("AB123CE", "AB123CD", 0.95, "dia"),  # FUZZY (un carácter)
        reading("AB123CF", "AB123CF", 0.40, "noche"),  # ilegible por confianza (con texto exacto)
        reading("AB123CG", "XY999ZW", 0.95, "noche"),  # patente distinta
        reading("ABC123", "AB", 0.95, "noche", fmt="moto"),  # de moto: fuera de alcance
    ]

    block = ocr.scope_report(
        [r for r in readings if ocr.sample_format(r.sample) in ocr.IN_SCOPE_FORMATS],  # type: ignore[attr-defined]
        0.78,
    )

    assert block["lecturas"] == 4 and block["coincidencia_exacta"] == 2
    assert block["distribucion"] == {
        "PLATE_MATCH": 1,
        "PLATE_FUZZY": 1,
        "PLATE_MISMATCH": 1,
        "PLATE_UNREADABLE": 1,
    }
    assert block["U2"]["marcadas"] == 1
    assert (
        block["por_luz"]["dia"]["coincidencia_exacta"] == 1 and block["por_luz"]["dia"]["U2"]["marcadas"] == 0
    )
    assert block["por_luz"]["noche"]["lecturas"] == 2 and block["por_luz"]["noche"]["U2"]["marcadas"] == 1


def test_sin_luz_etiquetada_no_hay_corte_por_luz() -> None:
    block = ocr.scope_report([reading("AB123CD", "AB123CD", 0.95, "")], 0.78)  # type: ignore[arg-type]

    assert "por_luz" not in block


def test_el_informe_del_set_etiquetado_trae_en_alcance_sin_las_motos(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    items, _ = ocr.read_samples(loaded.samples, photos, FixedReader("AB123CD"))

    report = ocr.build_report([r for r, _ in items], 0.78, "fake", loaded.discarded, 4, loaded.unreadable)

    scope = report["en_alcance"]
    assert scope["lecturas"] == 2  # las dos de auto; la moto etiquetada queda afuera
    assert scope["coincidencia_exacta"] == 1
    assert scope["U2"]["marcadas"] == 1 and scope["U2"]["lecturas"] == 2  # AB999ZZ leída como AB123CD
    assert scope["por_luz"]["dia"]["U2"]["marcadas"] == 0 and scope["por_luz"]["noche"]["U2"]["marcadas"] == 1
    assert report["pares_cruzados"]["todos"]["pares"] > 0  # S5 sigue en el informe


# --- exportación para otro lector ----------------------------------------------------------------


def exported(out: Path) -> list[str]:
    return sorted(path.name for path in out.iterdir())


def test_exporta_solo_las_legibles_de_auto_con_su_id_y_sin_el_texto(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    out = tmp_path / "mlkit"

    result = ocr.export_crops(loaded.samples, photos, out, set_dir)

    assert result == {"exportados": 2, "salteados": 0}
    assert exported(out) == ["001.png", "002.png", "indice.csv"]  # ni la ilegible (003) ni la moto (004)
    with (out / "indice.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"id": "001", "formato": "mercosur", "luz": "dia"},
        {"id": "002", "formato": "mercosur", "luz": "noche"},
    ]
    for path in out.iterdir():  # el texto de las etiquetas no está en ningún archivo
        assert b"AB123CD" not in path.read_bytes() and b"AB999ZZ" not in path.read_bytes()


def test_el_recorte_exportado_es_el_que_lee_el_ocr_la_caja_sin_margen(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    out = tmp_path / "mlkit"

    ocr.export_crops(loaded.samples, photos, out, set_dir)

    crop = cv2.imdecode(np.fromfile(str(out / "001.png"), np.uint8), cv2.IMREAD_COLOR)
    photo = cv2.imdecode(np.fromfile(str(photos / "train" / "p1.jpg"), np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape == (50, 140, 3)  # la caja anotada (60, 50, 140, 50), sin margen
    assert np.array_equal(crop, photo[50:100, 60:200])


def test_con_recorte_set_copia_el_recorte_con_margen_del_set(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    out = tmp_path / "mlkit"

    ocr.export_crops(loaded.samples, photos, out, set_dir, crop="set")

    assert (out / "001.png").read_bytes() == (set_dir / "recortes" / "001.png").read_bytes()
    with pytest.raises(ValueError, match="recorte desconocido"):
        ocr.export_crops(loaded.samples, photos, out, set_dir, crop="otro")


def test_una_nueva_exportacion_reemplaza_lo_que_habia(tmp_path: Path) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    out = tmp_path / "mlkit"
    out.mkdir()
    (out / "999.png").write_bytes(b"viejo")

    ocr.export_crops(loaded.samples, photos, out, set_dir)

    assert exported(out) == ["001.png", "002.png", "indice.csv"]


def test_la_linea_de_comandos_exporta_sin_modelos_ni_carpeta_de_salida(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, photos = labeled_set(tmp_path)
    out = tmp_path / "mlkit"

    code = ocr.main(
        ["--data-dir", str(tmp_path), "--labeled-set", str(set_dir), "--photos-dir", str(photos)]
        + ["--export-mlkit", str(out)]
    )

    assert code == 0 and exported(out) == ["001.png", "002.png", "indice.csv"]
    assert "Exportados 2 recortes" in capsys.readouterr().out


def test_la_exportacion_falla_si_faltan_etiquetas_y_no_se_hace_dentro_del_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_dir, photos = labeled_set(tmp_path)
    rows = ocr_labels.read_csv_rows(set_dir / "etiquetas.csv")
    rows[0].update(texto="AB123CD", estado="legible", formato="mercosur")
    rows[1]["estado"] = ""  # sin etiquetar, y el xlsx no existe más
    (set_dir / "etiquetas.xlsx").unlink()
    with (set_dir / "etiquetas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    code = ocr.main(
        ["--data-dir", str(tmp_path), "--labeled-set", str(set_dir), "--photos-dir", str(photos)]
        + ["--export-mlkit", str(tmp_path / "mlkit")]
    )
    assert code == 1 and "faltan etiquetas" in capsys.readouterr().err

    code = ocr.main(
        ["--data-dir", str(tmp_path), "--labeled-set", str(set_dir), "--photos-dir", str(photos)]
        + ["--export-mlkit", str(REPO_ROOT / "mlkit-de-prueba")]
    )
    assert code == 1 and "dentro del repositorio" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        ocr.parse_args(["--data-dir", str(tmp_path), "--export-mlkit", str(tmp_path / "x")])


def test_el_cache_de_fotos_es_acotado_y_no_cambia_las_lecturas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_dir, photos = labeled_set(tmp_path)
    loaded = ocr.load_set_samples(set_dir)
    samples = [*loaded.samples, loaded.samples[0], *loaded.samples]  # la foto 1 vuelve después de otras
    full, _ = ocr.read_samples(samples, photos, FixedReader())

    monkeypatch.setattr(ocr, "IMAGE_CACHE", 1)
    small, skipped = ocr.read_samples(samples, photos, FixedReader())

    assert not skipped and len(small) == len(full) == 7
    assert [(r.sample.ident, r.text, r.conf) for r, _ in small] == [
        (r.sample.ident, r.text, r.conf) for r, _ in full
    ]
