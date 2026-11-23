"""tools/eval/redaction_set.py: cuántas veces la redacción tapa la patente denunciada (modelos falsos)."""

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

from tests.fakes import REDACTION, THRESHOLDS
from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.null import (
    NullCarSegmenter,
    NullCurbSegmenter,
    NullDamageDetector,
    NullEmbedder,
    NullFaceDetector,
)
from vvi_ia.domain.model import Box, BoxLabel, Pixels
from vvi_ia.domain.redaction import RedactionThresholds
from vvi_ia.service.validate_evidence import Models, ValidateEvidence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_redaction_set", REPO_ROOT / "tools" / "eval" / "redaction_set.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_redaction_set"] = module
    spec.loader.exec_module(module)
    return module


rs = load()
from ocr import Sample  # noqa: E402

REDACT = RedactionThresholds(plate_min_conf=0.10)
TARGET = Box(100, 80, 200, 120)  # la caja anotada (la patente denunciada): 100 x 40
PIECE = Box(
    120, 85, 190, 115, 0.6, BoxLabel.PLATE
)  # un pedazo de la misma patente que el mosaico detecta de nuevo
FULL = Box(100, 80, 200, 120, 0.9, BoxLabel.PLATE)  # la patente completa
CAR = Box(0, 0, 300, 200, 0.95, BoxLabel.CAR)  # el vehículo dominante de la foto
STRAY = Box(10, 10, 70, 40, 0.5, BoxLabel.PLATE)  # la de otro auto
REPORTED = "AB123CD"


class FakeDetector:
    version = "fake-detector"

    def __init__(self, plates: list[Box] | None) -> None:
        self.plates = plates

    def detect(self, img: Pixels) -> list[Box] | None:
        return self.plates

    def detect_plates(self, img: Pixels) -> list[Box] | None:
        return self.plates


class FakeReader:
    """Lee la patente completa como la denunciada; cualquier otro recorte, como otra patente."""

    version = "fake-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop: Pixels) -> tuple[str, float] | None:
        self.calls += 1
        return (REPORTED, 0.95) if crop.shape[:2] == (40, 100) else ("ZZ999ZZ", 0.95)


def use_case(plates: list[Box] | None, reader: FakeReader | None = None) -> ValidateEvidence:
    store = rs.MemoryStore()
    models = Models(
        FakeDetector(plates),
        NullCarSegmenter(),
        NullCurbSegmenter(),
        NullEmbedder(),
        reader or FakeReader(),
        NullDamageDetector(),
        NullFaceDetector(),
    )
    return ValidateEvidence(
        store=store,
        writer=store,
        decryptor=rs.PassthroughDecryptor(),
        decoder=CvImageDecoder(),
        renderer=JpegRenderer(),
        models=models,
        thresholds=THRESHOLDS,
        redaction=REDACTION,
    )


def sample(ident: str = "001", label: str = REPORTED, light: str = "dia", file: str = "p1.jpg") -> Sample:
    return Sample("train", file, TARGET, label, format="mercosur", light=light, ident=ident)


def photos(tmp_path: Path, *names: str) -> Path:
    folder = tmp_path / "photos"
    (folder / "train").mkdir(parents=True)
    for name in names:
        image = np.full((200, 300, 3), 120, np.uint8)
        assert cv2.imwrite(str(folder / "train" / name), image)
    return folder


def measure(
    tmp_path: Path,
    plates: list[Box] | None,
    samples: list[Sample],
    annotated: dict[tuple[str, str], list[Box]] | None = None,
) -> list[Any]:
    names = sorted({s.file for s in samples})
    result, _ = rs.measure_samples(
        use_case(plates), samples, photos(tmp_path, *names), 1024, REDACT, annotated
    )
    return list(result)


# --- cobertura -------------------------------------------------------------------------------------


def test_cobertura_es_la_fraccion_del_area_de_la_caja_anotada_bajo_el_relleno() -> None:
    assert rs.coverage(TARGET, []) == 0.0
    assert rs.coverage(TARGET, [Box(0, 0, 400, 400)]) == 1.0
    assert rs.coverage(TARGET, [Box(100, 80, 150, 120)]) == pytest.approx(0.5)
    assert rs.coverage(TARGET, [Box(100, 80, 150, 120), Box(125, 80, 175, 120)]) == pytest.approx(
        0.75
    )  # unión
    assert rs.coverage(TARGET, [Box(300, 300, 400, 400)]) == 0.0
    assert rs.coverage(Box(10, 10, 10, 50), [Box(0, 0, 100, 100)]) == 0.0  # caja sin área


@pytest.mark.parametrize(
    ("fraction", "kind"),
    [
        (0.0, "ninguna"),
        (0.019, "ninguna"),
        (0.02, "parcial"),
        (0.5, "parcial"),
        (0.949, "parcial"),
        (0.95, "total"),
    ],
)
def test_clase_de_la_tapada(fraction: float, kind: str) -> None:
    assert rs.classify(fraction) == kind


# --- medición -------------------------------------------------------------------------------------


def test_un_pedazo_contenido_en_la_exceptuada_no_tapa_ni_antes_ni_despues(tmp_path: Path) -> None:
    [one] = measure(tmp_path, [FULL, PIECE], [sample()])

    assert one.plates == 2 and one.exempt == 1
    assert one.before == one.after == 0.0  # la regla de las contenidas ya estaba en el "antes"


def test_la_principal_ya_esta_en_el_antes_y_no_se_tapa_aunque_el_ocr_no_la_lea(tmp_path: Path) -> None:
    [one] = measure(tmp_path, [FULL, STRAY], [sample(label="XY987ZW")])  # el OCR lee otra: sin exención

    assert one.exempt == 0
    assert one.before == one.after == 0.0  # la principal (la más grande) es la denunciada
    assert one.has_main and one.main_is_target and not one.main_read_as_target


def test_si_la_principal_no_es_la_denunciada_esta_queda_tapada_y_se_cuenta(tmp_path: Path) -> None:
    big_other = Box(10, 10, 90, 70, 0.9, BoxLabel.PLATE)  # la más grande de la foto, otra patente
    [one] = measure(tmp_path, [FULL, big_other], [sample(label="XY987ZW")])

    assert one.has_main and not one.main_is_target
    assert rs.classify(one.before) == "total" and rs.classify(one.after) == "total"
    summary = rs.principal_summary([one])
    assert summary["no_es_la_denunciada"] == 1
    assert summary["no_es_la_denunciada_y_la_denunciada_queda_tapada_total"] == 1


def test_con_el_vehiculo_la_principal_es_la_de_mayor_confianza_adentro(tmp_path: Path) -> None:
    more_confident = Box(5, 5, 90, 50, 0.99, BoxLabel.PLATE)  # dentro del auto y más confiable que FULL (0,9)
    [one] = measure(tmp_path, [CAR, FULL, more_confident], [sample()])

    assert (
        one.has_main and not one.main_is_target
    )  # la principal es la otra, aunque la denunciada sea la mayor
    assert one.exempt == 1 and one.after == one.before == 0.0  # la denunciada sigue exceptuada por lectura


def test_una_caja_que_contiene_entera_a_la_exceptuada_se_tapa_antes_y_no_despues(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)  # la principal es otra (más confiable, dentro del auto)
    bigger = Box(60, 40, 240, 160, 0.2, BoxLabel.PLATE)  # la patente y lo que la rodea
    [one] = measure(tmp_path, [CAR, FULL, main, bigger], [sample()])

    assert rs.classify(one.before) == "total" and one.after == 0.0
    assert one.covers == ()


def test_el_margen_de_una_vecina_tapa_un_poco_antes_y_nada_despues(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    neighbour = Box(201, 80, 260, 120, 0.8, BoxLabel.PLATE)  # no la toca: solo llega con su margen del 10 %
    [one] = measure(tmp_path, [CAR, FULL, main, neighbour], [sample()])

    assert 0 < one.before < 0.1 and one.after == 0.0


def test_las_cajas_que_siguen_cubriendo_la_denunciada_se_describen_sin_texto(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    holds_90 = Box(
        110, 40, 240, 160, 0.2, BoxLabel.PLATE
    )  # contiene el 90 % de la exceptuada: no es "entera"
    overlapping = Box(150, 90, 260, 130, 0.8, BoxLabel.PLATE)  # la pisa por un costado
    neighbour = Box(202, 10, 260, 30, 0.8, BoxLabel.PLATE)  # solo su margen llegaba: ya no
    [one] = measure(tmp_path, [CAR, FULL, main, holds_90, overlapping, neighbour], [sample()])

    kinds = {cover["tipo"]: cover for cover in one.covers}
    assert set(kinds) == {rs.BIGGER, rs.OVERLAP}
    assert kinds[rs.BIGGER]["confianza"] == 0.2
    assert 0 < kinds[rs.OVERLAP]["cubre_de_la_denunciada"] <= 1
    assert REPORTED not in json.dumps(one.covers)


def test_si_la_caja_grande_contenia_otra_patente_sin_deteccion_propia_queda_expuesta(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    holds_both = Box(90, 60, 270, 140, 0.2, BoxLabel.PLATE)  # la exceptuada y otra patente al lado
    other = Box(210, 85, 250, 115)  # anotada, sin caja propia del detector
    annotated = {("train", "p1.jpg"): [TARGET, other]}

    [one] = measure(tmp_path, [CAR, FULL, main, holds_both], [sample()], annotated)

    [(key, before, after)] = one.others  # la denunciada no cuenta como otra
    assert before == 1.0 and after == 0.0 and key.startswith("train/p1.jpg#")
    summary = rs.others_summary([one])
    assert summary["patentes_ajenas_anotadas"] == 1 and summary["destapadas_por_las_reglas"] == 1
    assert summary["tapadas_por_completo_antes"] == 1 and summary["tapadas_por_completo_despues"] == 0


def test_si_la_otra_patente_tiene_caja_propia_no_queda_expuesta(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    holds_both = Box(90, 60, 270, 140, 0.2, BoxLabel.PLATE)
    other_box = Box(210, 85, 250, 115, 0.7, BoxLabel.PLATE)  # el detector sí la ve
    annotated = {("train", "p1.jpg"): [TARGET, other_box]}

    [one] = measure(tmp_path, [CAR, FULL, main, holds_both, other_box], [sample()], annotated)

    summary = rs.others_summary([one])
    assert summary["patentes_ajenas_anotadas"] == 1 and summary["destapadas_por_las_reglas"] == 0
    assert summary["menos_tapadas"] == 0


def test_la_misma_patente_con_otra_caja_no_cuenta_como_ajena(tmp_path: Path) -> None:
    same_plate = Box(102, 82, 198, 118)  # casi la misma caja anotada
    annotated = {("train", "p1.jpg"): [TARGET, same_plate]}

    [one] = measure(tmp_path, [CAR, FULL], [sample()], annotated)

    assert one.others == ()


def test_las_patentes_anotadas_salen_del_export_sin_las_cajas_invalidas(tmp_path: Path) -> None:
    folder = tmp_path / "export"
    (folder / "train").mkdir(parents=True)
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 400, "height": 300}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 40]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 200, 60, 25]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [50, 50, 300, 250]},  # contiene a las otras
        ],
        "categories": [{"id": 1, "name": "patente"}],
    }
    (folder / "train" / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    found = rs.annotated_plates(folder)

    assert [(b.x1, b.y1, b.x2, b.y2) for b in found[("train", "a.jpg")]] == [
        (100, 100, 200, 140),
        (300, 200, 360, 225),
    ]


def test_sin_detector_de_patentes_no_hay_redaccion(tmp_path: Path) -> None:
    [one] = measure(tmp_path, None, [sample()])

    assert one.unavailable and one.plates == 0


def test_solo_cuentan_las_patentes_de_auto_y_la_foto_se_lee_una_vez(tmp_path: Path) -> None:
    moto = Sample("train", "p1.jpg", TARGET, "ABC123", format="moto", light="dia", ident="002")
    reader = FakeReader()
    names = ["p1.jpg"]
    result, skipped = rs.measure_samples(
        use_case([FULL, PIECE], reader),
        [sample("001"), moto, sample("003")],
        photos(tmp_path, *names),
        1024,
        REDACT,
    )

    assert [m.ident for m in result] == ["001", "003"] and not skipped
    assert reader.calls == 2  # FULL y PIECE una sola vez, aunque dos filas usen la foto


def test_una_foto_que_no_se_puede_abrir_se_saltea_y_se_cuenta(tmp_path: Path) -> None:
    result, skipped = rs.measure_samples(
        use_case([FULL]), [sample(file="no-esta.jpg")], photos(tmp_path), 1024, REDACT
    )

    assert result == [] and skipped == {"foto ilegible": 1}


def test_el_informe_cuenta_antes_y_despues_total_y_por_luz(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    bigger = Box(60, 40, 240, 160, 0.2, BoxLabel.PLATE)
    neighbour = Box(201, 80, 260, 120, 0.8, BoxLabel.PLATE)
    measures = [
        *measure(tmp_path / "a", [CAR, FULL, main, bigger], [sample("001", light="dia")]),  # total -> nada
        *measure(
            tmp_path / "b", [CAR, FULL, main, neighbour], [sample("002", light="noche")]
        ),  # parcial -> nada
        *measure(tmp_path / "c", [CAR, FULL, STRAY], [sample("003", light="dia")]),  # sin cambios
        *measure(tmp_path / "d", None, [sample("004", light="noche")]),
    ]

    report = rs.build_report(measures, REDACT, {"detector": "d", "ocr": "o"})

    assert report["fotos"] == 4 and report["sin_detector_de_patentes"] == 1
    assert report["antes"]["tapa_total"] == 1 and report["antes"]["tapa_parcial"] == 1
    assert report["antes"]["tapa_algo"] == 2 and report["despues"]["tapa_algo"] == 0
    assert report["cambian_de_clase"] == 2 and report["empeoran"] == 0
    assert report["por_luz"]["dia"]["antes"]["tapa_total"] == 1
    assert report["por_luz"]["noche"]["antes"]["tapa_algo"] == 1
    assert report["por_luz"]["noche"]["despues"]["tapa_algo"] == 0
    assert (
        report["umbrales"]["contenida_minima"] == 0.8 and report["umbrales"]["contiene_entera_desde"] == 0.95
    )
    assert report["ajenas"]["patentes_ajenas_anotadas"] == 0


def test_los_casos_tapados_con_exencion_traen_el_tipo_de_caja(tmp_path: Path) -> None:
    main = Box(10, 10, 50, 30, 0.95, BoxLabel.PLATE)
    overlapping = Box(150, 90, 260, 130, 0.8, BoxLabel.PLATE)
    measures = measure(tmp_path, [CAR, FULL, main, overlapping], [sample("001")])

    report = rs.build_report(measures, REDACT, {})

    assert [case["id"] for case in report["casos_tapados_con_exencion"]] == ["001"]
    assert report["tipos_de_caja_que_cubre_con_exencion"] == {rs.OVERLAP: 1}


def test_las_filas_no_llevan_el_texto_de_la_patente(tmp_path: Path) -> None:
    measures = measure(tmp_path, [FULL, PIECE], [sample()])
    out = tmp_path / "rows.csv"

    rs.write_rows(out, measures)

    assert REPORTED not in out.read_text(encoding="utf-8")
    with out.open(encoding="utf-8", newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["id"] == "001" and row["exceptuadas"] == "1" and float(row["cobertura_despues"]) == 0.0


# --- línea de comandos ----------------------------------------------------------------------------


def test_se_niega_a_leer_o_escribir_dentro_del_repo(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = rs.main(
        [
            "--labeled-set",
            str(REPO_ROOT / "set"),
            "--photos-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "o"),
        ]
        + ["--models-dir", str(tmp_path)]
    )

    assert code == 1 and "dentro del repositorio" in capsys.readouterr().err


def test_faltan_argumentos_obligatorios(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        rs.parse_args(["--photos-dir", str(tmp_path)])


def test_el_informe_es_json_serializable(tmp_path: Path) -> None:
    measures = measure(tmp_path, [FULL, PIECE], [sample()])

    json.dumps(rs.build_report(measures, REDACT, {}))
