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


def measure(tmp_path: Path, plates: list[Box] | None, samples: list[Sample]) -> list[Any]:
    names = sorted({s.file for s in samples})
    result, _ = rs.measure_samples(use_case(plates), samples, photos(tmp_path, *names), 1024, REDACT)
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


def test_un_pedazo_contenido_tapa_parcialmente_antes_y_ya_no_despues(tmp_path: Path) -> None:
    [one] = measure(tmp_path, [FULL, PIECE], [sample()])

    assert one.plates == 2 and one.exempt == 1
    assert (
        rs.classify(one.before) == "parcial" and one.before > 0.5
    )  # el pedazo (con su margen) cubre la mitad
    assert one.after == 0.0 and rs.classify(one.after) == "ninguna"


def test_sin_detecciones_contenidas_antes_y_despues_son_iguales(tmp_path: Path) -> None:
    [one] = measure(tmp_path, [FULL, STRAY], [sample()])

    assert one.exempt == 1 and one.before == one.after == 0.0


def test_si_el_ocr_no_lee_la_denunciada_se_tapa_igual_antes_y_despues(tmp_path: Path) -> None:
    [one] = measure(tmp_path, [FULL], [sample(label="XY987ZW")])  # el OCR lee otra: sin exención

    assert one.exempt == 0
    assert rs.classify(one.before) == "total" and rs.classify(one.after) == "total"


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
    measures = [
        *measure(tmp_path / "a", [FULL, PIECE], [sample("001", light="dia")]),
        *measure(tmp_path / "b", [FULL, STRAY], [sample("002", light="noche")]),
        *measure(tmp_path / "c", [FULL], [sample("003", label="XY987ZW", light="dia")]),
        *measure(tmp_path / "d", None, [sample("004", light="noche")]),
    ]

    report = rs.build_report(measures, REDACT, {"detector": "d", "ocr": "o"})

    assert report["fotos"] == 4 and report["sin_detector_de_patentes"] == 1
    assert report["con_exencion"] == 2 and report["sin_exencion"] == 1
    assert report["antes"]["tapa_parcial"] == 1 and report["antes"]["tapa_total"] == 1
    assert report["despues"]["tapa_parcial"] == 0 and report["despues"]["tapa_total"] == 1
    assert report["antes"]["tapa_algo"] == 2 and report["despues"]["tapa_algo"] == 1
    assert report["antes"]["tapa_total_o_parcial"] == 2 and report["despues"]["tapa_total_o_parcial"] == 1
    assert (
        report["con_exencion_antes"]["tapa_parcial"] == 1 and report["con_exencion_despues"]["sin_tapar"] == 2
    )
    assert report["cambian_de_clase"] == 1 and report["empeoran"] == 0
    assert report["por_luz"]["dia"]["antes"]["tapa_total_o_parcial"] == 2
    assert report["por_luz"]["noche"]["antes"]["tapa_total_o_parcial"] == 0
    assert report["umbrales"]["contenida_minima"] == 0.8


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
