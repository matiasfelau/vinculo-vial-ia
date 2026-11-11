"""tools/eval/cascade.py: armado de alertas, estadísticas y medición con modelos falsos."""

import importlib.util
import logging
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from vvi_ia.domain.model import AlertType, Box, BoxLabel, EvidenceRole

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL = REPO_ROOT / "tools" / "eval"
pytest.importorskip("pycocotools")


def load() -> ModuleType:
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    spec = importlib.util.spec_from_file_location("eval_cascade", EVAL / "cascade.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_cascade"] = module
    spec.loader.exec_module(module)
    return module


cascade = load()


def write_photo(path: Path) -> Path:
    cv2.imwrite(str(path), np.full((20, 30, 3), 90, np.uint8))
    return path


def test_summarize_da_mediana_media_p95_y_maximo() -> None:
    result = cascade.summarize([100.0, 200.0, 300.0, 400.0, 1000.0])

    assert result == {"mediana": 300, "media": 400, "p95": 1000, "maximo": 1000}
    assert cascade.summarize([5.0]) == {"mediana": 5, "media": 5, "p95": 5, "maximo": 5}


def test_build_alert_sube_las_fotos_con_su_hash(tmp_path: Path) -> None:
    store = cascade.MemoryStore()
    car, detail = write_photo(tmp_path / "car.jpg"), write_photo(tmp_path / "detail.jpg")
    photos = {EvidenceRole.GENERAL: [car], EvidenceRole.PLATE: [car], EvidenceRole.DETAIL: [detail, detail]}
    layout = cascade.Layout(photos, b"AB123CD")

    payload = cascade.build_alert(AlertType.CRASHED, layout, store)

    assert [(i.role, i.index) for i in payload.evidence] == [
        (EvidenceRole.GENERAL, 0),
        (EvidenceRole.PLATE, 0),
        (EvidenceRole.DETAIL, 0),
        (EvidenceRole.DETAIL, 1),
    ]
    assert all(
        store.get(i.key) == car.read_bytes() or store.get(i.key) == detail.read_bytes()
        for i in payload.evidence
    )
    assert len({i.key for i in payload.evidence}) == 4


def test_photos_in_solo_toma_imagenes(tmp_path: Path) -> None:
    write_photo(tmp_path / "b.jpg")
    write_photo(tmp_path / "a.png")
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")

    assert [p.name for p in cascade.photos_in(tmp_path)] == ["a.png", "b.jpg"]


def test_duration_capture_guarda_las_ultimas_duraciones() -> None:
    capture = cascade.DurationCapture()
    logger = logging.getLogger("test_cascade_capture")
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)

    logger.info("sin duraciones")
    assert capture.last == {}
    logger.info("validación", extra={"durations_ms": {"load": 1.5}})
    assert capture.last == {"load": 1.5}


def test_readable_plates_descarta_las_fotos_sin_auto_patente_o_lectura_segura(tmp_path: Path) -> None:
    levels = {"ok": 40, "sin_auto": 90, "sin_patente": 140, "dudosa": 190}  # el brillo identifica la foto
    pool = []
    for name, level in levels.items():
        path = tmp_path / f"{name}.jpg"
        cv2.imwrite(str(path), np.full((20, 30, 3), level, np.uint8))
        pool.append(path)
    plate, car = Box(2, 2, 10, 8, 0.9, BoxLabel.PLATE), Box(0, 0, 30, 20, 0.9, BoxLabel.CAR)
    boxes = {40: [car, plate], 90: [plate], 140: [car], 190: [car, plate]}

    def which(img: np.ndarray) -> int:
        return min(boxes, key=lambda level: abs(level - int(img.mean())))

    class Detector:
        def detect(self, img: np.ndarray) -> list[Box]:
            return boxes[which(img)]

    class Reader:
        def read(self, img: np.ndarray) -> tuple[str, float]:
            return ("AB 123 CD", 0.99) if which(img) == 40 else ("XY999ZZ", 0.5)

    found = cascade.readable_plates(
        SimpleNamespace(vehicle=Detector(), plate=Reader()),
        pool,
        random.Random(0),  # noqa: S311
        wanted=1,
    )

    assert [(path.name, text) for path, text in found] == [("ok.jpg", b"AB123CD")]


def test_la_salida_dentro_del_repo_se_rechaza(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cascade.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(REPO_ROOT / "salida"), "--models-dir", str(tmp_path)]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_sin_fotos_no_arranca(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cascade.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--models-dir", str(tmp_path)]
    )

    assert code == 1 and "No hay fotos" in capsys.readouterr().err


def test_resized_copy_deja_el_lado_mayor_en_el_valor_pedido_y_no_repite_el_trabajo(tmp_path: Path) -> None:
    source = tmp_path / "car.jpg"
    cv2.imwrite(str(source), np.full((30, 60, 3), 90, np.uint8))

    target = cascade.resized_copy(source, 240, tmp_path / "copias")

    assert cv2.imread(str(target)).shape[:2] == (120, 240)
    stamp = target.stat().st_mtime_ns
    assert cascade.resized_copy(source, 240, tmp_path / "copias") == target
    assert target.stat().st_mtime_ns == stamp


def test_readable_plates_usa_la_foto_preparada(tmp_path: Path) -> None:
    import random

    original = write_photo(tmp_path / "a.jpg")
    prepared = write_photo(tmp_path / "grande.jpg")
    seen: list[Path] = []

    class Detector:
        def detect(self, img: np.ndarray) -> list[Box]:
            return [Box(0, 0, 30, 20, 0.9, BoxLabel.CAR), Box(2, 2, 10, 8, 0.9, BoxLabel.PLATE)]

    class Reader:
        def read(self, img: np.ndarray) -> tuple[str, float]:
            return "AB123CD", 0.99

    def prepare(path: Path) -> Path:
        seen.append(path)
        return prepared

    found = cascade.readable_plates(
        SimpleNamespace(vehicle=Detector(), plate=Reader()),
        [original],
        random.Random(0),  # noqa: S311
        wanted=1,
        prepare=prepare,
    )

    assert seen == [original] and found == [(prepared, b"AB123CD")]


def test_has_part_acepta_autos_o_patentes_grandes_y_rechaza_el_resto() -> None:
    image = np.zeros((100, 100, 3), np.uint8)

    def models(boxes: list[Box], seg: list[int]) -> SimpleNamespace:
        cars = [
            SimpleNamespace(mask=np.pad(np.ones((s, s), bool), ((0, 100 - s), (0, 100 - s)))) for s in seg
        ]
        return SimpleNamespace(
            vehicle=SimpleNamespace(detect=lambda img: boxes), cars=SimpleNamespace(cars=lambda img: cars)
        )

    wheel = Box(0, 0, 40, 40, 0.9, BoxLabel.WHEEL)
    assert not cascade.has_part(models([wheel], []), image, 0.05)  # la rueda ya no cuenta
    assert cascade.has_part(models([Box(0, 0, 40, 40, 0.9, BoxLabel.PLATE)], []), image, 0.05)
    assert not cascade.has_part(models([Box(0, 0, 10, 10, 0.9, BoxLabel.PLATE)], []), image, 0.05)  # chica
    assert not cascade.has_part(
        models([Box(0, 0, 40, 40, 0.3, BoxLabel.PLATE)], []), image, 0.05
    )  # poco confiable
    assert cascade.has_part(models([], [30]), image, 0.05)  # un auto segmentado


def test_duration_capture_guarda_tambien_el_veredicto() -> None:
    capture = cascade.DurationCapture()
    logger = logging.getLogger("test_cascade_capture_verdict")
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)

    logger.info("validación", extra={"durations_ms": {"load": 1.0}, "verdict": "GREEN"})

    assert capture.verdict == "GREEN"
