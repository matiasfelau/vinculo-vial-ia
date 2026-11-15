"""tools/training/reid/partial.py: eje del auto por las poses, vistas frontales y traseras y pares cruzados.

Todo sintético: azimuts y proporciones inventados, sin fotos ni modelos.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDER = REPO_ROOT / "tools" / "training" / "reid"


def load(name: str) -> ModuleType:
    for path in (str(FOLDER), str(REPO_ROOT / "tools" / "eval")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(f"reid_{name}", FOLDER / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"reid_{name}"] = module
    spec.loader.exec_module(module)
    return module


partial = load("partial")


def aspects(azimuths: list[float], side: float, mean: float = 2.0, amplitude: float = 0.8) -> list[float]:
    """Proporción ancho/alto de la caja: la más ancha de costado (`side`), la más angosta de frente."""
    return [mean + amplitude * math.cos(2 * math.radians(a - side)) for a in azimuths]


AZIMUTHS = [i * 30.0 for i in range(12)]


def test_side_axis_encuentra_la_vista_de_costado() -> None:
    for side in (0.0, 40.0, 100.0, 170.0):
        found = partial.side_axis(AZIMUTHS, aspects(AZIMUTHS, side))

        assert found is not None
        assert abs(((found[0] - side + 90) % 180) - 90) < 1.0  # el eje es módulo 180°
        assert found[2] > 0.99


def test_side_axis_con_ruido_sigue_encontrandolo() -> None:
    noisy = [a + (0.05 if i % 2 else -0.05) for i, a in enumerate(aspects(AZIMUTHS, 60.0))]

    found = partial.side_axis(AZIMUTHS, noisy)

    assert found is not None and abs(found[0] - 60.0) < 5.0


def test_side_axis_sin_cambio_de_proporcion_no_hay_eje() -> None:
    assert partial.side_axis(AZIMUTHS, [2.0] * 12) is None
    assert partial.side_axis(AZIMUTHS, aspects(AZIMUTHS, 30.0, amplitude=0.1)) is None  # amplitud chica


def test_side_axis_con_pocas_vistas_no_hay_eje() -> None:
    assert partial.side_axis(AZIMUTHS[:4], aspects(AZIMUTHS[:4], 0.0)) is None
    assert partial.side_axis(AZIMUTHS, aspects(AZIMUTHS, 0.0)[:-1]) is None  # largos distintos


def test_las_vistas_frontales_y_traseras_estan_a_noventa_grados_de_la_de_costado() -> None:
    side = 40.0

    front_or_rear = [a for a in AZIMUTHS if partial.is_front_or_rear(a, side)]

    assert front_or_rear == [120.0, 150.0, 300.0, 330.0]  # 130° y 310° +- 30°
    assert not partial.is_front_or_rear(40.0, side)  # de costado
    assert not partial.is_front_or_rear(220.0, side)  # del otro costado


def test_is_front_or_rear_da_la_vuelta_al_circulo() -> None:
    assert partial.is_front_or_rear(5.0, 275.0)  # 275 + 90 = 365 = 5
    assert partial.is_front_or_rear(355.0, 275.0)


def test_pick_separated_elige_las_de_mayor_puntaje_separadas() -> None:
    candidates = [(100.0, 0.9), (110.0, 0.95), (200.0, 0.5), (300.0, 0.7)]

    chosen = partial.pick_separated(candidates, 2, 30.0)

    assert chosen == [1, 3]  # la de 110° (0,95) y la de 300° (0,7); la de 100° está a 10° de la primera
    assert partial.pick_separated(candidates, 5, 30.0) == [1, 3, 2]


def test_plate_inside_car_pide_el_centro_dentro_y_el_ancho_en_rango() -> None:
    car = (0.0, 0.0, 1000.0, 600.0)

    assert partial.plate_inside_car((450.0, 400.0, 570.0, 440.0), car) == 0.12
    assert partial.plate_inside_car((450.0, 400.0, 460.0, 410.0), car) is None  # 1 % del ancho: muy chica
    assert partial.plate_inside_car((100.0, 100.0, 600.0, 200.0), car) is None  # 50 %: muy grande
    assert partial.plate_inside_car((1500.0, 400.0, 1620.0, 440.0), car) is None  # fuera del vehículo


def test_cross_same_pairs_pide_separacion_y_el_mismo_auto() -> None:
    general = [("a", 0.0), ("a", 20.0), ("a", 90.0), ("b", 0.0)]
    parcial = [("a", 10.0), ("b", 200.0)]

    pairs = partial.cross_same_pairs(general, parcial, 30.0)

    assert [(i, j) for i, j, _ in pairs] == [(2, 0), (3, 1)]  # 0° y 20° están a menos de 30° de 10°
    assert [round(a) for _, _, a in pairs] == [80, 160]


def test_cross_different_pairs_no_mezcla_el_mismo_auto_y_respeta_los_dificiles() -> None:
    general = ["a", "a", "b", "c"]
    parcial = ["a", "b", "c"]
    brand_color = {"a": ("x", "blanco"), "b": ("x", "blanco"), "c": ("y", "negro")}

    pairs = partial.cross_different_pairs(general, parcial, brand_color, limit=100)
    hard = partial.cross_different_pairs(general, parcial, brand_color, limit=100, hard=True)

    assert all(general[i] != parcial[j] for i, j in pairs)
    assert len(pairs) == 4 * 3 - (2 + 1 + 1)  # todas las combinaciones menos las del mismo auto
    assert sorted(hard) == [(0, 1), (1, 1), (2, 0)]  # a-b y b-a: misma marca y color
    assert partial.cross_different_pairs([], parcial, brand_color, 10) == []


def test_cross_different_pairs_con_limite_sortea_sin_repetir_y_es_determinista() -> None:
    general = [f"g{i}" for i in range(40)]
    parcial = [f"g{i}" for i in range(40)]
    brand_color = {f"g{i}": ("m", "c") for i in range(40)}

    first = partial.cross_different_pairs(general, parcial, brand_color, limit=50, seed=3)

    assert len(first) == 50 == len(set(first))
    assert first == partial.cross_different_pairs(general, parcial, brand_color, limit=50, seed=3)
    assert first != partial.cross_different_pairs(general, parcial, brand_color, limit=50, seed=4)


# --- simulación con un detector falso -----------------------------------------------------------------------


class FakeDetector:
    """Ve un auto grande y, si la foto es grande (la original), una patente chica en su centro-abajo."""

    def __init__(self) -> None:
        from vvi_ia.domain.model import Box, BoxLabel

        self.box, self.label = Box, BoxLabel

    def detect(self, img: object) -> list[object]:
        height, width = img.shape[:2]  # type: ignore[attr-defined]
        boxes = [self.box(0.1 * width, 0.2 * height, 0.9 * width, 0.9 * height, 0.95, self.label.CAR)]
        if width > 900:  # solo en la vista original reducida a 1024: en los recortes simulados no se pide
            plate_w = 0.15 * 0.8 * width
            cx, cy = 0.5 * width, 0.7 * height
            boxes.append(
                self.box(
                    cx - plate_w / 2,
                    cy - plate_w / 8,
                    cx + plate_w / 2,
                    cy + plate_w / 8,
                    0.9,
                    self.label.PLATE,
                )
            )
        return boxes


class FakeModels:
    vehicle = FakeDetector()


def test_simulate_car_hace_patentes_solo_de_vistas_frontales_o_traseras_y_detalles(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from vvi_ia.adapters.imaging import CvImageDecoder

    side = 40.0
    azimuths = [i * 30.0 for i in range(12)]
    ratios = aspects(azimuths, side)
    raw = tmp_path / "raw" / "auto1"
    raw.mkdir(parents=True)
    frames = []
    for number, (azimuth, ratio) in enumerate(zip(azimuths, ratios, strict=True), 1):
        image = np.full((1440, 1920, 3), 120, np.uint8)
        cv2.imencode(".jpg", image)[1].tofile(str(raw / f"frame_{number:05d}.jpg"))
        frames.append({"foto": number, "azimut": azimuth, "ancho": round(ratio * 400), "alto": 400})

    result = partial.simulate_car(
        "auto1", frames, tmp_path / "raw", tmp_path / "out", FakeModels(), CvImageDecoder(), seed=0
    )

    items = result["items"]
    plates = [i for i in items if i["tipo"] == "patente"]
    details = [i for i in items if i["tipo"] == "detalle"]
    assert result["stats"]["con_eje"] == 1
    assert 1 <= len(plates) <= partial.PLATE_PER_CAR
    assert all(partial.is_front_or_rear(i["azimut"], side) for i in plates)  # vistas frontales o traseras
    assert len(details) == partial.DETAIL_PER_CAR
    for item in items:
        assert (tmp_path / "out" / item["archivo"]).is_file()
        assert item["ancho"] > 0 and item["alto"] > 0 and len(item["ventana"]) == 4
    assert all(0.10 <= i["patente_sobre_vehiculo"] <= 0.30 for i in plates)


def test_simulate_car_sin_eje_no_hace_patentes_pero_si_detalles(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from vvi_ia.adapters.imaging import CvImageDecoder

    raw = tmp_path / "raw" / "auto2"
    raw.mkdir(parents=True)
    frames = []
    for number in range(1, 13):
        cv2.imencode(".jpg", np.full((1440, 1920, 3), 90, np.uint8))[1].tofile(
            str(raw / f"frame_{number:05d}.jpg")
        )
        frames.append(
            {"foto": number, "azimut": (number - 1) * 30.0, "ancho": 800, "alto": 400}
        )  # siempre 2:1

    result = partial.simulate_car(
        "auto2", frames, tmp_path / "raw", tmp_path / "out", FakeModels(), CvImageDecoder(), seed=0
    )

    assert result["stats"]["con_eje"] == 0
    assert [i["tipo"] for i in result["items"]].count("patente") == 0
    assert [i["tipo"] for i in result["items"]].count("detalle") == partial.DETAIL_PER_CAR
