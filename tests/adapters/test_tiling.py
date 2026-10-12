"""Mosaico: grilla de tiles, NMS por clase, descarte de cajas cortadas y su uso en el núcleo del detector."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from vvi_ia.adapters.models.rfdetr_detector import FACE, RfDetrDetections, Tiling
from vvi_ia.adapters.models.tiling import detect_tiled, iou, nms, tile_grid
from vvi_ia.domain.model import Box, BoxLabel


def box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9, label: str = FACE) -> Box:
    return Box(x1, y1, x2, y2, conf=conf, label=label)


# --- grilla --------------------------------------------------------------------------------


def test_una_foto_que_cabe_en_un_tile_no_se_mosaiquea() -> None:
    assert tile_grid(300, 400, 640, 0.25) == []
    assert tile_grid(640, 640, 640, 0.25) == []


def test_los_tiles_cubren_toda_la_foto_y_se_solapan() -> None:
    height, width = 900, 1600
    tiles = tile_grid(height, width, 640, 0.25)

    covered = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 <= 640 and y2 - y1 <= 640
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    xs = sorted({t[0] for t in tiles})
    assert all(b - a <= 640 * 0.75 + 1 for a, b in zip(xs, xs[1:], strict=False))  # solape de al menos 25 %
    assert max(t[2] for t in tiles) == width and max(t[3] for t in tiles) == height


def test_un_eje_corto_usa_todo_su_largo() -> None:
    tiles = tile_grid(300, 1600, 640, 0.25)

    assert {(t[1], t[3]) for t in tiles} == {(0, 300)}
    assert len(tiles) >= 3


@pytest.mark.parametrize(("tile", "overlap"), [(10, 0.2), (640, -0.1), (640, 0.95)])
def test_parametros_fuera_de_rango(tile: int, overlap: float) -> None:
    with pytest.raises(ValueError, match="fuera de rango"):
        tile_grid(2000, 2000, tile, overlap)


# --- NMS -----------------------------------------------------------------------------------


def test_iou_de_cajas_iguales_y_disjuntas() -> None:
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_nms_conserva_la_mas_confiable_y_distingue_clases() -> None:
    kept = nms(
        [
            box(0, 0, 10, 10, 0.5),
            box(1, 1, 11, 11, 0.9),
            box(0, 0, 10, 10, 0.8, BoxLabel.PLATE),  # otra clase: no compite
            box(50, 50, 60, 60, 0.3),
        ]
    )

    assert [(b.label, b.conf) for b in kept] == [(FACE, 0.9), (BoxLabel.PLATE, 0.8), (FACE, 0.3)]


# --- detect_tiled --------------------------------------------------------------------------


def test_detect_tiled_suma_lo_que_solo_se_ve_en_un_tile_y_descarta_lo_cortado() -> None:
    img = np.zeros((1000, 1000, 3), np.uint8)
    seen: list[tuple[int, int]] = []

    def detect(part: NDArray[Any]) -> list[Box]:
        seen.append(part.shape[:2])
        if part.shape[:2] == (1000, 1000):
            return []
        return [box(20, 30, 40, 50, 0.8)]

    tiles = tile_grid(1000, 1000, 640, 0.25)
    result = detect_tiled(detect, img, {FACE}, 640, 0.25, full=[])

    assert result is not None and len(seen) == len(tiles)
    assert len(result) == len(tiles)  # una cara por tile, en posiciones distintas
    offsets = {(b.x1, b.y1) for b in result}
    assert offsets == {(20.0 + x, 30.0 + y) for x, y, _, _ in tiles}


def test_una_caja_que_toca_el_borde_interior_de_un_tile_se_descarta() -> None:
    img = np.zeros((1000, 1000, 3), np.uint8)

    def detect(part: NDArray[Any]) -> list[Box]:
        if part.shape[0] == 1000:
            return []
        return [box(600, 100, 639, 140)]  # pegada al borde derecho de un tile

    result = detect_tiled(detect, img, {FACE}, 640, 0.25, full=[])

    # solo sobrevive en el/los tiles cuyo borde derecho es el borde de la foto
    assert result is not None
    assert all(b.x2 >= 1000 - 40 for b in result) and len(result) == 2


def test_detect_tiled_fusiona_con_la_foto_completa_y_filtra_por_clase() -> None:
    img = np.zeros((1000, 1000, 3), np.uint8)
    full = [box(100, 100, 140, 140, 0.9), box(0, 0, 500, 500, 0.9, BoxLabel.CAR)]

    def detect(part: NDArray[Any]) -> list[Box]:
        return (
            [box(100, 100, 140, 140, 0.6), box(0, 0, 50, 50, 0.9, BoxLabel.CAR)]
            if part.shape[0] < 1000
            else []
        )

    result = detect_tiled(detect, img, {FACE}, 640, 0.25, full=full)

    assert result is not None and all(b.label == FACE for b in result)
    top = [b for b in result if (b.x1, b.y1) == (100.0, 100.0)]
    assert [b.conf for b in top] == [0.9]  # el duplicado del tile se fusionó con el de la foto completa


def test_si_falla_un_tile_no_hay_resultado() -> None:
    img = np.zeros((1000, 1000, 3), np.uint8)
    calls = []

    def detect(part: NDArray[Any]) -> list[Box] | None:
        calls.append(1)
        return None if len(calls) == 3 else []

    assert detect_tiled(detect, img, {FACE}, 640, 0.25) is None
    assert detect_tiled(lambda part: None, img, {FACE}) is None


# --- núcleo del detector con mosaico ---------------------------------------------------------


class TiledSession:
    """Una cara chica solo en los tiles: la foto completa (la de 1000 px achicada) no la ve."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]) -> list[Any]:
        self.calls += 1
        dets = np.zeros((1, 2, 4), np.float32)
        labels = np.full((1, 2, 5), -6.0, np.float32)
        dets[0, 0] = (0.5, 0.5, 0.1, 0.1)
        labels[0, 0, 0] = 6.0 if self.calls == 1 else -6.0  # el auto, solo en la foto completa
        dets[0, 1] = (0.3, 0.3, 0.1, 0.1)
        labels[0, 1, 3] = -6.0 if self.calls == 1 else 6.0  # la cara, solo en los tiles
        return [dets, labels]


THRESHOLDS = {BoxLabel.CAR: 0.4, BoxLabel.WHEEL: 0.4, BoxLabel.PLATE: 0.1, FACE: 0.1}


def test_el_nucleo_con_mosaico_encuentra_la_cara_y_deja_el_auto_de_la_foto_completa() -> None:
    session = TiledSession()
    detections = RfDetrDetections(
        session, "input", (16, 16), THRESHOLDS, tiling=Tiling(frozenset({FACE}), tile=640, overlap=0.25)
    )

    boxes = detections.detect_all(np.zeros((1000, 1000, 3), np.uint8))

    assert boxes is not None
    assert session.calls == 1 + len(tile_grid(1000, 1000, 640, 0.25))
    assert [b.label for b in boxes].count(BoxLabel.CAR) == 1
    assert [b.label for b in boxes].count(FACE) >= 1


def test_sin_mosaico_o_con_una_foto_chica_es_una_sola_inferencia() -> None:
    plain = TiledSession()
    RfDetrDetections(plain, "input", (16, 16), THRESHOLDS).detect_all(np.zeros((1000, 1000, 3), np.uint8))
    small = TiledSession()
    RfDetrDetections(small, "input", (16, 16), THRESHOLDS, tiling=Tiling(frozenset({FACE}))).detect_all(
        np.zeros((300, 300, 3), np.uint8)
    )

    assert plain.calls == 1 and small.calls == 1


# --- configuración -------------------------------------------------------------------------


def test_los_flags_de_mosaico_viajan_a_los_adaptadores(base_env: pytest.MonkeyPatch) -> None:
    from vvi_ia.config import load_settings
    from vvi_ia.main import adapter_params

    params = adapter_params(load_settings())
    assert params["face_tiling"] == 0.0 and params["plate_tiling"] == 0.0  # apagado por defecto
    assert params["tiling_tile_px"] == 640.0 and params["tiling_overlap"] == 0.25

    base_env.setenv("FACE_TILING", "true")
    base_env.setenv("TILING_TILE_PX", "800")
    params = adapter_params(load_settings())
    assert (
        params["face_tiling"] == 1.0 and params["plate_tiling"] == 0.0 and params["tiling_tile_px"] == 800.0
    )


def test_el_mosaico_se_arma_solo_para_las_clases_pedidas() -> None:
    from vvi_ia.adapters.models.rfdetr_detector import _tiling

    assert _tiling({}) is None
    both = _tiling({"face_tiling": 1.0, "plate_tiling": 1.0, "tiling_tile_px": 512.0, "tiling_overlap": 0.3})
    assert both is not None and both.labels == {FACE, BoxLabel.PLATE} and both.tile == 512
    only_plate = _tiling({"plate_tiling": 1.0})
    assert only_plate is not None and only_plate.labels == {BoxLabel.PLATE}


def test_una_foto_enorme_se_achica_antes_de_cortar_y_las_cajas_vuelven_a_su_escala() -> None:
    img = np.zeros((2000, 4000, 3), np.uint8)  # 4000 x 2000
    seen: list[tuple[int, int]] = []

    def detect(part: NDArray[Any]) -> list[Box]:
        seen.append(part.shape[:2])
        return [box(100, 100, 150, 150, 0.9)] if part.shape[0] < 2000 else []

    result = detect_tiled(detect, img, {FACE}, 640, 0.25, full=[], max_side=1000)  # factor 0,25 → 1000 x 500

    assert result is not None and all(shape[0] <= 500 and shape[1] <= 640 for shape in seen)
    first = min(result, key=lambda b: (b.x1, b.y1))
    assert (first.x1, first.y1, first.x2, first.y2) == (400.0, 400.0, 600.0, 600.0)  # ×4
    assert all(b.x2 <= 4000 and b.y2 <= 2000 for b in result)
