import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from vvi_ia.domain.model import Box, BoxLabel
from vvi_ia.domain.ramp import DrivewayCheck, driveway_check, driveway_overlap, wheel_diameter

SIZE = 800
CURB_ROWS = slice(400, 410)
CUT_COLS = (300, 500)  # [inicio, fin)
WHEEL_DIAM = 50.0  # margen 0.08 → 4 px

Mask = NDArray[np.bool_]


def _scene(car_cols: tuple[int, int] | None, with_cut: bool = True) -> tuple[Mask, Mask, Mask]:
    curb = np.zeros((SIZE, SIZE), bool)
    cut = np.zeros((SIZE, SIZE), bool)
    car = np.zeros((SIZE, SIZE), bool)
    curb[CURB_ROWS, 100 : CUT_COLS[0]] = True
    curb[CURB_ROWS, CUT_COLS[1] : 700] = True
    if with_cut:
        cut[CURB_ROWS, CUT_COLS[0] : CUT_COLS[1]] = True
    else:
        curb[CURB_ROWS, CUT_COLS[0] : CUT_COLS[1]] = True
    if car_cols is not None:
        car[300:400, car_cols[0] : car_cols[1]] = True
    return curb, cut, car


def _rotate(masks: tuple[Mask, Mask, Mask], degrees: float) -> tuple[Mask, Mask, Mask]:
    matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), degrees, 1.0)
    rotated = [
        cv2.warpAffine(m.astype(np.uint8), matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST).astype(bool)
        for m in masks
    ]
    return rotated[0], rotated[1], rotated[2]


def _check(masks: tuple[Mask, Mask, Mask]) -> DrivewayCheck:
    return driveway_check(*masks, wheel_diam=WHEEL_DIAM)


def test_auto_sobre_el_rebaje_bloquea() -> None:
    assert _check(_scene((350, 450))) == DrivewayCheck.BLOCKING


def test_invasion_de_pocos_pixeles_bloquea() -> None:
    # el auto termina 3 px dentro del rebaje: la decisión es binaria, ya no hay un caso marginal
    assert _check(_scene((200, 303))) == DrivewayCheck.BLOCKING


def test_invasion_clara_bloquea() -> None:
    assert _check(_scene((200, 320))) == DrivewayCheck.BLOCKING


def test_contacto_exacto_bloquea() -> None:
    assert _check(_scene((200, 300))) == DrivewayCheck.BLOCKING


def test_a_menos_de_un_margen_del_rebaje_bloquea_y_mas_lejos_no() -> None:
    # margen 0,08 con ruedas de 50 px: 4 px. A 3 px del rebaje todavía bloquea; a 6 px, no
    assert _check(_scene((200, 298))) == DrivewayCheck.BLOCKING
    assert _check(_scene((200, 295))) == DrivewayCheck.NOT_BLOCKING


def test_el_score_es_el_solape_en_escalas_de_rueda() -> None:
    curb, cut, car = _scene((200, 325))  # 25 px dentro del rebaje

    measure = driveway_overlap(curb, cut, car, wheel_diam=WHEEL_DIAM)

    assert measure.status is None and measure.score == pytest.approx(25 / WHEEL_DIAM, abs=0.05)
    apart = driveway_overlap(*_scene((100, 250)), wheel_diam=WHEEL_DIAM)
    assert apart.score is not None and apart.score == pytest.approx(-50 / WHEEL_DIAM, abs=0.05)


def test_sin_auto_o_sin_rebaje_no_hay_score() -> None:
    assert driveway_overlap(*_scene(None), wheel_diam=WHEEL_DIAM) == driveway_overlap(
        *_scene(None), wheel_diam=WHEEL_DIAM
    )
    assert driveway_overlap(*_scene(None)).status == DrivewayCheck.NO_CAR
    assert driveway_overlap(*_scene((350, 450), with_cut=False)).score is None


def test_auto_separado_del_rebaje_no_bloquea() -> None:
    assert _check(_scene((100, 200))) == DrivewayCheck.NOT_BLOCKING


def test_sin_rebaje() -> None:
    assert _check(_scene((350, 450), with_cut=False)) == DrivewayCheck.NO_DRIVEWAY


def test_sin_auto() -> None:
    assert _check(_scene(None)) == DrivewayCheck.NO_CAR


def test_auto_diminuto_se_descarta_por_apertura_morfologica() -> None:
    curb, cut, car = _scene(None)
    car[390:392, 400:402] = True

    assert _check((curb, cut, car)) == DrivewayCheck.NO_CAR


def test_sin_rueda_usa_la_escala_de_respaldo() -> None:
    curb, cut, car = _scene((350, 450))

    assert driveway_check(curb, cut, car) == DrivewayCheck.BLOCKING


@pytest.mark.parametrize(
    ("car_cols", "expected"),
    [
        ((350, 450), DrivewayCheck.BLOCKING),
        ((100, 200), DrivewayCheck.NOT_BLOCKING),
    ],
)
@pytest.mark.parametrize("degrees", [30, -30])
def test_rotar_la_escena_no_cambia_el_resultado(
    car_cols: tuple[int, int], expected: DrivewayCheck, degrees: float
) -> None:
    assert _check(_rotate(_scene(car_cols), degrees)) == expected


def test_rotar_la_escena_sin_rebaje_sigue_sin_rebaje() -> None:
    assert _check(_rotate(_scene((350, 450), with_cut=False), 30)) == DrivewayCheck.NO_DRIVEWAY


def test_wheel_diameter_es_la_mediana_de_las_ruedas_dentro_del_auto() -> None:
    car = np.zeros((100, 100), bool)
    car[20:80, 20:80] = True
    wheels = [
        Box(25, 50, 45, 80, label=BoxLabel.WHEEL),  # centro dentro, alto 30
        Box(55, 55, 75, 95, label=BoxLabel.WHEEL),  # centro dentro, alto 40
        Box(0, 0, 10, 100, label=BoxLabel.WHEEL),  # centro fuera del auto
    ]

    assert wheel_diameter(wheels, car) == 35.0


def test_wheel_diameter_sin_ruedas_dentro_es_none() -> None:
    car = np.zeros((100, 100), bool)
    car[20:80, 20:80] = True

    assert wheel_diameter([Box(0, 0, 5, 5, label=BoxLabel.WHEEL)], car) is None
    assert wheel_diameter([], car) is None
