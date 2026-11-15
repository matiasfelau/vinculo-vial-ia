import pytest

from vvi_ia.domain.model import Box, BoxLabel
from vvi_ia.domain.redaction import RedactionThresholds, expand, plan_redaction

THRESHOLDS = RedactionThresholds(face_min_conf=0.30, plate_min_conf=0.20)
W, H = 1000, 800


def _plan(
    faces: list[Box] | None,
    detections: list[Box] | None,
    exempt: list[Box] | None = None,
) -> list[Box] | None:
    return plan_redaction(faces, detections, THRESHOLDS, W, H, exempt_plates=exempt or [])


def _plate(x1: float, conf: float = 0.9) -> Box:
    return Box(x1, 100, x1 + 100, 150, conf, BoxLabel.PLATE)


def test_cubre_todos_los_rostros_y_patentes() -> None:
    faces = [Box(10, 10, 60, 60, 0.9), Box(300, 300, 360, 380, 0.5)]
    detections = [_plate(500), _plate(700), Box(0, 0, 900, 700, 0.99, BoxLabel.CAR)]

    plan = _plan(faces, detections)

    assert plan is not None
    assert len(plan) == 4


def test_excepcion_solo_para_la_patente_que_coincide() -> None:
    matched, other = _plate(500), _plate(700)

    plan = _plan([], [matched, other], exempt=[matched])

    assert plan is not None
    assert len(plan) == 1
    assert plan[0].x1 < other.x1 < plan[0].x2  # lo que queda tapado es la otra patente


def test_los_rostros_nunca_se_eximen() -> None:
    face = Box(10, 10, 60, 60, 0.9)

    plan = _plan([face], [], exempt=[face])

    assert plan is not None
    assert len(plan) == 1


@pytest.mark.parametrize(
    ("faces", "detections"),
    [(None, []), ([], None), (None, None)],
)
def test_si_falta_un_detector_no_hay_plan(faces: list[Box] | None, detections: list[Box] | None) -> None:
    assert _plan(faces, detections) is None


def test_confianza_bajo_el_umbral_no_se_tapa() -> None:
    plan = _plan([Box(0, 0, 10, 10, 0.29)], [_plate(100, conf=0.19)])

    assert plan == []


def test_confianza_en_el_umbral_se_tapa() -> None:
    plan = _plan([Box(0, 0, 10, 10, 0.30)], [_plate(100, conf=0.20)])

    assert plan is not None
    assert len(plan) == 2


def test_con_el_umbral_de_patentes_de_0_10_se_tapa_la_de_confianza_baja_pero_no_la_de_0_08() -> None:
    """16/11/2026: PLATE_REDACT_MIN_CONF=0,10; los rostros siguen en su propio umbral."""
    thresholds = RedactionThresholds(face_min_conf=0.25, plate_min_conf=0.10)
    detections = [_plate(100, conf=0.12), _plate(400, conf=0.08)]

    plan = plan_redaction([Box(0, 0, 10, 10, 0.20)], detections, thresholds, W, H)

    assert plan is not None
    # el rostro de 0,20 queda bajo su umbral (0,25): solo se tapa la patente de 0,12
    assert [(box.label, round(box.x1)) for box in plan] == [(BoxLabel.PLATE, 90)]


def test_la_patente_exceptuada_no_se_tapa_aunque_baje_el_umbral() -> None:
    thresholds = RedactionThresholds(face_min_conf=0.25, plate_min_conf=0.10)
    reported, other = _plate(100, conf=0.9), _plate(400, conf=0.11)

    plan = plan_redaction([], [reported, other], thresholds, W, H, exempt_plates=[reported])

    assert plan is not None and [round(box.x1) for box in plan] == [390]


def test_las_cajas_que_no_son_patente_no_se_tapan() -> None:
    wheel = Box(10, 10, 50, 50, 0.99, BoxLabel.WHEEL)

    assert _plan([], [wheel]) == []


def test_margen_del_diez_por_ciento_por_lado() -> None:
    box = expand(Box(100, 100, 200, 150), 0.10, W, H)

    assert (box.x1, box.y1, box.x2, box.y2) == (90.0, 95.0, 210.0, 155.0)


def test_el_margen_se_recorta_a_la_imagen() -> None:
    box = expand(Box(0, 0, 100, 100), 0.10, 105, H)

    assert (box.x1, box.y1, box.x2) == (0.0, 0.0, 105.0)


def test_el_plan_aplica_el_margen() -> None:
    plan = _plan([Box(100, 100, 200, 200, 0.9)], [])

    assert plan is not None
    assert (plan[0].x1, plan[0].x2) == (90.0, 210.0)
