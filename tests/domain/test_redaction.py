from vvi_ia.domain.model import Box, BoxLabel
from vvi_ia.domain.redaction import (
    RedactionThresholds,
    contained_fraction,
    expand,
    main_plate,
    plan_redaction,
)

THRESHOLDS = RedactionThresholds(plate_min_conf=0.20)
W, H = 1000, 800


def _plan(detections: list[Box] | None, exempt: list[Box] | None = None) -> list[Box] | None:
    return plan_redaction(detections, THRESHOLDS, W, H, exempt_plates=exempt or [])


def _plate(x1: float, conf: float = 0.9) -> Box:
    return Box(x1, 100, x1 + 100, 150, conf, BoxLabel.PLATE)


def test_cubre_todas_las_patentes() -> None:
    detections = [_plate(500), _plate(700), Box(0, 0, 900, 700, 0.99, BoxLabel.CAR)]

    plan = _plan(detections)

    assert plan is not None
    assert len(plan) == 2


def test_excepcion_solo_para_la_patente_que_coincide() -> None:
    matched, other = _plate(500), _plate(700)

    plan = _plan([matched, other], exempt=[matched])

    assert plan is not None
    assert len(plan) == 1
    assert plan[0].x1 < other.x1 < plan[0].x2  # lo que queda tapado es la otra patente


def test_los_rostros_no_se_redactan() -> None:
    """Riesgo aceptado (16/11/2026): el producto tapa solo patentes; una caja de rostro no entra al plan."""
    face = Box(10, 10, 60, 60, 0.99, "face")  # type: ignore[arg-type]

    assert _plan([face]) == []


def test_la_disponibilidad_de_la_redaccion_solo_depende_del_detector_de_patentes() -> None:
    assert _plan(None) is None  # fail-closed: sin detecciones no hay plan
    assert _plan([]) == []  # miró y no encontró nada: nada que tapar, redacción disponible


def test_confianza_bajo_el_umbral_no_se_tapa() -> None:
    assert _plan([_plate(100, conf=0.19)]) == []


def test_confianza_en_el_umbral_se_tapa() -> None:
    plan = _plan([_plate(100, conf=0.20)])

    assert plan is not None
    assert len(plan) == 1


def test_con_el_umbral_de_patentes_de_0_10_se_tapa_la_de_confianza_baja_pero_no_la_de_0_08() -> None:
    """16/11/2026: PLATE_REDACT_MIN_CONF=0,10."""
    thresholds = RedactionThresholds(plate_min_conf=0.10)
    detections = [_plate(100, conf=0.12), _plate(400, conf=0.08)]

    plan = plan_redaction(detections, thresholds, W, H)

    assert plan is not None
    assert [(box.label, round(box.x1)) for box in plan] == [(BoxLabel.PLATE, 90)]


def test_la_patente_exceptuada_no_se_tapa_aunque_baje_el_umbral() -> None:
    thresholds = RedactionThresholds(plate_min_conf=0.10)
    reported, other = _plate(100, conf=0.9), _plate(400, conf=0.11)

    plan = plan_redaction([reported, other], thresholds, W, H, exempt_plates=[reported])

    assert plan is not None and [round(box.x1) for box in plan] == [390]


def test_las_cajas_que_no_son_patente_no_se_tapan() -> None:
    wheel = Box(10, 10, 50, 50, 0.99, BoxLabel.WHEEL)

    assert _plan([wheel]) == []


def test_margen_del_diez_por_ciento_por_lado() -> None:
    box = expand(Box(100, 100, 200, 150), 0.10, W, H)

    assert (box.x1, box.y1, box.x2, box.y2) == (90.0, 95.0, 210.0, 155.0)


def test_el_margen_se_recorta_a_la_imagen() -> None:
    box = expand(Box(0, 0, 100, 100), 0.10, 105, H)

    assert (box.x1, box.y1, box.x2) == (0.0, 0.0, 105.0)


# --- cajas contenidas en la patente exceptuada --------------------------------------------------


def test_una_caja_contenida_en_la_patente_exceptuada_no_se_tapa() -> None:
    """La misma patente detectada otra vez (o un pedazo, como en el mosaico) taparía la denunciada."""
    reported = Box(100, 100, 300, 160, 0.9, BoxLabel.PLATE)
    piece = Box(150, 110, 250, 150, 0.6, BoxLabel.PLATE)

    plan = _plan([reported, piece], exempt=[reported])

    assert plan == []


def test_el_umbral_de_contenida_es_el_ochenta_por_ciento_del_area() -> None:
    reported = Box(100, 100, 300, 160, 0.9, BoxLabel.PLATE)
    inside_80 = Box(
        220, 100, 320, 160, 0.9, BoxLabel.PLATE
    )  # 80 de 100 px de ancho adentro: justo en el umbral
    inside_79 = Box(221, 100, 321, 160, 0.9, BoxLabel.PLATE)  # 79 %: se tapa

    assert _plan([reported, inside_80], exempt=[reported]) == []
    plan = plan_redaction(
        [reported, inside_79], THRESHOLDS, W, H, exempt_plates=[reported], clip_margin=False
    )
    assert plan is not None and len(plan) == 1


def test_una_patente_ajena_que_solo_se_superpone_un_poco_se_tapa() -> None:
    reported = Box(100, 100, 300, 160, 0.9, BoxLabel.PLATE)
    neighbour = Box(280, 100, 480, 160, 0.9, BoxLabel.PLATE)  # 10 % adentro

    plan = plan_redaction([reported, neighbour], THRESHOLDS, W, H, [reported], clip_margin=False)

    assert plan is not None and len(plan) == 1


def test_sin_patente_exceptuada_una_caja_contenida_en_otra_se_tapa_igual() -> None:
    outer = Box(100, 100, 300, 160, 0.9, BoxLabel.PLATE)
    inner = Box(150, 110, 250, 150, 0.9, BoxLabel.PLATE)

    plan = _plan([outer, inner])

    assert plan is not None and len(plan) == 2


def test_sin_la_regla_de_contenidas_se_comporta_como_antes() -> None:
    reported = Box(100, 100, 300, 160, 0.9, BoxLabel.PLATE)
    piece = Box(150, 110, 250, 150, 0.6, BoxLabel.PLATE)

    plan = plan_redaction(
        [reported, piece], THRESHOLDS, W, H, exempt_plates=[reported], exempt_contained=None
    )

    assert plan is not None and len(plan) == 1  # el pedazo se tapa: es lo que se mide como "antes"


def test_contained_fraction() -> None:
    outer = Box(0, 0, 100, 100)

    assert contained_fraction(Box(10, 10, 20, 20), outer) == 1.0
    assert contained_fraction(Box(90, 0, 110, 100), outer) == 0.5
    assert contained_fraction(Box(200, 200, 300, 300), outer) == 0.0
    assert contained_fraction(Box(5, 5, 5, 9), outer) == 0.0  # sin área


# --- patente principal de la foto de patente --------------------------------------------------------

CAR = Box(100, 100, 900, 700, 0.95, BoxLabel.CAR)


def _plate_at(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, conf, BoxLabel.PLATE)


def test_la_principal_es_la_de_mayor_confianza_dentro_del_vehiculo_dominante() -> None:
    small_car = Box(0, 0, 50, 50, 0.99, BoxLabel.CAR)  # el dominante es el más grande
    best_inside = _plate_at(400, 500, 500, 540, conf=0.7)
    bigger_outside = _plate_at(920, 100, 990, 140, conf=0.95)
    weaker_inside = _plate_at(200, 500, 300, 540, conf=0.4)

    main = main_plate([small_car, CAR, bigger_outside, best_inside, weaker_inside], 0.10)

    assert main == best_inside


def test_si_ninguna_cae_en_el_vehiculo_la_principal_es_la_caja_de_patente_mas_grande() -> None:
    outside_big = _plate_at(920, 100, 990, 140, conf=0.2)
    outside_small = _plate_at(920, 300, 950, 320, conf=0.99)

    assert main_plate([CAR, outside_small, outside_big], 0.10) == outside_big


def test_sin_vehiculo_la_principal_es_la_caja_de_patente_mas_grande() -> None:
    big, small = _plate_at(0, 0, 100, 40, conf=0.2), _plate_at(0, 100, 50, 120, conf=0.9)

    assert main_plate([small, big], 0.10) == big


def test_la_principal_ignora_las_patentes_bajo_la_confianza_de_redaccion_y_sin_patentes_no_hay() -> None:
    weak = _plate_at(400, 500, 500, 540, conf=0.05)
    strong = _plate_at(200, 500, 250, 520, conf=0.3)

    assert main_plate([CAR, weak, strong], 0.10) == strong
    assert main_plate([CAR, weak], 0.10) is None
    assert main_plate([CAR], 0.10) is None
    assert main_plate([], 0.10) is None
    assert main_plate(None, 0.10) is None


def test_estar_dentro_del_vehiculo_es_tener_al_menos_la_mitad_del_area_adentro() -> None:
    half_in = _plate_at(850, 300, 950, 340, conf=0.5)  # 50 de 100 px adentro
    less_in = _plate_at(860, 300, 960, 340, conf=0.9)  # 40 %
    big_outside = _plate_at(920, 100, 1100, 200, conf=0.3)

    assert main_plate([CAR, half_in, less_in, big_outside], 0.10) == half_in


def test_la_principal_no_se_tapa_aunque_el_ocr_no_la_lea_y_las_demas_siguen_la_regla() -> None:
    main, other = _plate_at(400, 500, 500, 540, 0.9), _plate_at(150, 500, 250, 540, 0.5)

    plan = plan_redaction([CAR, main, other], THRESHOLDS, W, H, main_plate=main)

    assert plan is not None and len(plan) == 1
    assert plan[0].x1 < other.x1 < plan[0].x2  # lo que se tapa es la otra


def test_las_cajas_contenidas_en_la_principal_tampoco_se_tapan() -> None:
    main = _plate_at(400, 500, 500, 540, 0.9)
    piece = _plate_at(420, 505, 480, 535, 0.5)

    assert plan_redaction([CAR, main, piece], THRESHOLDS, W, H, main_plate=main) == []


def test_la_principal_se_suma_a_las_exceptuadas_por_lectura() -> None:
    read, main, other = (
        _plate_at(100, 500, 200, 540),
        _plate_at(400, 500, 500, 540),
        _plate_at(600, 500, 700, 540),
    )

    plan = plan_redaction([read, main, other], THRESHOLDS, W, H, exempt_plates=[read], main_plate=main)

    assert plan is not None and len(plan) == 1 and plan[0].x1 < other.x1


def test_sin_principal_el_plan_es_el_de_siempre() -> None:
    main = _plate_at(400, 500, 500, 540, 0.9)

    plan = plan_redaction([CAR, main], THRESHOLDS, W, H)

    assert plan is not None and len(plan) == 1


# --- cajas que contienen a la exceptuada y margen recortado ------------------------------------------


def _overlap(a: Box, b: Box) -> float:
    return Box(max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)).area


def test_una_caja_que_contiene_entera_a_la_exceptuada_no_se_tapa() -> None:
    exempt = _plate_at(400, 500, 500, 540, 0.9)
    bigger = _plate_at(350, 480, 560, 560, 0.2)  # la patente y lo que la rodea

    assert plan_redaction([exempt, bigger], THRESHOLDS, W, H, exempt_plates=[exempt]) == []


def test_el_umbral_de_contener_es_el_95_por_ciento_del_area_de_la_exceptuada() -> None:
    exempt = _plate_at(400, 500, 500, 540, 0.9)  # 100 x 40
    holds_95 = _plate_at(405, 480, 600, 560, 0.2)  # deja 5 de 100 px de la exceptuada afuera: 95 %
    holds_94 = _plate_at(406, 480, 600, 560, 0.2)  # 94 %: se tapa

    assert plan_redaction([exempt, holds_95], THRESHOLDS, W, H, exempt_plates=[exempt]) == []
    plan = plan_redaction([exempt, holds_94], THRESHOLDS, W, H, exempt_plates=[exempt])
    assert plan is not None and plan != []


def test_la_regla_de_contener_alcanza_tambien_a_la_principal() -> None:
    main = _plate_at(400, 500, 500, 540, 0.9)
    bigger = _plate_at(350, 480, 560, 560, 0.2)

    assert plan_redaction([CAR, main, bigger], THRESHOLDS, W, H, main_plate=main) == []


def test_sin_exceptuada_una_caja_grande_se_tapa_igual() -> None:
    plate = _plate_at(400, 500, 500, 540, 0.9)
    bigger = _plate_at(350, 480, 560, 560, 0.2)

    plan = plan_redaction([plate, bigger], THRESHOLDS, W, H)

    assert plan is not None and len(plan) == 2


def test_sin_la_regla_de_contener_se_comporta_como_antes() -> None:
    exempt = _plate_at(400, 500, 500, 540, 0.9)
    bigger = _plate_at(350, 480, 560, 560, 0.2)

    plan = plan_redaction(
        [exempt, bigger], THRESHOLDS, W, H, exempt_plates=[exempt], exempt_containing=None, clip_margin=False
    )

    assert plan is not None and len(plan) == 1


def test_el_margen_de_una_vecina_no_pisa_a_la_exceptuada_pero_la_caja_si_se_tapa_entera() -> None:
    exempt = _plate_at(100, 100, 200, 140, 0.9)
    neighbour = _plate_at(
        204, 100, 304, 140, 0.8
    )  # a 4 px: su margen de 10 px llega 6 px adentro de la exceptuada

    plan = plan_redaction([exempt, neighbour], THRESHOLDS, W, H, exempt_plates=[exempt])

    assert plan is not None
    assert all(_overlap(piece, exempt) == 0 for piece in plan)  # nada pisa a la exceptuada
    grown = expand(neighbour, 0.10, W, H)
    painted = sum(piece.area for piece in plan if piece is not neighbour)
    assert any(piece == neighbour for piece in plan)  # la caja de la vecina se tapa entera
    assert (
        painted >= grown.area - _overlap(grown, exempt) - 1e-6
    )  # y su margen, salvo lo que pisa la exceptuada


def test_el_margen_recortado_se_nota_en_el_plan() -> None:
    exempt = _plate_at(100, 100, 200, 140, 0.9)
    neighbour = _plate_at(204, 100, 304, 140, 0.8)
    clipped = plan_redaction([exempt, neighbour], THRESHOLDS, W, H, exempt_plates=[exempt])
    legacy = plan_redaction([exempt, neighbour], THRESHOLDS, W, H, exempt_plates=[exempt], clip_margin=False)

    assert legacy is not None and clipped is not None
    assert len(legacy) == 1 and _overlap(legacy[0], exempt) > 0  # antes el margen pisaba la exceptuada
    assert all(_overlap(piece, exempt) == 0 for piece in clipped)


def test_el_margen_solo_se_recorta_si_toca_una_exceptuada() -> None:
    exempt = _plate_at(100, 100, 200, 140, 0.9)
    far = _plate_at(600, 100, 700, 140, 0.8)

    plan = plan_redaction([exempt, far], THRESHOLDS, W, H, exempt_plates=[exempt])

    assert plan == [expand(far, 0.10, W, H)]  # una sola caja, con su margen entero


def test_si_la_caja_cruda_toca_a_la_exceptuada_esa_parte_se_tapa() -> None:
    """Solo se recorta el margen: la caja de otra patente se tapa aunque pise un poco a la exceptuada."""
    exempt = _plate_at(100, 100, 200, 140, 0.9)
    overlapping = _plate_at(180, 100, 280, 140, 0.8)  # 20 % de su área sobre la exceptuada: no es contenida

    plan = plan_redaction([exempt, overlapping], THRESHOLDS, W, H, exempt_plates=[exempt])

    assert plan is not None and any(piece == overlapping for piece in plan)


def test_el_margen_se_recorta_tambien_contra_la_principal() -> None:
    main = _plate_at(100, 100, 200, 140, 0.9)
    neighbour = _plate_at(204, 100, 304, 140, 0.8)

    plan = plan_redaction([CAR, main, neighbour], THRESHOLDS, W, H, main_plate=main)

    assert plan is not None and all(_overlap(piece, main) == 0 for piece in plan)
