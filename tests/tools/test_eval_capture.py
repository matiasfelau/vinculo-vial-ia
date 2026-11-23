"""tools/eval/capture_sets.py y capture_r1.py: ventanas de encuadre, conjuntos, mitades y análisis de R1.

Todo sintético: cajas y señales inventadas, sin fotos ni modelos.
"""

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    tools = str(REPO_ROOT / "tools" / "eval")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(f"eval_{name}", REPO_ROOT / "tools" / "eval" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"eval_{name}"] = module
    spec.loader.exec_module(module)
    return module


cs = load("capture_sets")
cr = load("capture_r1")
SIZE = (1024, 683)


def rng() -> random.Random:
    return random.Random(1)  # noqa: S311 - semilla fija de un test, no criptografía


def aspect(window: tuple[int, int, int, int]) -> float:
    return (window[2] - window[0]) / (window[3] - window[1])


# --- ventanas ----------------------------------------------------------------------------------------


def test_general_deja_el_vehiculo_casi_entero_en_una_ventana_16_9() -> None:
    car = (150.0, 200.0, 850.0, 560.0)

    found = cs.general_window(car, SIZE, rng())

    assert found is not None
    window, fraction = found
    assert abs(aspect(window) - 16 / 9) < 0.03
    assert 0 <= window[0] and window[2] <= SIZE[0] and 0 <= window[1] and window[3] <= SIZE[1]
    assert cs.GENERAL_CAR_WIDTH[0] - 0.02 <= (car[2] - car[0]) / (window[2] - window[0]) <= 1.0
    assert cs.GENERAL_CAR_WIDTH[0] <= fraction <= cs.GENERAL_CAR_WIDTH[1]


def test_general_no_arma_ventana_si_el_vehiculo_es_mas_ancho_que_la_foto() -> None:
    assert cs.general_window((0.0, 100.0, 1024.0, 600.0), SIZE, rng()) is None


def test_plate_pone_la_patente_en_el_ancho_pedido_y_el_vehiculo_entero() -> None:
    car = (200.0, 150.0, 820.0, 560.0)
    plate = (440.0, 420.0, 560.0, 460.0)  # 120 px: 19 % del vehículo

    found = cs.plate_window(plate, car, SIZE, rng())

    assert found is not None
    window, fraction = found
    width = window[2] - window[0]
    assert cs.PLATE_WIDTH[0] - 0.01 <= (plate[2] - plate[0]) / width <= cs.PLATE_WIDTH[1] + 0.01
    assert cs.PLATE_WIDTH[0] - 0.01 <= fraction <= cs.PLATE_WIDTH[1] + 0.01
    assert window[0] - 0.05 * width <= car[0] and car[2] <= window[2] + 0.05 * width
    assert window[3] >= car[3] - 0.10 * (window[3] - window[1])


def test_plate_descarta_si_la_ventana_no_entra_en_la_foto() -> None:
    plate = (500.0, 300.0, 520.0, 310.0)  # 20 px: la ventana al 8-15 % sería de 130 a 250 px: muy chica

    assert cs.plate_window(plate, (400.0, 200.0, 620.0, 340.0), SIZE, rng()) is None


def test_detail_cae_casi_entero_dentro_de_la_caja_del_vehiculo() -> None:
    car = (100.0, 150.0, 950.0, 600.0)

    found = cs.detail_window(car, SIZE, rng())

    assert found is not None
    window, _ = found
    assert cs.overlap_fraction(window, car) >= cs.DETAIL_MIN_INSIDE
    assert (
        cs.MIN_WINDOW_PX["detail"] <= window[2] - window[0] <= (car[2] - car[0]) * cs.DETAIL_WINDOW_OF_CAR[1]
    )


def test_detail_no_arma_ventana_en_un_vehiculo_chico() -> None:
    assert cs.detail_window((100.0, 100.0, 400.0, 250.0), SIZE, rng()) is None


def test_overlap_fraction() -> None:
    assert cs.overlap_fraction((0, 0, 10, 10), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert cs.overlap_fraction((0, 0, 10, 10), (5.0, 0.0, 15.0, 10.0)) == 0.5
    assert cs.overlap_fraction((0, 0, 10, 10), (20.0, 20.0, 30.0, 30.0)) == 0.0


# --- salida --------------------------------------------------------------------------------------------


def test_el_jpeg_de_salida_tiene_lado_mayor_1600_y_pesa_menos_de_900_kb() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (683, 1024, 3), dtype=np.uint8)

    data = cs.crop_to_jpeg(noise, (100, 100, 740, 460))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    assert max(decoded.shape[:2]) == cs.OUT_LONG_SIDE
    assert len(data) <= cs.MAX_BYTES
    assert abs(decoded.shape[1] / decoded.shape[0] - 16 / 9) < 0.03


def test_una_foto_entera_chica_no_se_agranda() -> None:
    small = np.zeros((300, 400, 3), np.uint8)

    decoded = cv2.imdecode(np.frombuffer(cs.crop_to_jpeg(small, None), np.uint8), cv2.IMREAD_COLOR)

    assert max(decoded.shape[:2]) == cs.OUT_LONG_SIDE or decoded.shape[:2] == (300, 400)


def test_las_mitades_son_deterministas_y_parejas() -> None:
    first = cs.halves(401, 0)

    assert first == cs.halves(401, 0)
    assert first.count("A") == 200 and first.count("B") == 201
    assert first != cs.halves(401, 1)


# --- armado de los conjuntos -----------------------------------------------------------------------------


def pool() -> object:
    box = cs.Box
    boxes = {
        "carro": [box("Car", 0.1, 0.3, 0.9, 0.9)],
        "camion": [box("Truck", 0.1, 0.3, 0.9, 0.9)],
        "camion_y_auto": [box("Truck", 0.1, 0.3, 0.9, 0.9), box("Car", 0.0, 0.0, 0.1, 0.1)],
        "viejo": [box("Car", 0.1, 0.3, 0.9, 0.9)],
    }
    return cs.Pool(
        boxes=boxes,
        presence={"carro", "camion", "camion_y_auto", "viejo", "ambulancia"},
        cars={"carro", "camion_y_auto", "viejo"},
        family={"rueda"},
        annotated={
            "carro",
            "camion",
            "camion_y_auto",
            "viejo",
            "ambulancia",
            "gato",
            "rueda",
            "perro",
            "usado",
        },
    )


def test_plan_no_usa_fotos_de_entrenamiento_y_separa_los_conjuntos() -> None:
    planned = cs.plan(pool(), {"viejo", "usado"}, rng())

    everything = [i for ids in planned.values() for i in ids if not i.startswith("neg")]
    assert "viejo" not in everything and len(everything) == len(set(everything))
    assert planned["neg"] and set(planned["neg"]) <= {
        "gato",
        "rueda",
        "perro",
    }  # sin presencia ni entrenamiento
    assert "camion_y_auto" not in planned["heavy_Truck"]  # un pesado con un auto anotado no es "sin Car"
    assert planned["heavy_Truck"] == ["camion"]


def test_make_specs_marca_las_negativas_con_familia_vehiculo() -> None:
    sizes = {"gato": SIZE, "rueda": SIZE}

    out = cs.make_specs(pool(), {"neg": ["gato", "rueda"]}, sizes, rng())

    assert [(s.image_id, s.detail["family"]) for s in out["neg"]] == [("gato", False), ("rueda", True)]
    assert all(s.window is None for s in out["neg"])


def test_make_specs_arma_general_y_detalle_de_un_auto_grande() -> None:
    boxes = {"auto": [cs.Box("Car", 0.10, 0.25, 0.90, 0.90)]}
    p = cs.Pool(boxes, {"auto"}, {"auto"}, set(), {"auto"})

    out = cs.make_specs(p, {"general": ["auto"], "detail": ["auto"]}, {"auto": SIZE}, rng())

    assert len(out["general"]) == 1 and len(out["detail"]) == 1
    assert out["detail"][0].window is not None and out["general"][0].window is not None


# --- negativas por principio y conjuntos extra --------------------------------------------------------------


def tagged_pool() -> object:
    box = cs.Box
    tags = {
        "auto": {"Car"},
        "taxi": {"Taxi"},
        "furgon": {"Van"},
        "camion": {"Truck"},
        "colectivo": {"Bus"},
        "colectivo_y_auto": {"Bus", "Car"},
        "ambulancia": {"Ambulance"},
        "moto": {"Motorcycle", "Wheel"},  # una moto con su rueda: es negativa, no ambigua
        "rueda": {"Wheel"},
        "patente": {cs.PLATE_CLASS},
        "generico": {"Vehicle"},
        "gato": set(),
        "usado": {"Bus"},
        "pickup1": {"Truck"},
        "pickup2": {"Car"},
        "pickup3": {"Car"},
    }
    boxes = {
        "colectivo": [box("Bus", 0.1, 0.35, 0.9, 0.85)],
        "ambulancia": [box("Ambulance", 0.1, 0.35, 0.9, 0.85)],
        "colectivo_y_auto": [box("Bus", 0.1, 0.35, 0.9, 0.85), box("Car", 0.0, 0.0, 0.2, 0.2)],
        "pickup1": [box("Truck", 0.1, 0.35, 0.9, 0.85)],
        "pickup2": [
            box("Car", 0.1, 0.35, 0.9, 0.85),
            box("Car", 0.0, 0.0, 0.1, 0.1),
        ],  # una grande y una chica
        "pickup3": [box("Car", 0.1, 0.35, 0.9, 0.85), box("Car", 0.1, 0.35, 0.8, 0.85)],  # dos parecidas
        "usado": [box("Bus", 0.1, 0.35, 0.9, 0.85)],
    }
    return cs.Pool(boxes, set(), set(), set(), set(tags), tags)


def test_classify_aplica_la_definicion_de_foto_sin_auto() -> None:
    for kind_tags in ({"Car"}, {"Limousine"}, {"Taxi"}, {"Van"}, {"Truck"}, {"Bus", "Car"}):
        assert cs.classify(kind_tags) == cs.EXCLUDED_KIND
    for kind_tags in (
        {"Bus"},
        {"Ambulance"},
        {"Motorcycle"},
        {"Train"},
        {"Boat"},
        {"Airplane"},
        {"Wheelchair"},
    ):
        assert cs.classify(kind_tags) == cs.OTHER_KIND
    assert cs.classify({"Motorcycle", "Wheel"}) == cs.OTHER_KIND  # hay un vehículo concreto
    for kind_tags in ({"Wheel"}, {"Tire"}, {cs.PLATE_CLASS}, {"Vehicle"}, {"Land vehicle"}):
        assert cs.classify(kind_tags) == cs.AMBIGUOUS_KIND
    assert cs.classify(set()) == cs.CLEAN_KIND


def test_negative_counts_cuenta_las_excluidas_por_etiqueta() -> None:
    counts = cs.negative_counts(tagged_pool(), {"usado"})

    assert counts["fotos_anotadas_no_usadas_en_entrenamiento"] == 15
    assert counts["por_etiqueta_excluida"] == {"Car": 4, "Limousine": 0, "Taxi": 1, "Van": 1, "Truck": 2}
    assert counts["excluidas_total"] == 8 == sum(counts["excluidas_por_la_primera"].values())
    assert counts["por_tipo"][cs.OTHER_KIND] == 3 and counts["por_tipo"][cs.AMBIGUOUS_KIND] == 3
    assert counts["negativas_con_otro_vehiculo_por_etiqueta"] == {"Bus": 1, "Ambulance": 1, "Motorcycle": 1}
    assert counts["ambiguas_por_etiqueta"]["Wheel"] == 1


def test_plan_extras_no_mezcla_conjuntos_ni_usa_fotos_de_entrenamiento() -> None:
    planned = cs.plan_extras(tagged_pool(), {"usado"}, {"pickup1"}, {"pickup1", "pickup2", "pickup3"}, rng())

    assert planned["colectivo"] == [
        "colectivo"
    ]  # `colectivo_y_auto` tiene un auto; `usado` es de entrenamiento
    assert planned["ambulancia"] == ["ambulancia"]
    assert "pickup1" not in planned["pickup"]  # ya está en otro conjunto
    assert set(planned["pickup"]) == {"pickup2", "pickup3"}
    assert set(planned["sinvehiculo"]) == {"moto", "rueda", "patente", "generico", "gato"}
    everything = [i for ids in planned.values() for i in ids]
    assert len(everything) == len(set(everything))


def test_pickup_box_pide_una_caja_de_vehiculo_dominante() -> None:
    box = cs.Box
    big = box("Truck", 0.1, 0.25, 0.9, 0.9)

    chosen = cs.pickup_box([big, box("Car", 0.0, 0.0, 0.1, 0.1)], SIZE)

    assert chosen is not None and chosen[0].label == "Truck"
    assert cs.pickup_box([box("Car", 0.1, 0.25, 0.9, 0.9), box("Car", 0.1, 0.25, 0.8, 0.9)], SIZE) is None
    assert cs.pickup_box([box("Bus", 0.1, 0.25, 0.9, 0.9)], SIZE) is None  # un colectivo no es una pickup
    assert cs.pickup_box([], SIZE) is None


def test_make_extra_specs_arma_cada_conjunto() -> None:
    p = tagged_pool()
    sizes = {i: SIZE for i in ("colectivo", "ambulancia", "pickup1", "pickup3", "gato", "rueda")}
    ids = {
        "colectivo": ["colectivo"],
        "ambulancia": ["ambulancia"],
        "pickup": ["pickup1", "pickup3"],
        "sinvehiculo": ["gato", "rueda"],
    }

    out = cs.make_extra_specs(p, ids, sizes, rng())

    assert [s.image_id for s in out["pickup"]] == ["pickup1"]  # pickup3 no tiene un vehículo dominante
    assert out["pickup"][0].detail["caja"] == "Truck" and out["pickup"][0].window is not None
    assert out["colectivo"][0].window is not None and out["ambulancia"][0].window is not None
    assert [(s.image_id, s.detail["tipo"]) for s in out["sinvehiculo"]] == [
        ("gato", cs.CLEAN_KIND),
        ("rueda", cs.AMBIGUOUS_KIND),
    ]
    assert all(s.window is None for s in out["sinvehiculo"])


def test_la_etiqueta_de_imagen_se_busca_por_nombre_y_confianza(tmp_path: Path) -> None:
    descriptions = tmp_path / "d.csv"
    descriptions.write_text("/m/0cvq3,Pickup truck\n/m/0abc,Pickup truck racing\n", encoding="utf-8")
    labels = tmp_path / "l.csv"
    labels.write_text(
        "ImageID,Source,LabelName,Confidence\na,x,/m/0cvq3,1\nb,x,/m/0cvq3,0\nc,x,/m/0abc,1\nd,x,/m/0cvq3,1\n",
        encoding="utf-8",
    )

    assert cs.class_id(descriptions, "Pickup truck") == "/m/0cvq3"
    assert cs.class_id(descriptions, "Pickup truck inexistente") is None
    assert cs.images_with_label(labels, "/m/0cvq3") == {"a", "d"}


# --- R1 y el análisis -------------------------------------------------------------------------------------


def sig(
    car: list[list[float]] | None = None,
    seg: list[list[float]] | None = None,
    truck: list[list[float]] | None = None,
    van: list[list[float]] | None = None,
) -> dict[str, object]:
    return {"det": {"car": car or [], "truck": truck or [], "van": van or []}, "seg": seg or []}


def test_accepted_pide_confianza_y_area_y_alguna_de_las_dos_senales() -> None:
    assert cr.accepted(sig(car=[[0.6, 0.30]]), 0.5, None)
    assert not cr.accepted(sig(car=[[0.45, 0.30]]), 0.5, None)  # poca confianza
    assert not cr.accepted(sig(car=[[0.9, 0.04]]), 0.5, None)  # menos del 5 % de la foto
    assert cr.accepted(sig(seg=[[0.7, 0.2]]), 0.9, 0.6)  # el segmentador alcanza (det ∨ seg)
    assert not cr.accepted(sig(seg=[[0.7, 0.2]]), 0.9, None)  # segmentador apagado


def test_accepted_cuenta_camion_y_van_y_solo_las_clases_pedidas() -> None:
    assert cr.accepted(sig(truck=[[0.8, 0.3]]), 0.5, None)
    assert cr.accepted(sig(van=[[0.8, 0.3]]), 0.5, None)
    assert not cr.accepted(sig(truck=[[0.8, 0.3]]), 0.5, None, classes=("car", "van"))  # sin camión


def test_passing_dice_que_dejo_pasar_la_foto() -> None:
    row = sig(car=[[0.9, 0.3]], truck=[[0.2, 0.3]], seg=[[0.8, 0.25]])

    found = cr.passing(row, 0.4, 0.5)

    assert found == [("detector", "car", 0.9, 0.3), ("segmentador", "car", 0.8, 0.25)]
    assert cr.passing(row, 0.4, None) == [("detector", "car", 0.9, 0.3)]
    assert cr.passing(sig(), 0.4, 0.5) == []


def test_u3_supone_independencia() -> None:
    rates = {name: {"tasa": 0.1, "cota_95": 0.2} for name in cr.VALID}

    assert cr.u3(rates, "tasa") == round(1 - 0.9**3, 4)
    assert cr.u3(rates, "cota_95") == round(1 - 0.8**3, 4)


def synthetic() -> dict[str, list[dict[str, object]]]:
    good = sig(car=[[0.9, 0.5]])
    weak = sig(car=[[0.35, 0.5]])  # solo pasa con un umbral bajo
    negative_hit = sig(car=[[0.38, 0.4]])
    out: dict[str, list[dict[str, object]]] = {
        "general": [good] * 18 + [weak] * 2,
        "plate": [good] * 18 + [weak] * 2,
        "detail": [good] * 18 + [weak] * 2,
        **{h: [good] * 4 for h in cr.HEAVY},
        "pickup": [good] * 3 + [sig(truck=[[0.6, 0.4]])],
        "colectivo": [sig()] * 3 + [good],
        "ambulancia": [good] * 4,
        "sinvehiculo": [sig()] * 38 + [negative_hit] * 2,
    }
    for name in (*cr.HEAVY, cr.PICKUP, *cr.NOT_VEHICLES):
        out[f"{name}_whole"] = list(out[name])
    return out


def test_evaluate_cuenta_s1_rechazo_u3_pesados_y_otros_vehiculos() -> None:
    signals = synthetic()
    ambiguous = [False] * 40

    result = cr.evaluate(signals, ambiguous, 0.5, None)

    assert result["negativas"]["todas"]["fallas"] == 0
    assert result["validas"]["general"]["todas"]["fallas"] == 2
    assert result["u3"]["todas"]["estimada"] == round(1 - 0.9**3, 4)
    assert result["pesados"]["camion_van"]["todas"] == cr.rate(0, 8)
    assert (
        result["pesados"]["pickup"]["todas"]["fallas"] == 0 and result["pesados"]["pickup"]["todas"]["n"] == 4
    )
    assert result["otros_vehiculos"]["colectivo"]["todas"]["fallas"] == 3  # rechazadas: lo que se espera
    assert result["otros_vehiculos"]["ambulancia"]["todas"]["fallas"] == 0  # la ambulancia se acepta como van

    loose = cr.evaluate(signals, ambiguous, 0.30, None)

    assert loose["negativas"]["todas"]["fallas"] == 2
    assert loose["validas"]["general"]["todas"]["fallas"] == 0


def test_evaluate_reporta_s1_sin_las_ambiguas() -> None:
    signals = synthetic()
    ambiguous = [False] * 39 + [True]  # la última negativa (una de las dos que se aceptan) es ambigua

    result = cr.evaluate(signals, ambiguous, 0.30, None)

    assert result["negativas"]["todas"]["fallas"] == 2
    assert result["negativas"]["todas_sin_ambiguas"] == cr.rate(1, 39)
    halves_n = result["negativas"]["A_sin_ambiguas"]["n"] + result["negativas"]["B_sin_ambiguas"]["n"]
    assert halves_n == 39


def test_accepted_details_dice_que_dejo_pasar_cada_negativa_de_la_mitad() -> None:
    labels = cs.halves(20, cs.SEED)
    in_b = [i for i, h in enumerate(labels) if h == "B"]
    rows = [sig()] * 20
    rows[in_b[0]] = sig(car=[[0.6, 0.3]], seg=[[0.7, 0.2]])
    rows[in_b[1]] = sig(seg=[[0.9, 0.3]])
    rows[next(i for i, h in enumerate(labels) if h == "A")] = sig(car=[[0.9, 0.5]])  # en A: no se cuenta en B
    specs = [
        {"detail": {"tipo": "ambigua" if i == in_b[1] else "sin_vehiculo", "etiquetas": ["Wheel"]}}
        for i in range(20)
    ]

    found = cr.accepted_details(rows, specs, "B", 0.4, 0.5)

    assert found["aceptadas"] == 2 and found["ambiguas"] == 1
    assert found["por_decisor"] == {"ambos": 1, "segmentador": 1}
    assert found["clases_del_detector"] == {"car": 1}
    by_index = {f["indice"]: f for f in found["fotos"]}
    assert by_index[in_b[0]]["decide"] == "ambos" and by_index[in_b[0]]["detector"][0]["clase"] == "car"
    assert by_index[in_b[1]]["decide"] == "segmentador" and by_index[in_b[1]]["tipo"] == "ambigua"


def test_best_elige_el_de_menor_u3_con_la_cota_de_s1_en_la_mitad_a() -> None:
    signals = synthetic()
    configs = cr.sweep(signals, [False] * 40)

    chosen = cr.best(configs, bound=0.30)  # con N chico la cota es ancha: alcanza para ver la elección

    assert chosen is not None and chosen["t_car"] is not None
    assert chosen["negativas"]["A"]["cota_95"] <= 0.30
    others = [
        c
        for c in configs
        if c["t_car"] is not None and c["t_seg"] is not None and c["negativas"]["A"]["cota_95"] <= 0.30
    ]
    assert chosen["u3"]["A"]["estimada"] == min(c["u3"]["A"]["estimada"] for c in others)


def test_best_con_el_segmentador_fijo_solo_barre_el_detector() -> None:
    configs = cr.sweep(synthetic(), [False] * 40)

    chosen = cr.best(configs, seg=0.50, bound=0.99)

    assert chosen is not None and chosen["t_seg"] == 0.50


def test_best_puede_exigir_la_cota_solo_sobre_las_negativas_sin_ambiguas() -> None:
    signals = synthetic()
    labels = cs.halves(200, cs.SEED)
    in_a = [i for i, h in enumerate(labels) if h == "A"][:2]
    negatives = [sig()] * 200
    for i in in_a:
        negatives[i] = sig(car=[[0.99, 0.9]])  # se acepta con cualquier umbral
    signals["sinvehiculo"] = negatives
    ambiguous = [i in in_a for i in range(200)]  # las dos aceptadas son ambiguas

    configs = cr.sweep(signals, ambiguous)

    assert cr.best(configs, bound=0.05) is None  # 2 de 100 en A: la cota supera el 5 %
    assert cr.best(configs, bound=0.05, negatives="A_sin_ambiguas") is not None  # 0 de 98


def test_best_sin_ninguna_que_cumpla_devuelve_none() -> None:
    signals = synthetic()
    signals["sinvehiculo"] = [sig(car=[[0.99, 0.9]])] * 40  # todas se aceptan con cualquier umbral

    assert cr.best(cr.sweep(signals, [False] * 40)) is None


def test_u2_es_el_rechazo_de_una_alerta_de_dos_fotos_general_y_patente() -> None:
    rates = {
        "general": {"tasa": 0.10, "cota_95": 0.20},
        "plate": {"tasa": 0.05, "cota_95": 0.10},
        "detail": {"tasa": 0.50, "cota_95": 0.60},  # el detalle no cuenta en una alerta de 2 fotos
    }

    assert cr.u3(rates, "tasa", cr.ALERT_2) == round(1 - 0.90 * 0.95, 4)
    assert cr.u3(rates, "cota_95", cr.ALERT_2) == round(1 - 0.80 * 0.90, 4)


def test_evaluate_incluye_u2_sin_el_detalle() -> None:
    signals = synthetic()
    signals["detail"] = [sig()] * 20  # el detalle se rechaza siempre

    result = cr.evaluate(signals, [False] * 40, 0.5, None)

    assert result["u3"]["todas"]["estimada"] > 0.9
    assert result["u2"]["todas"]["estimada"] == round(1 - 0.9 * 0.9, 4)  # general y patente rechazan 2 de 20
