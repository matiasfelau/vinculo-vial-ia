"""tools/eval/plate_zoom.py y plate_rule.py: ventanas de zoom, variantes P1 y P2 y U3 por alerta."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))
import plate_rule as rule  # noqa: E402
import plate_zoom as zoom  # noqa: E402
import presence_grid as grid  # noqa: E402

CAR = {"car": [[0.9, 0.5]]}
PLATE = {"plate": [[0.9, 0.3]]}
SEG = {"seg": [[0.9, 0.5]]}
BOTH = {"car": [[0.9, 0.5]], "plate": [[0.9, 0.3]]}
NOTHING: dict[str, list[list[float]]] = {}


def ref(
    width: int = 2000, height: int = 1500, box: tuple[float, float, float, float] = (900, 700, 200, 60)
) -> Any:
    return zoom.PlateRef("test", "t-0", Path("x.jpg"), (width, height), box)


# --- plate_zoom -------------------------------------------------------------------------------------


def test_la_ventana_deja_a_la_patente_con_el_ancho_pedido_y_en_4_a_3() -> None:
    window = zoom.crop_window(ref(), 0.25)

    assert window is not None
    x1, y1, x2, y2 = window
    assert (x2 - x1, y2 - y1) == (800, 600)  # 200 / 0,25 de ancho, 4:3
    assert x1 <= 900 and x2 >= 1100  # la patente (x 900 a 1100) cabe adentro
    assert 200 / (x2 - x1) == pytest.approx(0.25)


def test_la_ventana_esta_centrada_en_la_patente_y_se_corre_si_se_sale_de_la_imagen() -> None:
    centered = zoom.crop_window(ref(), 0.4)
    corner = zoom.crop_window(ref(box=(5, 5, 200, 60)), 0.4)

    assert centered is not None and corner is not None
    assert (centered[0] + centered[2]) / 2 == pytest.approx(1000, abs=1)
    assert corner[0] == 0 and corner[1] == 0  # se corrió hasta que entre


@pytest.mark.parametrize(
    ("kwargs", "z"),
    [
        ({"width": 700}, 0.25),  # la ventana de 800 px no cabe a lo ancho
        ({"height": 500}, 0.25),  # ni a lo alto
        ({"box": (100, 100, 40, 12)}, 0.60),  # 67 px de ancho: menos del mínimo
    ],
)
def test_si_la_ventana_no_cabe_o_es_muy_chica_no_hay_foto(kwargs: dict[str, Any], z: float) -> None:
    assert zoom.crop_window(ref(**kwargs), z) is None


def test_solo_se_usan_las_patentes_que_admiten_los_tres_zooms() -> None:
    fits = ref()
    too_wide = ref(width=700)  # no admite el 25 %, aunque sí el 60 %

    assert zoom.common_refs([fits, too_wide]) == [fits]


def test_zoom_key_y_recorte() -> None:
    assert [zoom.zoom_key(z) for z in zoom.ZOOMS] == ["plate_z25", "plate_z40", "plate_z60"]
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)

    cut = zoom.crop(image, (2, 1, 8, 5))

    assert cut.shape == (4, 6, 3) and cut.flags["C_CONTIGUOUS"]
    assert (cut == image[1:5, 2:8]).all()


def test_coco_refs_usa_ids_sin_nombres_de_archivo_y_descarta_patentes_diminutas(tmp_path: Path) -> None:
    import json

    folder = tmp_path / "valid"
    folder.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "AB123CD.jpg", "width": 1000, "height": 800}],
        "categories": [{"id": 1, "name": "plate"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 30]},
            {"id": 8, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 8]},  # < 16 px
            {"id": 9, "image_id": 1, "category_id": 2, "bbox": [0, 0, 500, 300]},  # no es patente
        ],
    }
    (folder / zoom.ANNOTATIONS).write_text(json.dumps(coco))

    refs = zoom.coco_refs(folder, "openimages-test", "plate")

    assert [r.ident for r in refs] == ["openimages-test-valid-7"]
    assert "AB123CD" not in refs[0].ident


# --- plate_rule ---------------------------------------------------------------------------------------


def feats(*photos: dict[str, list[list[float]]]) -> Any:
    return grid.features(list(photos))


def run(
    role: str, photos: list[dict[str, list[list[float]]]], decider: str = "or", variant: str = "P1"
) -> list[bool]:
    return rule.accept(feats(*photos), role, decider, variant, 0.4, 0.4, 0.5).tolist()


def test_en_general_y_detail_una_patente_sola_no_alcanza() -> None:
    photos = [PLATE, CAR, SEG, NOTHING]

    assert run("general", photos) == [False, True, True, False]
    assert run("detail", photos) == [False, True, True, False]


def test_p1_en_plate_basta_un_auto_o_una_patente() -> None:
    assert run("plate", [PLATE, CAR, SEG, NOTHING], variant="P1") == [True, True, True, False]


def test_p2_en_plate_pide_patente_y_auto() -> None:
    assert run("plate", [PLATE, CAR, SEG, BOTH, {**PLATE, **SEG}, NOTHING], variant="P2") == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_los_decisores_con_p2() -> None:
    photos = [{**PLATE, **CAR}, {**PLATE, **SEG}, {**PLATE, **CAR, **SEG}]

    assert run("plate", photos, "det", "P2") == [True, False, True]
    assert run("plate", photos, "seg", "P2") == [False, True, True]
    assert run("plate", photos, "and", "P2") == [False, False, True]


def test_el_umbral_de_auto_se_puede_bajar_a_0_30() -> None:
    weak = feats({"car": [[0.35, 0.5]], "plate": [[0.9, 0.3]]})

    assert not rule.accept(weak, "plate", "det", "P2", 0.40, 0.4, 0.5)[0]
    assert rule.accept(weak, "plate", "det", "P2", 0.30, 0.4, 0.5)[0]
    assert rule.THRESHOLDS[0] == 0.30 and rule.THRESHOLDS[-1] == 0.95 and len(rule.THRESHOLDS) == 14


def test_u3_por_alerta_supone_independencia() -> None:
    assert rule.alert_rejection(0.0, 0.0, 0.0) == 0.0
    assert rule.alert_rejection(0.1, 0.0, 0.0) == pytest.approx(0.1)
    assert rule.alert_rejection(0.1, 0.2, 0.3) == pytest.approx(1 - 0.9 * 0.8 * 0.7)
    assert rule.alert_rejection(1.0, 0.2, 0.3) == 1.0


def scores() -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[str]]]:
    base = {
        "general": [CAR, CAR, NOTHING, SEG],  # 1 rechazada
        "plate": [PLATE, BOTH],
        "detail": [CAR, NOTHING],  # 1 rechazada
        "negativas": [NOTHING, NOTHING],
        "negativas_nuevas": [PLATE, CAR, NOTHING, {"car": [[0.5, 0.5]]}],
    }
    zooms = {
        "plate_z25": [PLATE, BOTH, NOTHING],
        "plate_z40": [BOTH, BOTH, PLATE],
        "plate_z60": [BOTH, CAR, BOTH],
    }
    names = {"negativas": ["n0", "n1"], "negativas_nuevas": ["a", "b", "c", "381d3671020dca70"]}
    return base, zooms, names


def test_sweep_cuenta_rechazos_por_zoom_negativas_por_rol_y_u3() -> None:
    base, zooms, names = scores()

    rows = rule.sweep(base, zooms, names)
    p1 = next(r for r in rows if r["decisor"] == "or" and r["regla"] == "P1" and rule.is_current(r))
    p2 = next(r for r in rows if r["decisor"] == "or" and r["regla"] == "P2" and rule.is_current(r))

    assert p1["rechazadas"]["general"] == 1 and p1["rechazadas"]["detail"] == 1
    assert (
        p1["rechazadas"]["plate_z25"] == 1 and p2["rechazadas"]["plate_z25"] == 2
    )  # P2 rechaza la patente sola
    assert p2["rechazadas"]["plate_z60"] == 1  # el auto sin patente
    assert (
        p1["negativas_aceptadas"]["plate"]["nuevas"] == 3
        and p2["negativas_aceptadas"]["plate"]["nuevas"] == 0
    )
    assert p1["negativas_aceptadas"]["general"]["nuevas"] == 2  # la patente sola ya no pasa en GENERAL
    assert p1["negativas_aceptadas"]["general"]["nuevas_sin_fondo"] == 1  # una es la del auto real al fondo
    assert p1["u3_por_alerta"]["plate_z25"] == pytest.approx(
        rule.alert_rejection(1 / 4, 1 / 3, 1 / 2), abs=1e-4
    )
    assert p1["u3_medio_zooms"] is not None
    assert p1["cumple_s1"]  # con tan pocas fotos el tope de 5 no se alcanza


def test_las_negativas_excluidas_por_revision_se_sacan_de_nuevas_sin_fondo_y_de_vistas() -> None:
    base, zooms, names = scores()
    base["negativas"] = [CAR, NOTHING]
    names["negativas"] = ["vista-con-camion", "limpia"]
    names["negativas_nuevas"] = ["a", "b", "c", "sin-auto-pero-con-un-colectivo"]

    rows = rule.sweep(
        base, zooms, names, excluded_stems=["vista-con-camion", "sin-auto-pero-con-un-colectivo"]
    )
    row = next(r for r in rows if r["decisor"] == "or" and r["regla"] == "P1" and rule.is_current(r))

    assert row["negativas_aceptadas"]["general"]["vistas"] == 1
    assert row["negativas_aceptadas"]["general"]["vistas_sin_excluidas"] == 0
    assert row["negativas_aceptadas"]["general"]["nuevas"] == 2  # a (patente no) y b: el auto y la de 0,5
    assert row["negativas_aceptadas"]["general"]["nuevas_sin_fondo"] == 1  # sin la del colectivo


def test_la_mejor_que_cumple_s1_minimiza_u3_y_no_existe_si_ninguna_cumple() -> None:
    base, zooms, names = scores()
    rows = rule.sweep(base, zooms, names)

    chosen = rule.best([r for r in rows if r["decisor"] == "or" and r["regla"] == "P2"], "promedio")

    assert chosen is not None and chosen["cumple_s1"]
    assert all(
        r["u3_medio_zooms"] >= chosen["u3_medio_zooms"]
        for r in rows
        if r["decisor"] == "or" and r["regla"] == "P2" and r["cumple_s1"]
    )
    assert rule.best([{**rows[0], "cumple_s1": False}], "promedio") is None
    table = rule.best_table(rows)
    assert len(table) == len(rule.DECIDERS) * len(rule.RULES) * 4


def test_el_area_minima_de_la_patente_cambia_el_rechazo_de_los_zooms_chicos() -> None:
    small_plate = {"plate": [[0.9, 0.03]], "car": [[0.9, 0.5]]}  # una patente del 3 % de la foto, y un auto
    base = {"plate": [small_plate]}

    out = rule.area_sensitivity(base, {"plate_z25": [small_plate]}, floors=(0.05, 0.01))
    by = {
        (r["area_minima_de_la_patente"], r["regla"]): r["rechazadas"]["plate_z25"]["rechazadas"] for r in out
    }

    assert by[(0.05, "P1")] == 0 and by[(0.05, "P2")] == 1  # con el 5 % la patente no cuenta: P2 la rechaza
    assert by[(0.01, "P2")] == 0


def test_main_rechaza_salidas_en_el_repo_y_senales_faltantes(tmp_path: Path) -> None:
    import json

    base, zooms, _ = scores()
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"senales": base, "senales_faltantes": {}}))
    zoom_report = tmp_path / "z.json"
    zoom_report.write_text(json.dumps({"senales": zooms, "senales_faltantes": {"plate_z25": 1}}))
    args = [
        "--report",
        str(report),
        "--zoom-report",
        str(zoom_report),
        "--data-dir",
        str(tmp_path),
        "--fresh-negatives",
        str(tmp_path),
    ]

    assert rule.main([*args, "--out-dir", str(REPO_ROOT / "salida-de-prueba")]) == 1
    assert rule.main([*args, "--out-dir", str(tmp_path / "out")]) == 2
