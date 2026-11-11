"""tools/eval/presence_grid.py: barrido de R1 (decisor, patente y umbrales) sobre señales ya medidas."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))
import presence_grid as grid  # noqa: E402

CAR = {"car": [[0.9, 0.5]]}
PLATE = {"plate": [[0.9, 0.3]]}
SEG = {"seg": [[0.9, 0.5]]}
NOTHING: dict[str, list[list[float]]] = {}


def feats(*photos: dict[str, list[list[float]]]) -> Any:
    return grid.features(list(photos))


def test_features_toma_la_mejor_confianza_con_area_minima() -> None:
    out = feats({"car": [[0.9, 0.04], [0.6, 0.06]], "plate": [[0.5, 0.01]], "seg": [[0.7, 0.2]]})

    assert out.tolist() == [[0.6, 0.0, 0.7]]


def test_los_cuatro_decisores() -> None:
    both, only_det, only_seg, none = feats({**CAR, **SEG}, CAR, SEG, NOTHING)

    def run(decider: str) -> list[bool]:
        return grid.accept(
            np.array([both, only_det, only_seg, none]), "general", decider, "ninguna", 0.4, 0.4, 0.5
        ).tolist()

    assert run("det") == [True, True, False, False]
    assert run("seg") == [True, False, True, False]
    assert run("or") == [True, True, True, False]
    assert run("and") == [True, False, False, False]


def test_la_patente_suma_solo_en_los_roles_de_su_modo() -> None:
    only_plate = feats(PLATE)

    def accepted(mode: str, role: str) -> bool:
        return bool(grid.accept(only_plate, role, "det", mode, 0.4, 0.4, 0.5)[0])

    assert [accepted("todas", r) for r in grid.ROLES] == [True, True, True]
    assert [accepted("partes", r) for r in grid.ROLES] == [False, True, True]  # la regla actual del servicio
    assert [accepted("solo_plate", r) for r in grid.ROLES] == [False, True, False]
    assert [accepted("ninguna", r) for r in grid.ROLES] == [False, False, False]


def test_los_umbrales_de_r1_filtran_por_confianza() -> None:
    weak = feats({"car": [[0.45, 0.5]], "plate": [[0.45, 0.3]]})

    assert grid.accept(weak, "plate", "det", "partes", 0.40, 0.40, 0.5)[0]
    assert not grid.accept(weak, "plate", "det", "partes", 0.50, 0.50, 0.5)[0]


def test_grid_points_solo_varia_lo_que_importa() -> None:
    assert len(grid.grid_points("det", "ninguna")) == 12  # solo t_car
    assert len(grid.grid_points("det", "partes")) == 12 * 12
    assert len(grid.grid_points("seg", "ninguna")) == 10
    assert len(grid.grid_points("or", "partes")) == 12 * 12 * 10
    assert grid.T_CAR[0] == 0.40 and grid.T_CAR[-1] == 0.95 and grid.T_SEG[0] == 0.50


def scores() -> dict[str, list[dict[str, list[list[float]]]]]:
    return {
        "general": [CAR, CAR, SEG, NOTHING],
        "plate": [PLATE, CAR],
        "detail": [CAR, SEG, NOTHING],
        "negativas": [NOTHING, NOTHING],
        "negativas_nuevas": [CAR, PLATE, NOTHING, {"car": [[0.5, 0.5]], "seg": [[0.9, 0.5]]}],
    }


NAMES = {
    "general": ["g0", "g1", "g2", "g3"],
    "plate": ["recorte-0", "recorte-1"],
    "detail": ["d0", "d1", "d2"],
    "negativas": ["n0", "n1"],
    "negativas_nuevas": [
        "x0.jpg",
        "x1.jpg",
        "x2.jpg",
        "381d3671020dca70.jpg",
    ],  # la última: auto real al fondo
}


def test_sweep_cuenta_rechazos_y_aceptadas_por_rol_y_marca_las_de_fondo() -> None:
    rows = grid.sweep(scores(), NAMES)
    current = next(
        r for r in rows if r["decisor"] == "det" and r["patente"] == "partes" and grid.is_current(r)
    )

    assert current["general"]["rechazadas"] == 2  # la del segmentador y la vacía
    assert current["plate"]["rechazadas"] == 0 and current["detail"]["rechazadas"] == 2
    assert current["general"]["nuevas"] == 2  # auto en x0 y en la de fondo
    assert current["plate"]["nuevas"] == 3  # además la patente
    assert current["general"]["nuevas_sin_fondo"] == 1 and current["general"]["nuevas_sin_fondo_de"] == 3
    assert current["u3"] == 2 and current["validas_rechazadas"] == 4
    assert current["cumple_s1"]  # 3 aceptadas de 4 en el peor rol: con pocas fotos el tope de 5 no se alcanza


def test_cumple_s1_con_5_o_menos_y_su_version_sin_las_de_fondo() -> None:
    valid = {role: feats(CAR) for role in grid.ROLES}
    # 6 negativas aceptadas, una de ellas con auto real al fondo: 6 > 5, pero 5 sin ella
    negatives = {"negativas": feats(NOTHING), "negativas_nuevas": feats(*([CAR] * 6), NOTHING)}
    background = np.array([True] + [False] * 6)

    out = grid.evaluate(
        valid, negatives, decider="det", plate_mode="partes", thresholds=(0.4, 0.4, 0.5), excluded=background
    )

    assert not out["cumple_s1"] and out["cumple_s1_sin_fondo"]
    assert out["general"]["nuevas_sin_fondo_de"] == 6


def test_la_mejor_que_cumple_s1_rechaza_menos_validas() -> None:
    rows = grid.sweep(scores(), NAMES)

    best = grid.cheapest(rows, "cumple_s1")

    assert best is not None and best["cumple_s1"]
    assert all(r["validas_rechazadas"] >= best["validas_rechazadas"] for r in rows if r["cumple_s1"])
    assert grid.cheapest([{"cumple_s1": False}], "cumple_s1") is None


def test_frontier_solo_deja_puntos_que_bajan_los_rechazos() -> None:
    rows = grid.sweep(scores(), NAMES)

    points = grid.frontier(rows, "det", "partes")

    assert [p["validas_rechazadas"] for p in points] == sorted(
        (p["validas_rechazadas"] for p in points), reverse=True
    )
    assert [p["peor_rol_nuevas"] for p in points] == sorted(p["peor_rol_nuevas"] for p in points)


def test_per_photo_y_la_tabla_cruzada_separan_validas_y_sin_auto() -> None:
    photos = grid.per_photo(scores(), NAMES)
    table = grid.cross_table(photos)

    assert {p["conjunto"] for p in photos} == {"validas", "sin_auto"}
    assert next(p for p in photos if p["foto"] == "g2")["encendido"] == ["auto_segmentador"]
    assert next(p for p in photos if p["foto"] == "g3")["encendido"] == ["nada"]
    assert table["general"] == {
        "fotos": 4,
        "det_si_seg_si": 0,
        "det_si_seg_no": 2,
        "det_no_seg_si": 1,
        "det_no_seg_no": 1,
        "det_solo_por_patente": 0,
    }
    assert table["plate"]["det_solo_por_patente"] == 1  # en PLATE el detector también cuenta la patente
    assert table["negativas_nuevas"]["det_si_seg_si"] == 1


def test_accepted_negatives_dice_si_el_segmentador_tambien_vio_un_auto() -> None:
    found = {a["foto"]: a for a in grid.accepted_negatives(scores(), NAMES)}

    assert set(found) == {"x0.jpg", "x1.jpg", "381d3671020dca70.jpg"}
    assert found["x0.jpg"]["senal"] == "auto" and not found["x0.jpg"]["segmentador_vio_auto"]
    assert found["x1.jpg"]["senal"] == "patente"
    assert found["381d3671020dca70.jpg"]["segmentador_vio_auto"]
    assert found["381d3671020dca70.jpg"]["auto_real_al_fondo"] and not found["x0.jpg"]["auto_real_al_fondo"]


def test_main_rechaza_salidas_en_el_repo_e_informes_con_senales_faltantes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"senales": scores(), "senales_faltantes": {"detail": 1}}))
    args = ["--report", str(report), "--data-dir", str(tmp_path), "--fresh-negatives", str(tmp_path)]

    assert grid.main([*args, "--out-dir", str(REPO_ROOT / "salida-de-prueba")]) == 1
    assert grid.main([*args, "--out-dir", str(tmp_path / "out")]) == 2
