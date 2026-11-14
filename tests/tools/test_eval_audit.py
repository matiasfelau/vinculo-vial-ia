"""tools/eval: configuraciones de R1 (`presence_configs`) y hojas de contacto (`audit_sheets`)."""

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))
import audit_sheets  # noqa: E402
import presence  # noqa: E402
import presence_configs as configs  # noqa: E402

CAR_PHOTO = {"car": [[0.9, 0.5]]}
NOTHING: dict[str, list[list[float]]] = {}


def scores() -> dict[str, list[dict[str, list[list[float]]]]]:
    """Un caso chico: el segmentador salva un detalle, la rueda otro y una negativa pasa solo por rueda."""
    return {
        "general": [CAR_PHOTO, {"seg": [[0.9, 0.5]]}],
        "plate": [{"plate": [[0.9, 0.3]]}, {"wheel": [[0.6, 0.2]]}],
        "detail": [{"seg": [[0.9, 0.4]]}, {"wheel": [[0.5, 0.3]]}, {"car": [[0.8, 0.6]]}, NOTHING],
        "negativas": [NOTHING, NOTHING],
        "negativas_nuevas": [
            {"wheel": [[0.52, 0.1]]},
            {"car": [[0.5, 0.2]]},
            {"car": [[0.5, 0.02]]},
            NOTHING,
        ],
    }


NAMES = {
    "general": ["g0", "g1"],
    "plate": ["recorte-0", "recorte-1"],
    "detail": ["d0", "d1", "d2", "d3"],
    "negativas": ["n0", "n1"],
    "negativas_nuevas": ["x0", "x1", "x2", "x3"],
}


def test_el_segmentador_recupera_un_detalle_y_no_deja_pasar_ninguna_negativa_mas() -> None:
    result = configs.compare(scores(), NAMES)

    a, b = result["configuraciones"]["a_sin_segmentador"], result["configuraciones"]["b_con_segmentador"]

    assert a["roles"]["detail"]["rechazadas"] == 2  # el del segmentador y el sin nada
    assert b["roles"]["detail"]["rechazadas"] == 1
    assert a["U3"]["rechazadas"] == 2 and b["U3"]["rechazadas"] == 1
    assert result["mejor_de_a_y_b"] == "b_con_segmentador"
    assert (
        a["S1"]["plate"]["nuevas"] == b["S1"]["plate"]["nuevas"] == "2/4"
    )  # la rueda y el auto; el chico no pasa
    assert a["S1"]["general"]["nuevas"] == "1/4"  # general no usa rueda


def test_sin_rueda_quita_la_negativa_que_solo_pasaba_por_rueda_y_dice_que_senal_pasaba() -> None:
    result = configs.compare(scores(), NAMES)

    with_wheel = result["configuraciones"]["b_con_segmentador"]["aceptadas"]["plate"]
    without = result["configuraciones"]["c1_b_sin_rueda"]["aceptadas"]["plate"]

    assert [(x["foto"], [s["senal"] for s in x["senales"]]) for x in with_wheel] == [
        ("x0", ["wheel"]),
        ("x1", ["car"]),
    ]
    assert with_wheel[0]["senales"][0]["modelo"] == "detector (clase wheel)"
    assert [x["foto"] for x in without] == ["x1"]
    assert result["configuraciones"]["c1_b_sin_rueda"]["S1"]["detail"]["nuevas"] == "1/4"


def test_la_rueda_con_umbral_mas_alto_deja_de_pasar_y_el_barrido_no_incluye_el_valor_base() -> None:
    result = configs.compare(scores(), NAMES)

    assert "0.40" not in result["barrido_de_la_rueda"]
    assert result["barrido_de_la_rueda"]["0.50"]["S1"]["plate"]["nuevas"] == "2/4"  # 0,52 >= 0,50
    assert result["barrido_de_la_rueda"]["0.60"]["S1"]["plate"]["nuevas"] == "1/4"  # 0,52 < 0,60
    assert any(key.startswith("c2_b_rueda_") for key in result["configuraciones"])


def test_la_mejor_se_elige_primero_por_s1_y_despues_por_u3() -> None:
    def config(accepted: int, rejected: int) -> dict[str, Any]:
        roles = {r: {"negativas_nuevas": {"que_pasan": accepted}} for r in configs.ROLES}
        return {"roles": roles, "U3": {"rechazadas": rejected}}

    assert configs.badness(config(1, 9)) < configs.badness(
        config(2, 0)
    )  # menos aceptadas gana aunque rechace más
    assert configs.badness(config(1, 3)) < configs.badness(config(1, 4))


def test_triggers_pide_confianza_y_area() -> None:
    signals = {"car": [[0.39, 0.5], [0.8, 0.04], [0.8, 0.06]], "seg": [[0.9, 0.5]]}

    found = configs.triggers(signals, ("car",), presence.DEFAULT_THRESHOLDS)

    assert found == [{"senal": "car", "modelo": "detector (clase car)", "confianza": 0.8, "area": 0.06}]


def test_service_rules_sin_rueda_la_quita_de_plate_y_detail() -> None:
    rules = presence.service_rules(segmenter=True, wheel=False)

    assert "wheel" not in rules["plate"] and "wheel" not in rules["detail"]
    assert "seg" in rules["detail"] and "car" in rules["general"]


def test_reconstruct_names_ordena_como_presence_y_falla_si_no_cuadra(tmp_path: Path) -> None:
    for folder, names in (("damage-fp/images", ("b", "a")), ("cardd-detector/test", ("z",))):
        (tmp_path / folder).mkdir(parents=True)
        for name in names:
            (tmp_path / folder / f"{name}.jpg").write_bytes(b"x")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for name in ("q", "p", "r"):
        (fresh / f"{name}.jpg").write_bytes(b"x")
    measured: dict[str, list[Any]] = {
        "general": [{}, {}],
        "detail": [{}],
        "negativas": [],
        "plate": [{}],
        "negativas_nuevas": [{}, {}],
    }

    names = configs.reconstruct_names(measured, tmp_path, fresh)

    assert [Path(n).name for n in names["general"]] == ["a.jpg", "b.jpg"]
    assert [Path(n).name for n in names["negativas_nuevas"]] == ["p.jpg", "q.jpg"]  # solo las que se midieron
    assert names["plate"] == ["recorte-0"]
    measured["general"] = [{}]
    with pytest.raises(SystemExit, match="general"):
        configs.reconstruct_names(measured, tmp_path, fresh)


def test_main_escribe_el_informe_y_rechaza_salidas_en_el_repo(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"senales": scores(), "senales_faltantes": {}}))
    for group, count in (("general", 2), ("detail", 4)):
        folder = tmp_path / ("damage-fp/images" if group == "general" else "cardd-detector/test")
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"{index}.jpg").write_bytes(b"x")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for index in range(4):
        (fresh / f"{index}.jpg").write_bytes(b"x")
    args = ["--report", str(report), "--data-dir", str(tmp_path), "--fresh-negatives", str(fresh)]
    scored = json.loads(report.read_text())
    scored["senales"]["negativas"] = []  # `negative_photos` no encuentra el dataset de Open Images: ninguna
    report.write_text(json.dumps(scored))

    assert configs.main([*args, "--out-dir", str(tmp_path / "out")]) == 0
    saved = json.loads((tmp_path / "out" / "presence_configs.json").read_text())
    assert saved["mejor_de_a_y_b"] == "b_con_segmentador" and saved["fotos_aceptadas_por_alguna"] == 2

    assert configs.main([*args, "--out-dir", str(REPO_ROOT / "salida-de-prueba")]) == 1
    scored["senales_faltantes"] = {"detail": 1}
    report.write_text(json.dumps(scored))
    assert configs.main([*args, "--out-dir", str(tmp_path / "out2")]) == 2


# --- hojas de contacto ---------------------------------------------------------------------------------


def write_photo(path: Path, width: int = 200, height: int = 120) -> None:
    pixels = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), pixels)


def test_with_margin_no_se_sale_de_la_foto() -> None:
    assert audit_sheets.with_margin((10, 10, 30, 20), (100, 200), 0.5) == (0.0, 5.0, 40.0, 25.0)
    # contra el borde derecho e inferior: el recorte se corre hacia adentro en vez de achicarse
    assert audit_sheets.with_margin((150, 80, 200, 100), (100, 200), 1.0) == (50.0, 40.0, 200.0, 100.0)


def test_with_margin_pide_un_minimo_de_contexto_y_lo_corre_si_la_caja_toca_el_borde() -> None:
    left, top, right, bottom = audit_sheets.with_margin((0, 40, 10, 60), (100, 200), 0.1, 0.5)

    assert (right - left, bottom - top) == (100.0, 50.0)  # la mitad de la foto, de ancho y de alto
    assert (
        left == 0.0 and 40 >= top >= 0 and bottom >= 60
    )  # pegada al borde izquierdo: se corre hacia adentro


def test_sheets_arma_hojas_de_a_doce_paneles_y_dibuja_las_cajas(tmp_path: Path) -> None:
    write_photo(tmp_path / "a.jpg")
    panels = [
        audit_sheets.Panel(
            tmp_path / "a.jpg",
            f"{n:03d} car",
            crop=(10, 10, 100, 80),
            boxes=[((20, 20, 60, 50), audit_sheets.RED), ((25, 25, 70, 55), audit_sheets.GREEN)],
        )
        for n in range(13)
    ]

    written = audit_sheets.sheets(panels, tmp_path / "out", columns=4, rows=3, cell=100)

    assert [p.name for p in written] == ["hoja-01.jpg", "hoja-02.jpg"]
    first = cv2.imread(str(written[0]))
    assert first.shape == (300, 400, 3)
    assert (cv2.imread(str(written[1]))[110:, :] > 245).all()  # la 2.ª hoja tiene 1 panel: el resto, blanco
