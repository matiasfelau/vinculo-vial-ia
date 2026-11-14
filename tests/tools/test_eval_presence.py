"""tools/eval/presence.py: reglas de presencia, tasas por conjunto, recortes de patentes y fotos negativas."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_presence", REPO_ROOT / "tools" / "eval" / "presence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_presence"] = module
    spec.loader.exec_module(module)
    return module


presence = load()
T = presence.DEFAULT_THRESHOLDS


def test_passes_si_alguna_senal_de_la_regla_llega_al_umbral() -> None:
    assert presence.passes({"wheel": [[0.5, 0.1]]}, presence.RULES["R1"], T)
    assert not presence.passes({"wheel": [[0.3, 0.1]], "seg": [[0.9, 0.5]]}, presence.RULES["R1"], T)
    assert presence.passes({"seg": [[0.9, 0.5]]}, presence.RULES["R2"], T)
    assert presence.passes({"damage": [[0.8, 0.1]]}, presence.RULES["R3"], T)
    assert not presence.passes({"damage": [[0.8, 0.1]]}, presence.RULES["R2"], T)
    assert not presence.passes({}, presence.RULES["R3"], T)


def test_passes_exige_el_area_minima_en_el_mismo_hallazgo() -> None:
    signals = {"car": [[0.9, 0.01], [0.2, 0.5]]}  # uno confiable pero chico y otro grande pero poco confiable

    assert presence.passes(signals, ("car",), T, min_area=0.0)
    assert not presence.passes(signals, ("car",), T, min_area=0.05)


def test_rates_cuenta_las_fotos_que_pasan_por_regla() -> None:
    scores = {
        "plate": [{"plate": [[0.9, 0.2]]}, {"seg": [[0.9, 0.2]]}, {}],
        "negativas": [{"car": [[0.1, 0.2]]}],
    }

    table = presence.rates(scores)

    assert table["plate"] == {"fotos": 3, "solo_car": 0, "R1": 1, "R2": 2, "R3": 2}
    assert table["negativas"]["R3"] == 0
    assert presence.rates(scores, min_area=0.5)["plate"]["R2"] == 0


def test_verdict_marca_que_reglas_cumplen_a_y_c() -> None:
    table = {
        "general": {"fotos": 10, "solo_car": 10, "R1": 10, "R2": 10, "R3": 10},
        "plate": {"fotos": 5, "solo_car": 0, "R1": 5, "R2": 5, "R3": 5},
        "detail": {"fotos": 8, "solo_car": 0, "R1": 6, "R2": 8, "R3": 8},
        "negativas": {"fotos": 20, "solo_car": 0, "R1": 0, "R2": 1, "R3": 3},
    }

    result = presence.verdict(table)

    assert result["solo_car"]["cumple_A"] and result["solo_car"]["cumple_C"]
    assert result["R1"]["cumple_A"] and not result["R1"]["cumple_C"]
    assert result["R1"]["validas_rechazadas_por_rol"] == {"plate": 0, "detail": 2}
    assert not result["R2"]["cumple_A"] and result["R2"]["cumple_C"]
    assert result["R3"]["negativas_que_pasan"] == 3


def test_la_cota_es_3_sobre_n() -> None:
    assert presence.bound(60) == 0.05
    assert presence.bound(1) == 1.0
    assert presence.bound(0) is None


def make_coco(folder: Path, images: dict[str, list[tuple[str, tuple[float, float, float, float]]]]) -> None:
    folder.mkdir(parents=True)
    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "wheel"},
        {"id": 3, "name": "plate"},
        {"id": 4, "name": "face"},
    ]
    ids = {c["name"]: c["id"] for c in categories}
    coco_images, annotations = [], []
    for index, (name, boxes) in enumerate(images.items(), start=1):
        cv2.imwrite(str(folder / name), np.full((100, 200, 3), 90, np.uint8))
        coco_images.append({"id": index, "file_name": name, "width": 200, "height": 100})
        for label, bbox in boxes:
            annotations.append(
                {"id": len(annotations) + 1, "image_id": index, "category_id": ids[label], "bbox": list(bbox)}
            )
    (folder / presence.ANNOTATIONS).write_text(
        json.dumps({"images": coco_images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )


def test_negative_photos_deja_las_que_no_tienen_car_plate_ni_wheel(tmp_path: Path) -> None:
    make_coco(
        tmp_path / presence.OPEN_IMAGES / "test",
        {
            "auto.jpg": [("car", (0, 0, 50, 50))],
            "rueda.jpg": [("wheel", (0, 0, 20, 20))],
            "patente.jpg": [("plate", (0, 0, 20, 10))],
            "cara.jpg": [("face", (0, 0, 20, 20))],
            "nada.jpg": [],
        },
    )

    names = sorted(p.name for p in presence.negative_photos(tmp_path))

    assert names == ["cara.jpg", "nada.jpg"]


def test_plate_crops_agrega_margen_y_descarta_las_cajas_chicas(tmp_path: Path) -> None:
    make_coco(
        tmp_path / "train",
        {"a.jpg": [("plate", (50, 40, 40, 20)), ("plate", (0, 0, 8, 8))]},
    )

    crops = presence.plate_crops(tmp_path)

    assert len(crops) == 1
    assert crops[0].shape[:2] == (
        40,
        80,
    )  # 20 px de alto con 50 % de margen a cada lado → 40; 40 de ancho → 80


def test_plate_crops_respeta_el_limite(tmp_path: Path) -> None:
    make_coco(tmp_path / "train", {"a.jpg": [("plate", (10, 10, 40, 20)), ("plate", (100, 10, 40, 20))]})

    assert len(presence.plate_crops(tmp_path, limit=1)) == 1


def test_la_salida_dentro_del_repo_se_rechaza(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = presence.main(
        [
            *("--data-dir", str(tmp_path), "--eval-dir", str(tmp_path)),
            *("--out-dir", str(REPO_ROOT / "salida"), "--models-dir", str(tmp_path)),
        ]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_calibrate_damage_presence_pone_el_umbral_sobre_la_peor_negativa() -> None:
    negatives = [{"damage": [[0.20, 0.1]]}, {"damage": [[0.9692, 0.3], [0.5, 0.1]]}, {}]
    rejected = [{"damage": [[0.99, 0.2]]}, {"damage": [[0.50, 0.2]]}, {}]

    result = presence.calibrate_damage_presence(negatives, rejected)

    assert result["DAMAGE_PRESENCE_T"] == 0.97
    assert result["negativas_que_pasan"] == 0 and result["cota_95"] == 1.0
    assert result["detalles_recuperados"] == 1 and result["detalles_rechazados_por_las_otras_reglas"] == 3


def test_calibrate_damage_presence_si_una_negativa_llega_a_uno_no_hay_umbral() -> None:
    result = presence.calibrate_damage_presence([{"damage": [[1.0, 0.1]]}], [])

    assert result["DAMAGE_PRESENCE_T"] is None


def test_service_rule_aplica_la_regla_de_cada_rol_con_area_y_dano() -> None:
    car = {"car": [[0.9, 0.30]]}
    small_car = {"car": [[0.9, 0.02]]}
    wheel = {"wheel": [[0.9, 0.30]]}
    damage = {"damage": [[0.98, 0.01]]}
    scores = {
        "general": [car, small_car, wheel],  # la rueda sola no es un auto entero
        "plate": [wheel, small_car],
        "detail": [wheel, damage, {"damage": [[0.9, 0.5]]}],
        "negativas": [{}, damage],
    }

    out = presence.service_rule(scores, damage_t=0.97)

    assert (out["general"]["fotos"], out["general"]["rechazadas"]) == (3, 2)
    assert out["plate"]["rechazadas"] == 2  # la rueda ya no cuenta como presencia
    assert out["detail"]["rechazadas"] == 2  # la rueda y el daño de 0,90, que no llega a 0,97
    assert presence.service_rule(scores, damage_t=0.97, wheel=True)["plate"]["rechazadas"] == 1
    plate_only = {
        "general": [{"plate": [[0.9, 0.3]]}],
        "plate": [{"plate": [[0.9, 0.3]]}],
        "detail": [{"plate": [[0.9, 0.3]]}],
    }
    now = presence.service_rule(plate_only, damage_t=1.0)
    before = presence.service_rule(plate_only, damage_t=1.0, detail_plate=True)
    assert (now["general"]["rechazadas"], now["plate"]["rechazadas"], now["detail"]["rechazadas"]) == (
        1,
        0,
        1,
    )
    assert before["detail"]["rechazadas"] == 0  # antes una patente sola alcanzaba en DETAIL
    assert out["detail"]["negativas_que_pasan"] == 1  # el daño sin auto pasa en DETAIL
    assert out["general"]["negativas_que_pasan"] == 0 and out["general"]["cota_95_negativas"] == 1.0


def test_service_rules_sin_segmentador_quita_solo_esa_senal() -> None:
    with_seg = presence.service_rules()
    without = presence.service_rules(segmenter=False)

    assert all("seg" in rule for rule in with_seg.values())
    assert all("seg" not in rule for rule in without.values())
    assert without["general"] == ("car",) and without["detail"] == ("car",)  # la patente no cuenta en DETAIL
    assert without["plate"] == ("car", "plate")
    anterior = presence.service_rules(segmenter=False, wheel=True, detail_plate=True)  # la regla anterior
    assert anterior["detail"] == ("car", "wheel", "plate")


def test_service_rule_sin_segmentador_rechaza_lo_que_solo_el_segmentador_veia() -> None:
    seg_only = {"seg": [[0.9, 0.30]]}
    car = {"car": [[0.9, 0.30]]}
    scores = {"general": [seg_only, car], "plate": [seg_only], "detail": [seg_only], "negativas": [seg_only]}

    with_seg = presence.service_rule(scores, damage_t=0.97)
    without = presence.service_rule(scores, damage_t=0.97, segmenter=False)

    assert with_seg["general"]["rechazadas"] == 0 and without["general"]["rechazadas"] == 1
    assert without["plate"]["rechazadas"] == 1 and without["detail"]["rechazadas"] == 1
    assert with_seg["general"]["negativas_que_pasan"] == 1  # el segmentador deja pasar una negativa
    assert without["general"]["negativas_que_pasan"] == 0


def test_service_rule_sin_dano_no_recupera_detalles_por_dano() -> None:
    damage = {"damage": [[0.99, 0.01]]}
    scores = {"general": [], "plate": [], "detail": [damage], "negativas": [damage]}

    with_damage = presence.service_rule(scores, damage_t=0.97)
    without = presence.service_rule(scores, damage_t=0.97, damage=False)

    assert with_damage["detail"]["rechazadas"] == 0 and with_damage["detail"]["negativas_que_pasan"] == 1
    assert without["detail"]["rechazadas"] == 1 and without["detail"]["negativas_que_pasan"] == 0


def test_service_rule_reporta_las_negativas_nuevas_por_separado() -> None:
    car = {"car": [[0.9, 0.30]]}
    scores = {
        "general": [car],
        "plate": [car],
        "detail": [car],
        "negativas": [{}, {}],
        "negativas_nuevas": [{}, {}, {}, car],
    }

    out = presence.service_rule(scores, damage_t=0.97)

    assert out["general"]["negativas"] == 2 and out["general"]["negativas_que_pasan"] == 0
    assert out["general"]["cota_95_negativas"] == 1.0  # 3/2 se acota a 1
    fresh = out["general"]["negativas_nuevas"]
    assert fresh["fotos"] == 4 and fresh["que_pasan"] == 1 and fresh["cota_95"] > 0.25


def test_service_comparison_trae_las_tres_configuraciones() -> None:
    out = presence.service_comparison({"general": [], "plate": [], "detail": [], "negativas": []}, 0.97)

    assert set(out) == {"con_segmentador", "sin_segmentador", "sin_segmentador_ni_dano"}


def test_upper_bound_cero_fallas_es_3_sobre_n_y_con_fallas_es_clopper_pearson() -> None:
    assert presence.upper_bound(0, 1200) == pytest.approx(0.0025)
    assert presence.upper_bound(0, 0) is None
    assert presence.upper_bound(3, 3) == 1.0
    assert presence.upper_bound(1, 100) == pytest.approx(0.0466, abs=5e-4)


def test_measure_junta_las_senales_y_reduce_la_foto_como_el_servicio() -> None:
    from types import SimpleNamespace

    from vvi_ia.domain.model import Box, BoxLabel, CarInstance

    seen: list[tuple[int, int]] = []

    def detect(img: np.ndarray) -> list[Box]:  # type: ignore[type-arg]
        seen.append(img.shape[:2])
        return [Box(0, 0, 50, 50, 0.9, BoxLabel.CAR), Box(0, 0, 5, 5, 0.8, BoxLabel.WHEEL)]

    mask = np.zeros((100, 200), bool)
    mask[:50, :100] = True
    models = SimpleNamespace(
        vehicle=SimpleNamespace(detect=detect),
        cars=SimpleNamespace(cars=lambda img: [CarInstance(mask, 0.7)]),
        damage=SimpleNamespace(detect=lambda img: []),
    )

    out = presence.measure(models, None, np.zeros((100, 200, 3), np.uint8), max_side=0)

    assert out["car"] == [[0.9, pytest.approx(2500 / 20000)]] and out["seg"] == [[0.7, 0.25]]
    assert out["wheel"] == [[0.8, pytest.approx(25 / 20000)]] and out["plate"] == [] and out["damage"] == []
    presence.measure(models, None, np.zeros((400, 800, 3), np.uint8), max_side=200)
    assert seen[-1] == (100, 200)


@pytest.mark.parametrize("missing", ["vehicle", "cars", "damage"])
def test_measure_no_deja_pasar_un_modelo_que_no_responde(missing: str) -> None:
    from types import SimpleNamespace

    models = SimpleNamespace(
        vehicle=SimpleNamespace(detect=lambda img: []),
        cars=SimpleNamespace(cars=lambda img: []),
        damage=SimpleNamespace(detect=lambda img: []),
    )
    getattr(models, missing).__dict__[next(iter(getattr(models, missing).__dict__))] = lambda img: None

    with pytest.raises(presence.MissingSignalError):
        presence.measure(models, None, np.zeros((10, 10, 3), np.uint8))
