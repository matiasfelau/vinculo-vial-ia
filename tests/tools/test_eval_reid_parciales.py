"""tools/eval/reid_parciales.py: pares general-patente, umbral único por S4 y cifras por luz.

Sintético: recortes lisos cuyo tono dice de qué auto son y un embedder falso que da un vector según ese tono.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DIM = 64


def load() -> ModuleType:
    for path in (str(REPO_ROOT / "tools" / "training" / "reid"), str(REPO_ROOT / "tools" / "eval")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "eval_reid_parciales", REPO_ROOT / "tools" / "eval" / "reid_parciales.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_reid_parciales"] = module
    spec.loader.exec_module(module)
    return module


rp = load()


class FakeEmbedder:
    def embed(self, crop: Any, mask: Any) -> Any:
        tone = int(crop[0, 0, 0])
        vector = np.zeros(DIM, np.float32)
        vector[tone % DIM] = 1.0
        return vector


class FakeModels:
    embedder = FakeEmbedder()


def lazy() -> Any:
    return rp.LazyModels(lambda: FakeModels())


def write(folder: Path, name: str, tone: int) -> None:
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.full((24, 24, 3), tone, np.uint8))[1].tofile(str(target))


def make_world(
    root: Path, cars_per_split: int = 30, plates: int = 12
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    whole: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    plate: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    index = 0
    for split in whole:
        for k in range(cars_per_split):
            index += 1
            car, tone = f"c{index}", index % DIM
            light = ("standard", "reflective", "dark")[k % 3]
            meta = {
                "auto": car,
                "marca": f"m{k % 2}",
                "modelo": "x",
                "tipo": "Sedan",
                "color": "blanco",
                "luz": light,
            }
            for view in range(4):
                name = f"{car}/{view}.png"
                write(root / "crops-v2", name, tone)
                whole[split].append({**meta, "archivo": name, "azimut": view * 90.0 + 10, "foto": view})
            for n in range(plates):
                name = f"{car}/patente{n}.png"
                write(root / "partial-v2", name, tone)
                plate[split].append({**meta, "archivo": name, "azimut": 45.0, "foto": 9, "tipo": "patente"})
    return whole, plate


def run(root: Path, whole: Any, plate: Any, cache: Path | None = None, models: Any = None) -> dict[str, Any]:
    return rp.evaluate_model(
        models or lazy(), "falso@1", root, whole, plate, workers=1, seed=0, cache_dir=cache
    )


def test_el_umbral_es_el_menor_que_alcanza_el_s4_pedido() -> None:
    different = np.linspace(-0.5, 0.5, 1000)  # 1000 pares de autos distintos, de -0,5 a 0,5

    threshold = rp.threshold_for_s4(different, 0.70)

    assert (different < threshold).mean() >= 0.70
    assert (different < np.nextafter(threshold, -np.inf)).mean() < 0.70  # y no hay uno menor que lo alcance


def test_evaluate_model_elige_el_umbral_con_general_patente_y_reporta_en_prueba(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path)

    report = run(tmp_path, whole, plate)

    choice = report["eleccion_en_validacion"]
    assert choice["validacion"]["S4"] >= rp.TARGET_S4
    assert choice["cota_de_u1_en_validacion_aceptable"] is True
    assert set(report["test"]) == {"entero", "general_patente"}
    patente = report["test"]["general_patente"]
    assert (
        patente["total"]["U1"] == 0.0 and patente["total"]["S4"] == 1.0
    )  # el embedder falso separa perfecto
    assert patente["auc"] == 1.0
    assert set(patente["por_luz"]) == set(rp.LIGHTING)
    assert patente["total"]["mismo_auto"] == sum(c["mismo_auto"] for c in patente["por_luz"].values())
    assert report["cumple_la_cota_de_u1_en_prueba"] is True


def test_el_detalle_no_entra_a_ninguna_cifra(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path)

    report = run(tmp_path, whole, plate)

    assert "general_detalle" not in report["test"] and "general_detalle" not in report["val"]
    assert not hasattr(rp, "SCHEMES")


def test_con_pocos_pares_la_cota_de_u1_de_validacion_no_se_acepta(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path, cars_per_split=4, plates=1)  # 16 pares del mismo auto: 3/16 > 2 %

    report = run(tmp_path, whole, plate)

    assert report["eleccion_en_validacion"]["cota_de_u1_en_validacion_aceptable"] is False


def test_los_pares_de_patente_respetan_la_separacion_angular(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path, cars_per_split=3, plates=1)
    # las vistas enteras están a 10, 100, 190 y 280°; la patente sale de la de 45°: todas separan >= 30°
    folder, partial = tmp_path / "crops-v2", tmp_path / "partial-v2"
    w = rp.embed_files(lazy(), folder, [r["archivo"] for r in whole["val"]], 1)
    p = rp.embed_files(lazy(), partial, [r["archivo"] for r in plate["val"]], 1)

    sets = rp.build_sets(whole["val"], plate["val"], w, p, seed=0)

    assert set(sets) == {"entero", "general_patente"}
    assert sets["general_patente"].same.size == 3 * 4  # 3 autos x 4 vistas enteras


def test_el_cache_de_embeddings_se_reutiliza_sin_cargar_los_modelos(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path, cars_per_split=3, plates=1)
    names = [w["archivo"] for w in whole["val"]]
    cache = tmp_path / "c" / "m.npy"
    first = rp.embed_files(lazy(), tmp_path / "crops-v2", names, 1, cache)

    def boom() -> Any:
        raise AssertionError("no tendría que cargar el modelo")

    second = rp.embed_files(rp.LazyModels(boom), tmp_path / "crops-v2", names, 1, cache)

    assert cache.is_file() and (first == second).all()


def test_evaluate_model_con_todo_en_cache_no_carga_los_modelos(tmp_path: Path) -> None:
    whole, plate = make_world(tmp_path)
    cache = tmp_path / "cache"
    first = run(tmp_path, whole, plate, cache)

    def boom() -> Any:
        raise AssertionError("no tendría que cargar el modelo")

    second = run(tmp_path, whole, plate, cache, models=rp.LazyModels(boom))

    assert first["eleccion_en_validacion"]["umbral"] == second["eleccion_en_validacion"]["umbral"]
