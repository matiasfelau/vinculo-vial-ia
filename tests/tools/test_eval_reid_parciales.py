"""tools/eval/reid_parciales.py: los tres tipos de pares, la calibración conjunta y los desgloses.

Sintético: recortes lisos cuyo tono dice de qué auto son y un embedder falso que devuelve un vector
según ese tono.
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

    def versions(self) -> dict[str, str]:
        return {"embedder": "falso@1"}


def write(folder: Path, name: str, tone: int) -> None:
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.full((24, 24, 3), tone, np.uint8))[1].tofile(str(target))


def make_world(
    root: Path, cars_per_split: int = 30
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    parts: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    partial: dict[str, dict[str, list[dict[str, Any]]]] = {s: {"patente": [], "detalle": []} for s in parts}
    index = 0
    for split in parts:
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
                parts[split].append({**meta, "archivo": name, "azimut": view * 90.0 + 10, "foto": view})
            for kind in ("patente", "detalle"):
                name = f"{car}/{kind}.png"
                write(root / "partial-v2", name, tone)
                partial[split][kind].append(
                    {**meta, "archivo": name, "azimut": 45.0, "foto": 9, "tipo": kind}
                )
    return parts, partial


def test_evaluate_model_arma_los_tres_tipos_y_calibra_con_todos(tmp_path: Path) -> None:
    parts, partial = make_world(tmp_path)

    report = rp.evaluate_model(FakeModels(), tmp_path, parts, partial, workers=1, seed=0)

    assert report["version"] == "falso@1"
    assert set(report["test"]) == {"total", *rp.TYPES}
    assert report["calibracion_en_val_con_los_tres_tipos"]["umbral"] is not None
    assert set(report["calibracion_en_val_por_tipo"]) == set(rp.TYPES)
    for scope in report["test"].values():
        assert scope["todos"]["auc"] == 1.0  # el embedder falso separa perfecto
        assert scope["pares"]["mismo_auto"] > 0 and scope["pares"]["distintos"] > 0
        assert set(scope["por_luz"]) == set(rp.LIGHTING)
    total = report["test"]["total"]["pares"]["mismo_auto"]
    assert total == sum(report["test"][t]["pares"]["mismo_auto"] for t in rp.TYPES)


def test_los_pares_parciales_respetan_la_separacion_angular(tmp_path: Path) -> None:
    parts, partial = make_world(tmp_path, cars_per_split=3)
    # las vistas enteras están a 10, 100, 190 y 280°; el parcial sale de la de 45°: todas separan >= 30°
    whole = rp.embed_files(FakeModels(), tmp_path / "crops-v2", [w["archivo"] for w in parts["val"]], 1)
    part = {
        "patente": rp.embed_files(
            FakeModels(), tmp_path / "partial-v2", [p["archivo"] for p in partial["val"]["patente"]], 1
        )
    }

    types = rp.build_types(parts["val"], {"patente": partial["val"]["patente"]}, whole, part, seed=0)

    assert set(types) == {"entero", "general_patente"}
    assert types["general_patente"].same.size == 3 * 4  # 3 autos x 4 vistas enteras
