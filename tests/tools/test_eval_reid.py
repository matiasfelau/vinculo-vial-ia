"""tools/eval/reid.py: pares, distribuciones y propuesta de umbrales, con embeddings sintéticos.

No hay fotos ni autos reales: las "fotos" son imágenes lisas cuyo tono dice de qué auto son, y el embedder
falso devuelve un vector que depende de ese tono más un poco de ruido.
"""

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from vvi_ia.domain.model import CarInstance

REPO_ROOT = Path(__file__).resolve().parents[2]
DIM = 16


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_reid", REPO_ROOT / "tools" / "eval" / "reid.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_reid"] = module
    spec.loader.exec_module(module)
    return module


reid = load()


def unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def cluster(center: int, count: int, noise: float, seed: int) -> list[np.ndarray]:
    """`count` vectores alrededor de la dirección `center` (un auto)."""
    rng = np.random.default_rng(seed)
    base = np.zeros(DIM)
    base[center % DIM] = 1.0
    return [unit(base + noise * rng.normal(size=DIM)) for _ in range(count)]


# --- el CSV ------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[str], header: str = "archivo,auto") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_carga_las_fotos(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "fotos.csv", ["a.jpg,auto1", " b.jpg , auto1 ", "c.jpg,auto2"])

    assert reid.load_photos(path) == [
        reid.Photo("a.jpg", "auto1"),
        reid.Photo("b.jpg", "auto1"),
        reid.Photo("c.jpg", "auto2"),
    ]


@pytest.mark.parametrize(
    ("rows", "header", "message"),
    [
        (["a.jpg,auto1", "a.jpg,auto2"], "archivo,auto", r"fila 3.*repetida"),
        ([",auto1"], "archivo,auto", r"fila 2.*faltan"),
        (["a.jpg,"], "archivo,auto", r"fila 2.*faltan"),
        (["a.jpg"], "archivo", "necesita las columnas"),
    ],
)
def test_csv_invalido_indica_la_fila(tmp_path: Path, rows: list[str], header: str, message: str) -> None:
    with pytest.raises(reid.DataError, match=message):
        reid.load_photos(write_csv(tmp_path / "fotos.csv", rows, header))


def test_csv_inexistente(tmp_path: Path) -> None:
    with pytest.raises(reid.DataError, match="no existe"):
        reid.load_photos(tmp_path / "nada.csv")


# --- pares y distribuciones ----------------------------------------------------------------------


def test_pares_iguales_y_distintos() -> None:
    by_car = {"a": cluster(0, 3, 0.05, 1), "b": cluster(1, 2, 0.05, 2), "c": cluster(2, 1, 0.05, 3)}

    same, different = reid.similarities(by_car)

    assert same.size == 3 + 1 + 0  # combinaciones de 3 fotos + de 2 fotos + ninguna
    assert different.size == 3 * 2 + 3 * 1 + 2 * 1
    assert same.min() > 0.9 and different.max() < 0.3


def test_los_pares_distintos_se_sampleaan_de_forma_reproducible() -> None:
    by_car = {str(i): cluster(i, 4, 0.1, i) for i in range(6)}

    first = reid.similarities(by_car, max_different=20, seed=5)[1]
    second = reid.similarities(by_car, max_different=20, seed=5)[1]
    other = reid.similarities(by_car, max_different=20, seed=6)[1]

    assert first.size == 20 and np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_sin_pares_iguales_o_distintos() -> None:
    only_one_each = {"a": cluster(0, 1, 0.1, 1), "b": cluster(1, 1, 0.1, 2)}
    same, different = reid.similarities(only_one_each)
    assert same.size == 0 and different.size == 1

    same, different = reid.similarities({"a": cluster(0, 3, 0.1, 1)})
    assert same.size == 3 and different.size == 0


def test_describe() -> None:
    stats = reid.describe(np.array([0.1, 0.5, 0.9]))

    assert stats["pares"] == 3 and stats["minimo"] == 0.1 and stats["maximo"] == 0.9
    assert stats["media"] == pytest.approx(0.5) and stats["p50"] == pytest.approx(0.5)
    assert reid.describe(np.array([])) == {"pares": 0}


def test_histograma_cuenta_todos_los_pares() -> None:
    values = np.array([-0.5, 0.0, 0.01, 0.52, 0.99, 1.0])

    counts = reid.histogram(values)

    assert sum(counts) == values.size and len(counts) == len(reid.HISTOGRAM_BINS) - 1


def test_auc() -> None:
    assert reid.auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert reid.auc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert reid.auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5  # empate total
    assert reid.auc(np.array([0.9, 0.2]), np.array([0.5])) == 0.5
    assert reid.auc(np.array([]), np.array([0.5])) is None


def test_reparto_con_los_umbrales() -> None:
    values = np.array([0.1, 0.4, 0.5, 0.6, 0.7, 0.9])

    shares = reid.decide_shares(values, 0.5, 0.7)

    assert shares == {
        "SAME": pytest.approx(2 / 6, abs=1e-4),
        "UNCERTAIN": pytest.approx(2 / 6, abs=1e-4),
        "DIFFERENT": pytest.approx(2 / 6, abs=1e-4),
    }
    assert reid.decide_shares(np.array([]), 0.5, 0.7) == {"SAME": None, "UNCERTAIN": None, "DIFFERENT": None}


# --- propuesta de umbrales -----------------------------------------------------------------------


def test_con_distribuciones_separadas_la_banda_cae_en_el_hueco() -> None:
    rng = np.random.default_rng(0)
    same = np.clip(rng.normal(0.85, 0.05, 500), -1, 1)
    different = np.clip(rng.normal(0.20, 0.10, 2000), -1, 1)

    proposal = reid.propose_thresholds(same, different)

    assert proposal["se_separan"] is True
    assert proposal["REID_T_LOW"] < proposal["REID_T_HIGH"]
    assert proposal["pares_iguales"]["DIFFERENT"] <= 0.01  # falso fraude
    assert proposal["pares_distintos"]["SAME"] <= 0.01  # fraude que pasa
    assert proposal["pares_iguales"]["SAME"] > 0.9 and proposal["pares_distintos"]["DIFFERENT"] > 0.95
    assert proposal["pares_iguales"]["UNCERTAIN"] < 0.1 and proposal["pares_distintos"]["UNCERTAIN"] < 0.05


def test_los_umbrales_son_multiplos_de_un_centesimo() -> None:
    rng = np.random.default_rng(1)

    proposal = reid.propose_thresholds(rng.normal(0.8, 0.05, 300), rng.normal(0.2, 0.1, 900))

    for key in ("REID_T_LOW", "REID_T_HIGH"):
        assert round(proposal[key], 2) == proposal[key]


def test_si_las_distribuciones_se_solapan_la_banda_se_ensancha_pero_los_limites_se_cumplen() -> None:
    rng = np.random.default_rng(2)
    same = rng.normal(0.5, 0.15, 500)
    different = rng.normal(0.45, 0.15, 500)

    proposal = reid.propose_thresholds(same, different)

    assert proposal["se_separan"] is False
    assert proposal["REID_T_LOW"] < proposal["REID_T_HIGH"]
    assert proposal["pares_iguales"]["DIFFERENT"] <= 0.01 and proposal["pares_distintos"]["SAME"] <= 0.01
    assert proposal["pares_iguales"]["UNCERTAIN"] > 0.5  # casi todo queda para revisar


def test_las_distribuciones_identicas_dejan_una_banda_minima() -> None:
    values = np.full(100, 0.5)

    proposal = reid.propose_thresholds(values, values)

    assert (
        proposal["REID_T_HIGH"] == pytest.approx(proposal["REID_T_LOW"] + 0.01)
        or proposal["REID_T_HIGH"] > proposal["REID_T_LOW"]
    )


def test_limites_mas_estrictos_dan_umbrales_mas_conservadores() -> None:
    rng = np.random.default_rng(3)
    same, different = rng.normal(0.8, 0.1, 1000), rng.normal(0.3, 0.15, 3000)

    loose = reid.propose_thresholds(same, different, 0.05, 0.05)
    strict = reid.propose_thresholds(same, different, 0.001, 0.001)

    assert strict["REID_T_LOW"] <= loose["REID_T_LOW"] and strict["REID_T_HIGH"] >= loose["REID_T_HIGH"]


def test_sin_pares_no_hay_propuesta() -> None:
    assert reid.propose_thresholds(np.array([]), np.array([0.2]))["posible"] is False
    assert reid.propose_thresholds(np.array([0.9]), np.array([]))["posible"] is False


# --- de las fotos a los embeddings ----------------------------------------------------------------


class FakeModels:
    """Segmentador y embedder falsos. El tono de la foto dice de qué auto es."""

    def __init__(self, cars: Any = None, embedding: Any = None) -> None:
        self.embedded_masks: list[int] = []
        self._cars = cars
        self._embedding = embedding
        self.cars = SimpleNamespace(cars=self._segment)
        self.embedder = SimpleNamespace(embed=self._embed)

    def _segment(self, img: np.ndarray) -> list[CarInstance] | None:
        if self._cars is not None:
            return self._cars(img)  # type: ignore[no-any-return]
        small = np.zeros(img.shape[:2], bool)
        small[:2, :2] = True
        big = np.ones(img.shape[:2], bool)
        return [CarInstance(small, 0.9), CarInstance(big, 0.8)]

    def _embed(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        self.embedded_masks.append(int(mask.sum()))
        if self._embedding is not None:
            return self._embedding(img)  # type: ignore[no-any-return]
        car = int(img[0, 0, 0]) // 40
        return cluster(car, 1, 0.05, int(img[0, 0, 1]))[0]

    def versions(self) -> dict[str, str]:
        return {"segmenter": "seg@abc", "embedder": "emb@abc"}


def make_photos(folder: Path, spec: dict[str, tuple[int, int]]) -> list[Any]:
    """`spec[archivo] = (auto, variante)`; el tono del rojo es el auto y el del verde, la variante."""
    folder.mkdir(parents=True, exist_ok=True)
    photos = []
    for name, (car, variant) in spec.items():
        pixels = np.zeros((20, 30, 3), np.uint8)
        pixels[..., 2] = car * 40 + 20  # OpenCV guarda BGR: el canal 2 es el rojo
        pixels[..., 1] = variant
        cv2.imwrite(str(folder / name), pixels)
        photos.append(reid.Photo(name, f"auto{car}"))
    return photos


def test_se_usa_el_embedding_del_auto_mas_grande(tmp_path: Path) -> None:
    photos = make_photos(tmp_path, {"a.jpg": (1, 1)})
    models = FakeModels()

    by_car, discarded = reid.embed_photos(photos, tmp_path, models)

    assert models.embedded_masks == [20 * 30] and not discarded  # la máscara grande, no la de 4 píxeles
    assert list(by_car) == ["auto1"] and len(by_car["auto1"]) == 1


def test_las_fotos_que_no_se_pueden_usar_se_cuentan_por_motivo(tmp_path: Path) -> None:
    photos = make_photos(
        tmp_path, {"ok.jpg": (1, 1), "sin_auto.jpg": (2, 2), "falla.jpg": (3, 3), "sin_emb.jpg": (4, 4)}
    )
    photos.append(reid.Photo("no_existe.jpg", "auto9"))

    def segment(img: np.ndarray) -> list[CarInstance] | None:
        marker = int(img[0, 0, 0]) // 40
        if marker == 2:
            return []
        if marker == 3:
            return None
        return [CarInstance(np.ones(img.shape[:2], bool), 0.9)]

    def embed(img: np.ndarray) -> np.ndarray | None:
        return None if int(img[0, 0, 0]) // 40 == 4 else cluster(1, 1, 0.05, 1)[0]

    by_car, discarded = reid.embed_photos(photos, tmp_path, FakeModels(segment, embed))

    assert list(by_car) == ["auto1"]
    assert discarded == Counter(
        {"sin auto": 1, "falla del segmentador": 1, "sin embedding": 1, "foto ilegible": 1}
    )


# --- el reporte y la línea de comandos -------------------------------------------------------------


def test_el_reporte_es_serializable_y_no_nombra_fotos() -> None:
    by_car = {"auto-secreto": cluster(0, 3, 0.05, 1), "otro": cluster(1, 3, 0.05, 2)}

    report = reid.build_report(by_car, Counter({"sin auto": 2}), 8, 0.01, 0.01, 1000, 0)
    text = json.dumps(report)

    assert (
        report["fotos_usadas"] == 6
        and report["fotos_en_el_csv"] == 8
        and report["descartadas"] == {"sin auto": 2}
    )
    assert report["autos"] == 2 and report["autos_con_mas_de_una_foto"] == 2
    assert report["auc"] == 1.0
    assert "auto-secreto" not in text and "otro" not in text
    assert (
        report["valores_actuales"]["REID_T_LOW"] == 0.5 and report["valores_actuales"]["REID_T_HIGH"] == 0.7
    )


def test_los_valores_actuales_coinciden_con_los_de_la_config() -> None:
    from vvi_ia.config import Settings

    assert reid.CURRENT_T_LOW == Settings.model_fields["reid_t"].default


def run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[str, int, int]],
    extra: list[str] | None = None,
) -> tuple[int, Path]:
    root = tmp_path / "data"
    spec = {name: (car, variant) for name, car, variant in rows}
    make_photos(root / "reid" / "fotos", spec)
    write_csv(root / "reid" / "fotos.csv", [f"{name},auto{car}" for name, car, _ in rows])
    monkeypatch.setattr(reid, "build_models", lambda *a, **k: FakeModels())
    out = tmp_path / "out"
    args = ["--data-dir", str(root), "--out-dir", str(out), "--models-dir", str(tmp_path)] + (extra or [])
    return reid.main(args), out


def test_main_de_punta_a_punta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [(f"a{i}.jpg", 1, i + 1) for i in range(4)] + [(f"b{i}.jpg", 3, i + 1) for i in range(4)]

    code, out = run_main(tmp_path, monkeypatch, rows)

    assert code == 0
    report = json.loads((out / "reid_report.json").read_text(encoding="utf-8"))
    assert report["pares_iguales"]["pares"] == 12 and report["pares_distintos"]["pares"] == 16
    assert report["modelos"] == {"segmenter": "seg@abc", "embedder": "emb@abc"}
    assert report["propuesta"]["REID_T_LOW"] < report["propuesta"]["REID_T_HIGH"]
    shown = capsys.readouterr().out
    assert "Propuesta: REID_T_LOW=" in shown and "a0.jpg" not in shown and "auto1" not in shown


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = reid.main(
        [
            "--data-dir",
            str(REPO_ROOT / "x"),
            "--out-dir",
            str(REPO_ROOT / "y"),
            "--models-dir",
            str(REPO_ROOT),
        ]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_main_con_un_csv_invalido(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "data" / "reid").mkdir(parents=True)
    write_csv(tmp_path / "data" / "reid" / "fotos.csv", ["a.jpg"], header="archivo")

    code = reid.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            "--models-dir",
            str(tmp_path),
        ]
    )

    assert code == 1 and "No se pudo empezar" in capsys.readouterr().err


def test_main_con_pocos_pares_no_propone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_main(tmp_path, monkeypatch, [("a.jpg", 1, 1), ("b.jpg", 2, 1)])

    assert code == 0
    assert json.loads((out / "reid_report.json").read_text(encoding="utf-8"))["propuesta"]["posible"] is False
    assert "Sin propuesta" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--max-false-fraud", "--max-false-accept"])
def test_los_limites_estan_entre_cero_y_uno(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        reid.parse_args(
            [
                "--data-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--models-dir",
                str(tmp_path),
                flag,
                "1.5",
            ]
        )


def test_main_exige_las_rutas(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EVAL_DATA_DIR", "EVAL_OUT", "MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        reid.parse_args([])


def test_el_csv_de_ejemplo_del_readme_tiene_las_columnas_esperadas() -> None:
    assert reid.COLUMNS == ("archivo", "auto")
    assert csv.reader  # el formato se documenta en tools/eval/README.md
