"""tools/labeling/prelabel_curb.py: polígonos, COCO para CVAT, tandas con retome y exploración de prompts.

SAM 3 no se carga en los tests: un segmentador falso devuelve máscaras rectangulares conocidas.
"""

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHAPE = (60, 100)


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prelabel_curb", REPO_ROOT / "tools" / "labeling" / "prelabel_curb.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prelabel_curb"] = module
    spec.loader.exec_module(module)
    return module


prelabel = load()


def box_mask(y1: int, y2: int, x1: int, x2: int, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[y1:y2, x1:x2] = True
    return mask


class FakeBackend:
    """Devuelve instancias fijas por prompt y cuenta cuántas veces se la usó."""

    def __init__(self, results: dict[str, list[tuple[float, np.ndarray]]] | None = None) -> None:
        self.results = (
            results
            if results is not None
            else {
                "curb": [(0.9, box_mask(40, 46, 5, 95))],
                "dropped curb": [(0.8, box_mask(40, 46, 30, 60))],
                "curb ramp": [(0.6, box_mask(40, 46, 31, 61)), (0.5, box_mask(10, 14, 70, 90))],
            }
        )
        self.calls = 0

    def segment(self, image: np.ndarray, prompts: list[str]) -> dict[str, list[tuple[float, np.ndarray]]]:
        self.calls += 1
        return {prompt: self.results.get(prompt, []) for prompt in prompts}


PROMPTS = {"curb": ("curb",), "curb_cut": ("dropped curb", "curb ramp")}
SETTINGS = (0.3, 0.5, 1.5, 0.0005)


def make_photos(folder: Path, count: int = 4) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"f{index}.jpg"
        cv2.imwrite(str(path), np.full((*SHAPE, 3), 40 * (index + 1), np.uint8))
        paths.append(path)
    return paths


# --- polígonos -------------------------------------------------------------------------------------


def test_un_rectangulo_da_un_poligono_de_cuatro_puntos() -> None:
    polygons = prelabel.mask_to_polygons(box_mask(10, 20, 30, 70))

    assert len(polygons) == 1 and len(polygons[0]) == 8
    xs, ys = polygons[0][0::2], polygons[0][1::2]
    assert (min(xs), max(xs), min(ys), max(ys)) == (30.0, 69.0, 10.0, 19.0)


def test_dos_partes_dan_dos_poligonos_y_el_ruido_se_descarta() -> None:
    mask = box_mask(10, 20, 5, 40) | box_mask(30, 45, 60, 95)
    mask[0, 0] = True  # un píxel suelto

    assert len(prelabel.mask_to_polygons(mask)) == 2


def test_una_mascara_vacia_no_da_poligonos() -> None:
    assert prelabel.mask_to_polygons(np.zeros(SHAPE, bool)) == []


def test_los_poligonos_vuelven_a_la_misma_mascara() -> None:
    mask = box_mask(10, 30, 20, 80)

    back = prelabel.polygons_to_mask(prelabel.mask_to_polygons(mask), SHAPE)

    assert prelabel.mask_iou(mask, back) > 0.9


def test_instancia_con_caja_y_area() -> None:
    instance = prelabel.to_instance("curb", "curb", 0.91234, box_mask(10, 20, 30, 70), 1.5, 0.0005)

    assert instance is not None
    assert instance.bbox == [30.0, 10.0, 40.0, 10.0] and instance.area == 400.0 and instance.score == 0.9123
    assert prelabel.to_instance("curb", "curb", 0.9, np.zeros(SHAPE, bool), 1.5, 0.0005) is None


def test_dedupe_quita_las_que_se_pisan_de_la_misma_clase_y_deja_las_de_otra() -> None:
    def instance(label: str, score: float, mask: np.ndarray) -> Any:
        return prelabel.to_instance(label, "p", score, mask, 1.5, 0.0005)

    a = instance("curb_cut", 0.8, box_mask(40, 46, 30, 60))
    b = instance("curb_cut", 0.6, box_mask(40, 46, 31, 61))  # casi igual a la anterior
    c = instance("curb_cut", 0.5, box_mask(10, 14, 70, 90))  # otra zona
    d = instance("curb", 0.7, box_mask(40, 46, 30, 60))  # misma zona, otra clase

    kept = prelabel.dedupe([b, a, c, d], SHAPE)

    assert [(i.label, i.score) for i in kept] == [("curb_cut", 0.8), ("curb", 0.7), ("curb_cut", 0.5)]


# --- COCO para CVAT ----------------------------------------------------------------------------------


def test_el_coco_tiene_las_clases_en_orden_y_poligonos() -> None:
    instances = prelabel.process_photo(np.zeros((*SHAPE, 3), np.uint8), FakeBackend(), PROMPTS, 1.5, 0.0005)

    coco = prelabel.build_coco({"a.jpg": (SHAPE, instances)})

    assert [c["name"] for c in coco["categories"]] == ["curb", "curb_cut"] and [
        c["id"] for c in coco["categories"]
    ] == [1, 2]
    assert coco["images"] == [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 60}]
    assert {a["category_id"] for a in coco["annotations"]} == {1, 2}
    assert all(isinstance(a["segmentation"][0], list) and a["iscrowd"] == 0 for a in coco["annotations"])
    assert all(set(a["attributes"]) == {"score", "prompt"} for a in coco["annotations"])


def test_los_dos_prompts_del_rebaje_se_unen_sin_duplicados() -> None:
    instances = prelabel.process_photo(np.zeros((*SHAPE, 3), np.uint8), FakeBackend(), PROMPTS, 1.5, 0.0005)

    assert sorted(i.label for i in instances) == ["curb", "curb_cut", "curb_cut"]  # la casi-igual se descartó
    assert {i.prompt for i in instances if i.label == "curb_cut"} == {"dropped curb", "curb ramp"}


def test_la_foto_sin_resultados_queda_sin_anotaciones() -> None:
    backend = FakeBackend({})

    instances = prelabel.process_photo(np.zeros((*SHAPE, 3), np.uint8), backend, PROMPTS, 1.5, 0.0005)

    assert instances == []
    assert prelabel.build_coco({"a.jpg": (SHAPE, instances)})["annotations"] == []


# --- tandas con retome ---------------------------------------------------------------------------------


def test_una_corrida_procesa_todo_y_otra_no_repite_nada(tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "raw")
    backend = FakeBackend()

    first = prelabel.run_batch(photos, tmp_path / "out", backend, PROMPTS, SETTINGS, log=lambda _: None)
    second = prelabel.run_batch(photos, tmp_path / "out", backend, PROMPTS, SETTINGS, log=lambda _: None)

    assert first["procesadas"] == 4 and first["faltan"] == 0
    assert second["procesadas"] == 0 and second["ya_hechas"] == 4
    assert backend.calls == 4


def test_max_images_deja_el_resto_para_la_proxima_tanda(tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "raw", 5)
    backend = FakeBackend()

    first = prelabel.run_batch(
        photos, tmp_path / "out", backend, PROMPTS, SETTINGS, max_images=2, log=lambda _: None
    )
    second = prelabel.run_batch(
        photos, tmp_path / "out", backend, PROMPTS, SETTINGS, max_images=2, log=lambda _: None
    )
    third = prelabel.run_batch(
        photos, tmp_path / "out", backend, PROMPTS, SETTINGS, max_images=2, log=lambda _: None
    )

    assert (first["procesadas"], first["faltan"]) == (2, 3)
    assert (second["procesadas"], second["faltan"]) == (2, 1)
    assert (third["procesadas"], third["faltan"]) == (1, 0)
    assert backend.calls == 5


def test_si_se_corta_a_la_mitad_se_retoma_donde_quedo(tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "raw", 4)

    class Flaky(FakeBackend):
        def segment(self, image: np.ndarray, prompts: list[str]) -> dict[str, list[tuple[float, np.ndarray]]]:
            if self.calls == 2:
                raise KeyboardInterrupt
            return super().segment(image, prompts)

    with pytest.raises(KeyboardInterrupt):
        prelabel.run_batch(photos, tmp_path / "out", Flaky(), PROMPTS, SETTINGS, log=lambda _: None)
    backend = FakeBackend()

    resumed = prelabel.run_batch(photos, tmp_path / "out", backend, PROMPTS, SETTINGS, log=lambda _: None)

    assert resumed["ya_hechas"] == 2 and resumed["procesadas"] == 2 and backend.calls == 2


def test_si_cambian_los_prompts_se_vuelve_a_procesar(tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "raw", 2)
    backend = FakeBackend()
    prelabel.run_batch(photos, tmp_path / "out", backend, PROMPTS, SETTINGS, log=lambda _: None)

    other = {"curb": ("curb",), "curb_cut": ("driveway",)}
    again = prelabel.run_batch(photos, tmp_path / "out", backend, other, SETTINGS, log=lambda _: None)

    assert again["procesadas"] == 2 and again["ya_hechas"] == 0


def test_una_foto_ilegible_se_cuenta_y_no_frena_la_tanda(tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "raw", 2)
    (tmp_path / "raw" / "rota.jpg").write_bytes(b"no es una imagen")

    stats = prelabel.run_batch(
        [*photos, tmp_path / "raw" / "rota.jpg"],
        tmp_path / "out",
        FakeBackend(),
        PROMPTS,
        SETTINGS,
        log=lambda _: None,
    )

    assert stats["procesadas"] == 2 and stats["ilegibles"] == 1 and stats["faltan"] == 0


def test_la_configuracion_cambia_el_hash() -> None:
    base = prelabel.config_hash(PROMPTS, 0.3, 0.5, 1.5, 0.0005)

    assert base == prelabel.config_hash(PROMPTS, 0.3, 0.5, 1.5, 0.0005)
    assert base != prelabel.config_hash(PROMPTS, 0.4, 0.5, 1.5, 0.0005)
    assert base != prelabel.config_hash({**PROMPTS, "curb": ("kerb",)}, 0.3, 0.5, 1.5, 0.0005)


# --- línea de comandos -----------------------------------------------------------------------------------


def run_main(
    root: Path, extra: list[str] | None = None, backend: FakeBackend | None = None
) -> tuple[int, FakeBackend]:
    shared = backend or FakeBackend()
    code = prelabel.main(["--data-dir", str(root), "--device", "cpu", *(extra or [])], lambda _: shared)
    return code, shared


def test_main_genera_el_coco_el_zip_y_las_muestras(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 4)

    code, _ = run_main(root, ["--samples", "2"])

    out = root / "curb" / "prelabel"
    assert code == 0
    coco = json.loads((out / "annotations" / "instances_default.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 4 and len(coco["annotations"]) == 12
    with zipfile.ZipFile(out / "prelabel_coco.zip") as archive:
        assert archive.namelist() == ["annotations/instances_default.json"]
    assert len(list((out / "samples").glob("*.jpg"))) == 2
    shown = capsys.readouterr().out
    assert "COCO para CVAT" in shown and "'instancias'" in shown


def test_main_por_tandas_y_el_coco_crece(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 4)
    backend = FakeBackend()

    run_main(root, ["--max-images", "1"], backend)
    first = json.loads(
        (root / "curb" / "prelabel" / "annotations" / "instances_default.json").read_text(encoding="utf-8")
    )
    run_main(root, ["--max-images", "2"], backend)
    second = json.loads(
        (root / "curb" / "prelabel" / "annotations" / "instances_default.json").read_text(encoding="utf-8")
    )

    assert len(first["images"]) == 1 and len(second["images"]) == 3 and backend.calls == 3


def test_main_con_prompts_propios(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 1)
    backend = FakeBackend({"kerb": [(0.9, box_mask(40, 46, 5, 95))]})

    run_main(root, ["--curb-prompts", "kerb", "--curb-cut-prompts", "lowered kerb"], backend)

    coco = json.loads(
        (root / "curb" / "prelabel" / "annotations" / "instances_default.json").read_text(encoding="utf-8")
    )
    assert [(a["category_id"], a["attributes"]["prompt"]) for a in coco["annotations"]] == [(1, "kerb")]


def test_el_modo_explorar_compara_los_prompts_y_guarda_superposiciones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 3)

    code, _ = run_main(root, ["--explore", "dropped curb", "driveway", "--explore-photos", "2"])

    assert code == 0
    shown = capsys.readouterr().out
    assert "'dropped curb': {'fotos_con_resultado': 2, 'instancias': 2, 'confianza_media': 0.8}" in shown
    assert "'driveway': {'fotos_con_resultado': 0, 'instancias': 0, 'confianza_media': None}" in shown
    assert len(list((root / "curb" / "prelabel" / "explore").glob("*.jpg"))) == 4
    assert not (root / "curb" / "prelabel" / "annotations").exists()  # explorar no genera el COCO


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = prelabel.main(
        ["--images", str(REPO_ROOT / "x"), "--out-dir", str(REPO_ROOT / "y")], lambda _: FakeBackend()
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_main_sin_fotos_o_sin_carpeta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "vacia").mkdir()

    assert (
        prelabel.main(
            ["--images", str(tmp_path / "vacia"), "--out-dir", str(tmp_path / "o")], lambda _: FakeBackend()
        )
        == 1
    )
    assert "No hay fotos" in capsys.readouterr().err
    assert (
        prelabel.main(
            ["--images", str(tmp_path / "nada"), "--out-dir", str(tmp_path / "o")], lambda _: FakeBackend()
        )
        == 1
    )
    assert "No existe la carpeta" in capsys.readouterr().err


def test_si_no_se_puede_cargar_sam3_se_explica(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 1)

    def broken(_: argparse.Namespace) -> Any:
        raise OSError("Cannot access gated repo")

    code = prelabel.main(["--data-dir", str(root), "--device", "cpu"], broken)

    assert code == 1
    assert "No se pudo cargar SAM 3 (OSError): Cannot access gated repo" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--threshold", "--mask-threshold"])
def test_los_umbrales_estan_entre_cero_y_uno(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        prelabel.parse_args(["--data-dir", str(tmp_path), flag, "1.5"])


def test_los_prompts_por_defecto_estan_en_ingles() -> None:
    assert prelabel.DEFAULT_PROMPTS["curb"] == ("curb",)
    assert "dropped curb" in prelabel.DEFAULT_PROMPTS["curb_cut"]


def test_si_el_segmentador_informa_su_pico_de_memoria_se_imprime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 1)

    class WithMemory(FakeBackend):
        def peak_memory_mb(self) -> int:
            return 3857

    run_main(root, backend=WithMemory())

    assert "'pico_memoria_gpu_mb': 3857" in capsys.readouterr().out


def test_con_fotos_csv_se_saltea_lo_descartado_y_lo_que_no_figura(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 4)  # f0..f3.jpg
    header = "archivo,tipo,rebaje,parecido,iluminacion,clima,oclusion,notas"
    rows = [
        "f0.jpg,positivo_claro,garage,no_aplica,dia,seco,ninguna,",
        "f1.jpg,descartada,desconocido,no_aplica,desconocido,desconocido,desconocido,repetida",
        "f2.jpeg,negativo_auto_sin_rebaje,ninguno,no_aplica,dia,seco,ninguna,",  # el CSV y la foto difieren en la extensión
    ]
    (root / "curb" / "fotos.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    backend = FakeBackend()

    code, _ = run_main(root, [], backend)

    assert code == 0 and backend.calls == 2  # f0 y f2; f1 está descartada y f3 no figura
    coco = json.loads(
        (root / "curb" / "prelabel" / "annotations" / "instances_default.json").read_text(encoding="utf-8")
    )
    assert sorted(i["file_name"].split("/")[-1] for i in coco["images"]) == ["f0.jpg", "f2.jpg"]
    assert "2 fotos quedan afuera" in capsys.readouterr().out


def test_un_fotos_csv_con_errores_se_informa(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "data"
    make_photos(root / "curb" / "raw", 1)
    (root / "curb" / "fotos.csv").write_text(
        "archivo,tipo,rebaje,parecido,iluminacion,clima,oclusion,notas\nf0.jpg,raro,garage,,dia,seco,ninguna,\n",
        encoding="utf-8",
    )

    code, _ = run_main(root, [])

    assert code == 1 and "fila 2" in capsys.readouterr().err
