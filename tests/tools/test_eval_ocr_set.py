"""tools/eval/ocr_set.py: el conjunto de evaluación del OCR con fotos sintéticas (sin datos reales ni OCR)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_ocr_set", REPO_ROOT / "tools" / "eval" / "ocr_set.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_ocr_set"] = module
    spec.loader.exec_module(module)
    return module


ocr_set = load()

PLATE_W, PLATE_H = 160, 60
BOX = (80.0, 90.0, float(PLATE_W), float(PLATE_H))
MOTO_BOX = (100.0, 80.0, 70.0, 70.0)  # casi cuadrada: relación ancho/alto de 1,0
TEXTS = [
    "AB123CD",
    "KLM456",
    "XY789ZW",
    "QRS012",
    "TUV345",
    "GHJ678",
    "NPQ901",
    "DEF234",
    "WXY567",
    "ZAB890",
    "CDE123",
    "FGH456",
]


def plate_image(text: str) -> np.ndarray:
    plate = np.full((PLATE_H, PLATE_W, 3), 235, np.uint8)
    cv2.putText(plate, text, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return plate


def moto_plate_image(text: str) -> np.ndarray:
    plate = np.full((70, 70, 3), 235, np.uint8)
    cv2.putText(plate, text[:3], (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(plate, text[3:6], (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return plate


def photo(
    text: str, night: bool = False, seed: int = 0, scale: float = 1.0, moto: bool = False
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = 25 if night else 190
    blocks = rng.uniform(level - 20, level + 20, (6, 8, 1)).astype(
        np.float32
    )  # un fondo distinto por semilla
    background = cv2.resize(blocks, (320, 240), interpolation=cv2.INTER_CUBIC)[..., None] * np.ones(
        3, np.float32
    )
    image = np.clip(background + rng.normal(0, 6, (240, 320, 3)), 0, 255).astype(np.uint8)
    for _ in range(5):  # objetos al azar: que las fotos de distinta semilla no se parezcan en miniatura
        bx, by = int(rng.integers(0, 260)), int(rng.integers(0, 190))
        image[by : by + int(rng.integers(20, 50)), bx : bx + int(rng.integers(30, 60))] = int(
            rng.integers(0, 255)
        )
    x, y, w, h = (int(v) for v in (MOTO_BOX if moto else BOX))
    image[y : y + h, x : x + w] = moto_plate_image(text) if moto else plate_image(text)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def write_export(root: Path, photos: dict[str, Any]) -> None:
    """Un export COCO de un solo split: `photos` mapea el nombre a (imagen, escala[, cajas])."""
    (root / "train").mkdir(parents=True)
    images, annotations = [], []
    for number, (name, value) in enumerate(photos.items()):
        image, scale = value[0], value[1]
        boxes = value[2] if len(value) > 2 else [BOX]
        assert cv2.imwrite(str(root / "train" / name), image)
        images.append({"id": number, "file_name": name, "width": image.shape[1], "height": image.shape[0]})
        for index, box in enumerate(boxes):
            annotations.append(
                {
                    "id": number * 10 + index,
                    "image_id": number,
                    "category_id": 1,
                    "bbox": [v * scale for v in box],
                }
            )
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "patente"}]}
    (root / "train" / ocr_set.ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")


def described(
    root: Path, name: str, image: np.ndarray, scale: float = 1.0, box: tuple[float, ...] = BOX
) -> Any:
    write_export(root, {name: (image, scale, [box])})
    return ocr_set.describe(root, ocr_set.load_annotations(root))[0]


def test_la_caja_se_agranda_un_15_por_ciento_a_cada_lado_y_se_recorta_a_la_foto() -> None:
    assert ocr_set.pixel_box((100.0, 50.0, 200.0, 100.0), 1000, 1000, 0.15) == (70, 35, 330, 165)
    assert ocr_set.pixel_box((5.0, 5.0, 100.0, 40.0), 120, 60, 0.15) == (0, 0, 120, 51)


def test_la_misma_patente_a_otra_escala_es_duplicada_y_una_distinta_no(tmp_path: Path) -> None:
    original = described(tmp_path / "a", "a.jpg", photo("AB123CD", seed=1))
    smaller = described(tmp_path / "b", "b.jpg", photo("AB123CD", seed=2, scale=0.6), scale=0.6)
    other = described(tmp_path / "c", "c.jpg", photo("KLM456", seed=3))
    assert ocr_set.same_plate(original, smaller)
    assert not ocr_set.same_plate(original, other)


def test_de_cada_grupo_de_duplicados_queda_el_de_mayor_resolucion(tmp_path: Path) -> None:
    write_export(
        tmp_path,
        {
            "grande.jpg": (photo("AB123CD", seed=1), 1.0),
            "chica.jpg": (photo("AB123CD", seed=2, scale=0.6), 0.6),
            "otra.jpg": (photo("KLM456", seed=3), 1.0),
        },
    )
    items = ocr_set.describe(tmp_path, ocr_set.load_annotations(tmp_path))
    kept, dropped = ocr_set.dedupe(items)
    assert sorted(item.annotation.file for item in kept) == ["grande.jpg", "otra.jpg"]
    assert [(gone.annotation.file, keep.annotation.file) for gone, keep in dropped] == [
        ("chica.jpg", "grande.jpg")
    ]


def test_el_brillo_de_la_escena_no_cuenta_la_patente(tmp_path: Path) -> None:
    day = ocr_set.surround_brightness(cv2.cvtColor(photo("AB123CD"), cv2.COLOR_BGR2GRAY), BOX)
    night = ocr_set.surround_brightness(cv2.cvtColor(photo("AB123CD", night=True), cv2.COLOR_BGR2GRAY), BOX)
    assert day is not None and night is not None
    assert day > 150 and night < 60
    assert ocr_set.estimate_light(day) == ocr_set.LIGHT_DAY
    assert ocr_set.estimate_light(night) == ocr_set.LIGHT_NIGHT


def test_en_un_primer_plano_no_hay_escena_para_juzgar_la_luz() -> None:
    close_up = np.full((60, 160), 20, np.uint8)
    assert ocr_set.surround_brightness(close_up, (0.0, 0.0, 160.0, 60.0)) is None
    assert ocr_set.estimate_light(None) == ocr_set.LIGHT_DAY


def _pool(tmp_path: Path, nights: int, days: int) -> list[Any]:
    pool = []
    for number in range(nights + days):
        item = described(tmp_path / str(number), "x.jpg", photo(TEXTS[number % len(TEXTS)], seed=number))
        item.light = ocr_set.LIGHT_NIGHT if number < nights else ocr_set.LIGHT_DAY
        pool.append(item)
    return pool


def test_la_seleccion_es_reproducible_y_tiene_al_menos_las_de_noche_pedidas(tmp_path: Path) -> None:
    pool = _pool(tmp_path, nights=4, days=8)
    first = ocr_set.select(pool, count=6, night_min=3, seed=7)
    again = ocr_set.select(pool, count=6, night_min=3, seed=7)
    assert [id(item) for item in first] == [id(item) for item in again]
    assert len({id(item) for item in first}) == 6
    assert sum(item.light == ocr_set.LIGHT_NIGHT for item in first) >= 3
    other = ocr_set.select(pool, count=6, night_min=3, seed=8)
    assert [id(item) for item in other] != [id(item) for item in first]


def test_la_seleccion_falla_si_no_alcanzan_las_de_noche_o_las_candidatas(tmp_path: Path) -> None:
    pool = _pool(tmp_path, nights=2, days=3)
    with pytest.raises(ValueError, match="de noche"):
        ocr_set.select(pool, count=4, night_min=3, seed=1)
    with pytest.raises(ValueError, match="candidatas"):
        ocr_set.select(pool, count=9, night_min=1, seed=1)


def test_el_armado_completo_descarta_cada_cosa_y_deja_las_etiquetas_vacias(tmp_path: Path) -> None:
    src, reference, out = tmp_path / "src", tmp_path / "reference", tmp_path / "out"
    photos = {f"p{n}.jpg": (photo(TEXTS[n], night=n < 3, seed=n), 1.0) for n in range(8)}
    photos["copia.jpg"] = (photo(TEXTS[4], seed=40, scale=0.6), 0.6)  # la misma patente que p4, más chica
    photos["ya_calibrada.jpg"] = (photo(TEXTS[8], seed=50), 1.0)  # la misma foto que está en la referencia
    photos["patente_conocida.jpg"] = (photo(TEXTS[9], seed=60), 1.0)  # su patente está en la referencia
    photos["chica.jpg"] = (np.full((60, 80, 3), 120, np.uint8), 0.1)  # caja de 16x6 píxeles
    write_export(src, photos)
    write_export(
        reference,
        {
            "ref1.jpg": (photo(TEXTS[8], seed=50), 1.0),
            "ref2.jpg": (photo(TEXTS[9], seed=99, night=True), 1.0),
        },
    )

    report = ocr_set.run(src, reference, out, count=6, night_min=2, seed=3)

    assert report["caja_chica"] == 1
    assert report["foto_en_dataset_de_calibracion"] == 1
    assert report["patente_en_dataset_de_calibracion"] == 1
    assert report["duplicados"] == 1
    assert report["candidatas"] == 8
    assert report["elegidas"] == 6
    assert report["elegidas_de_noche"] >= 2
    with (out / "etiquetas.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["id", "archivo", "recorte", "texto", "formato", "luz", "estado"]
    assert [row["id"] for row in rows] == ["001", "002", "003", "004", "005", "006"]
    assert all(row["texto"] == row["formato"] == row["estado"] == "" for row in rows)
    assert {row["luz"] for row in rows} <= {"dia", "noche"}
    assert len({row["archivo"] for row in rows}) == 6
    assert not any(row["archivo"] in ("train/copia.jpg", "train/ya_calibrada.jpg") for row in rows)
    for row in rows:
        assert (out / row["recorte"]).is_file()
    assert (out / "hojas" / "hoja-01.jpg").is_file()
    reasons = {r["motivo"] for r in csv.DictReader((out / "descartes.csv").open(encoding="utf-8-sig"))}
    assert reasons == {
        "caja_chica",
        "foto_en_dataset_de_calibracion",
        "patente_en_dataset_de_calibracion",
        "duplicado",
    }
    assert json.loads((out / "informe.json").read_text(encoding="utf-8"))["parametros"]["semilla"] == 3


def test_el_recorte_tiene_el_margen_de_15_por_ciento(tmp_path: Path) -> None:
    src, reference, out = tmp_path / "src", tmp_path / "reference", tmp_path / "out"
    write_export(src, {f"p{n}.jpg": (photo(TEXTS[n], seed=n), 1.0) for n in range(3)})
    write_export(reference, {"r.jpg": (photo(TEXTS[11], seed=77), 1.0)})
    ocr_set.run(src, reference, out, count=3, night_min=0, seed=1)
    crop = cv2.imdecode(np.fromfile(str(out / "recortes" / "001.png"), np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (round(PLATE_H * 1.3), round(PLATE_W * 1.3))


def moto_photos(first_seed: int, texts: list[str]) -> dict[str, Any]:
    return {
        f"moto{n}.jpg": (photo(text, seed=first_seed + n, moto=True), 1.0, [MOTO_BOX])
        for n, text in enumerate(texts)
    }


def test_una_caja_con_otra_adentro_o_vista_a_ojo_es_invalida(tmp_path: Path) -> None:
    inner = (120.0, 100.0, 80.0, 30.0)
    outer = (60.0, 70.0, 220.0, 120.0)
    write_export(
        tmp_path,
        {
            "hoja.jpg": (photo("AB123CD", seed=1), 1.0, [outer, inner]),
            "sola.jpg": (photo("KLM456", seed=2), 1.0),
        },
    )
    annotations = ocr_set.load_annotations(tmp_path)
    invalid = ocr_set.invalid_boxes(annotations)
    assert {(a.file, a.box) for a in invalid} == {("hoja.jpg", outer)}
    pile = ocr_set.Annotation("train", "17397969544621434_jpg.rf.abc.jpg", (32.0, 13.0, 142.0, 212.0))
    assert ocr_set.invalid_boxes([pile]) == {pile}


def test_la_caja_se_clasifica_como_invalida_moto_o_de_auto() -> None:
    plate = ocr_set.Annotation("train", "a.jpg", (0.0, 0.0, 200.0, 100.0))  # 2,0
    edge = ocr_set.Annotation("train", "b.jpg", (0.0, 0.0, 180.0, 100.0))  # justo 1,8
    moto = ocr_set.Annotation("train", "c.jpg", (0.0, 0.0, 179.0, 100.0))
    assert ocr_set.classify(plate, set()) is None
    assert ocr_set.classify(edge, set()) is None
    assert ocr_set.classify(moto, set()) == ocr_set.REASON_MOTORCYCLE
    assert ocr_set.classify(plate, {plate}) == ocr_set.REASON_INVALID
    assert ocr_set.classify(moto, {moto}) == ocr_set.REASON_INVALID  # inválida gana a moto


def test_el_armado_nuevo_no_deja_motos(tmp_path: Path) -> None:
    src, reference, out = tmp_path / "src", tmp_path / "reference", tmp_path / "out"
    photos = {f"p{n}.jpg": (photo(TEXTS[n], seed=n), 1.0) for n in range(4)}
    photos["moto.jpg"] = (photo(TEXTS[5], seed=30, moto=True), 1.0, [MOTO_BOX])
    write_export(src, photos)
    write_export(reference, {"r.jpg": (photo(TEXTS[11], seed=77), 1.0)})
    report = ocr_set.run(src, reference, out, count=4, night_min=0, seed=1)
    assert report["probable_moto"] == 1
    assert report["elegidas_probable_moto"] == 0
    with (out / "cajas.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert all(float(row["relacion_aspecto"]) >= 1.8 for row in csv.DictReader(handle))


def test_el_reemplazo_toma_el_id_de_la_que_sale_y_prioriza_las_de_noche(tmp_path: Path) -> None:
    moto = described(tmp_path / "m", "m.jpg", photo(TEXTS[0], seed=1, moto=True), box=MOTO_BOX)
    car = described(tmp_path / "c", "c.jpg", photo(TEXTS[1], seed=2))
    day = described(tmp_path / "d", "d.jpg", photo(TEXTS[2], seed=3))
    night = described(tmp_path / "n", "n.jpg", photo(TEXTS[3], night=True, seed=4))
    for item in (moto, car, day, night):
        item.light = ocr_set.estimate_light(item.surround)
    swaps = ocr_set.plan_swaps([car, moto], [car, moto, day, night], set(), seed=1)
    assert [(s.ident, s.reason, s.replacement is night) for s in swaps] == [
        ("002", ocr_set.REASON_MOTORCYCLE, True)
    ]


def test_sin_candidatas_de_auto_alcanzan_el_reemplazo_falla(tmp_path: Path) -> None:
    moto = described(tmp_path / "m", "m.jpg", photo(TEXTS[0], seed=1, moto=True), box=MOTO_BOX)
    with pytest.raises(ValueError, match="candidatas de auto"):
        ocr_set.plan_swaps([moto], [moto], set(), seed=1)


def _old_set(tmp_path: Path) -> tuple[Path, Path, Path, Any]:
    """Un set armado como el anterior a la limpieza: con motos entre las elegidas."""
    src, reference, out = tmp_path / "src", tmp_path / "reference", tmp_path / "out"
    photos: dict[str, Any] = {f"p{n}.jpg": (photo(TEXTS[n], night=n < 2, seed=n), 1.0) for n in range(6)}
    photos.update(moto_photos(10, TEXTS[6:10]))
    write_export(src, photos)
    write_export(reference, {"r.jpg": (photo(TEXTS[11], seed=77), 1.0)})
    built = ocr_set.build_pool(src, reference)
    first = ocr_set.select(built.items, 6, 1, 5)
    ocr_set.build_outputs(src, [(f"{n:03d}", item) for n, item in enumerate(first, start=1)], out)
    return src, reference, out, first


def test_la_limpieza_reemplaza_las_motos_y_conserva_los_demas_ids(tmp_path: Path) -> None:
    src, reference, out, first = _old_set(tmp_path)
    before = list(csv.DictReader((out / "etiquetas.csv").open(encoding="utf-8-sig")))
    motos = [n for n, item in enumerate(first) if ocr_set.aspect(item.annotation) < 1.8]
    assert motos  # con esta semilla, al menos una moto entre las elegidas

    summary = ocr_set.refresh(src, reference, out, count=6, night_min=1, seed=5)

    after = list(csv.DictReader((out / "etiquetas.csv").open(encoding="utf-8-sig")))
    assert summary["reemplazadas"] == len(motos)
    assert summary["elegidas_probable_moto"] == 0
    assert [row["id"] for row in after] == [f"{n:03d}" for n in range(1, 7)]
    for n, (old, new) in enumerate(zip(before, after, strict=True)):
        assert (old == new) == (n not in motos)
    assert len({row["archivo"] for row in after}) == 6
    backup = out.with_name(out.name + ocr_set.BACKUP_SUFFIX)
    assert list(csv.DictReader((backup / "etiquetas.csv").open(encoding="utf-8-sig"))) == before
    swaps = list(csv.DictReader((out / "reemplazos.csv").open(encoding="utf-8-sig")))
    assert [row["id"] for row in swaps] == [f"{n + 1:03d}" for n in motos]
    assert {row["motivo"] for row in swaps} == {ocr_set.REASON_MOTORCYCLE}
    assert json.loads((out / "informe.json").read_text(encoding="utf-8"))["limpieza"]["reemplazadas"] == len(
        motos
    )


def test_la_limpieza_no_pisa_etiquetas_cargadas_ni_un_respaldo_existente(tmp_path: Path) -> None:
    src, reference, out, _ = _old_set(tmp_path)
    original = (out / "etiquetas.csv").read_bytes()
    (out.with_name(out.name + ocr_set.BACKUP_SUFFIX)).mkdir()
    with pytest.raises(ValueError, match="ya existe"):
        ocr_set.refresh(src, reference, out, count=6, night_min=1, seed=5)
    assert (out / "etiquetas.csv").read_bytes() == original

    labeled = original.replace(b"recortes/001.png,,", b"recortes/001.png,ABC123,")
    assert labeled != original
    (out / "etiquetas.csv").write_bytes(labeled)
    with pytest.raises(ValueError, match="etiquetas cargadas"):
        ocr_set.refresh(src, reference, out, count=6, night_min=1, seed=5)
    assert (out / "etiquetas.csv").read_bytes() == labeled


def test_se_niega_a_leer_o_escribir_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    inside = REPO_ROOT / "ocr-set-prueba"
    code = ocr_set.main(["--src", str(inside), "--reference", str(inside), "--out", str(inside)])
    assert code == 1
    assert "dentro del repositorio" in capsys.readouterr().out
    assert not inside.exists()


def test_no_pisa_unas_etiquetas_existentes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "etiquetas.csv").write_text("id,texto\n001,ABC123\n", encoding="utf-8")
    code = ocr_set.main(["--src", str(tmp_path), "--reference", str(tmp_path), "--out", str(out)])
    assert code == 1
    assert "no se pisa" in capsys.readouterr().out
    assert (out / "etiquetas.csv").read_text(encoding="utf-8") == "id,texto\n001,ABC123\n"
