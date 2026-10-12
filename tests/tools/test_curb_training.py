"""Rebajes: validación de fotos.csv, reparto por escena y armado del dataset, con datos sintéticos.

No hay fotos reales: las imágenes son arreglos lisos y las anotaciones, polígonos inventados.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CURB = REPO_ROOT / "tools" / "training" / "curb"


def load(name: str) -> ModuleType:
    if str(CURB) not in sys.path:
        sys.path.insert(0, str(CURB))
    spec = importlib.util.spec_from_file_location(f"curb_{name}", CURB / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # photos.py y splits.py se importan entre sí por su nombre corto
    spec.loader.exec_module(module)
    return module


photos_mod = load("photos")
splits_mod = load("splits")
prepare = load("prepare")

HEADER = ",".join(photos_mod.COLUMNS)


def row(
    archivo: str = "a.jpg",
    escena: str = "",
    tipo: str = "positivo_claro",
    rebaje: str = "garage",
    parecido: str = "",
    iluminacion: str = "dia",
    clima: str = "seco",
    oclusion: str = "ninguna",
    notas: str = "",
) -> str:
    return ",".join(
        [
            archivo,
            escena,
            tipo,
            rebaje,
            parecido,
            iluminacion,
            clima,
            oclusion,
            notas,
        ]
    )


def write_csv(path: Path, rows: list[str], header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


# --- fotos.csv --------------------------------------------------------------------------------


def test_carga_fotos_validas_y_excluye_las_descartadas(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fotos.csv",
        [
            row("a.jpg"),
            row("b.jpg", tipo="descartada"),
            row("c.jpg", tipo="negativo_auto_sin_rebaje", rebaje="ninguno"),
        ],
    )

    photos = photos_mod.load_photos(path)

    assert [p.file for p in photos] == ["a.jpg", "c.jpg"]
    assert photos[0].kind == "positivo_claro" and photos[0].lighting == "dia" and photos[0].scene == ""


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("tipo", "positivo_dudoso"),
        ("rebaje", "puerta"),
        ("iluminacion", "mediodia"),
        ("clima", "nublado"),
        ("oclusion", "total"),
    ],
)
def test_un_valor_desconocido_se_rechaza_con_su_numero_de_fila(
    tmp_path: Path, column: str, value: str
) -> None:
    path = write_csv(tmp_path / "fotos.csv", [row("a.jpg"), row("b.jpg", **{column: value})])

    with pytest.raises(photos_mod.PhotosError, match=rf"fila 3: .*{column} desconocido '{value}'"):
        photos_mod.load_photos(path)


def test_se_listan_todos_los_problemas_y_no_solo_el_primero(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fotos.csv",
        [row("a.jpg", clima="x"), row("b.jpg"), row("c.jpg", rebaje="z", oclusion="y")],
    )

    with pytest.raises(photos_mod.PhotosError) as error:
        photos_mod.load_photos(path)

    assert len(error.value.problems) == 2
    assert error.value.problems[0].startswith("fila 2:") and error.value.problems[1].startswith("fila 4:")
    assert (
        "rebaje desconocido" in error.value.problems[1] and "oclusion desconocido" in error.value.problems[1]
    )


def test_parecido_solo_con_negativo_parecido(tmp_path: Path) -> None:
    ok = row("a.jpg", tipo="negativo_parecido", rebaje="ninguno", parecido="boca_tormenta")
    sobra = row("b.jpg", parecido="cordon_roto")
    falta = row("c.jpg", tipo="negativo_parecido", rebaje="ninguno")
    desconocido = row("d.jpg", tipo="negativo_parecido", rebaje="ninguno", parecido="alcantarilla")
    path = write_csv(tmp_path / "fotos.csv", [ok, sobra, falta, desconocido])

    with pytest.raises(photos_mod.PhotosError) as error:
        photos_mod.load_photos(path)

    assert [p.split(":")[0] for p in error.value.problems] == ["fila 3", "fila 4", "fila 5"]
    assert "solo se usa con tipo negativo_parecido" in error.value.problems[0]
    assert "falta parecido" in error.value.problems[1]
    assert "parecido desconocido 'alcantarilla'" in error.value.problems[2]


def test_el_parecido_de_una_descartada_tambien_se_valida(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "fotos.csv", [row("a.jpg", tipo="descartada", parecido="cordon_roto")])

    with pytest.raises(photos_mod.PhotosError, match="solo se usa"):
        photos_mod.load_photos(path)


def test_el_archivo_es_obligatorio_y_la_escena_opcional(tmp_path: Path) -> None:
    with pytest.raises(photos_mod.PhotosError, match="falta archivo"):
        photos_mod.load_photos(write_csv(tmp_path / "a.csv", [row("")]))

    photos = photos_mod.load_photos(write_csv(tmp_path / "b.csv", [row("a.jpg"), row("b.jpg", escena="e01")]))

    assert [p.scene for p in photos] == ["", "e01"]


def test_las_columnas_son_las_nuevas_y_en_ese_orden() -> None:
    assert photos_mod.COLUMNS == (
        "archivo",
        "escena",
        "tipo",
        "rebaje",
        "parecido",
        "iluminacion",
        "clima",
        "oclusion",
        "notas",
    )


def test_un_csv_con_las_columnas_viejas_se_rechaza(tmp_path: Path) -> None:
    old = "archivo,fecha,barrio,calle,tipo,rebaje,parecido,iluminacion,clima,oclusion,dispositivo,notas"

    with pytest.raises(photos_mod.PhotosError, match="faltan: escena"):
        photos_mod.load_photos(write_csv(tmp_path / "fotos.csv", [], header=old))


def test_una_escena_comparte_rebaje_iluminacion_y_clima(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fotos.csv",
        [
            row("a.jpg", escena="e01"),
            row("b.jpg", escena="E01"),  # mayúsculas: es la misma escena
            row("c.jpg", escena="e01", rebaje="accesibilidad", iluminacion="noche"),
            row("d.jpg", escena="e01", tipo="descartada", clima="lluvia"),  # las descartadas no cuentan
            row("e.jpg", escena="e02", rebaje="accesibilidad"),
        ],
    )

    with pytest.raises(photos_mod.PhotosError) as error:
        photos_mod.load_photos(path)

    assert len(error.value.problems) == 1
    problem = error.value.problems[0]
    assert problem.startswith("fila 4:") and "la escena 'e01' tiene rebaje 'accesibilidad'" in problem
    assert "fila 2" in problem and "iluminacion 'noche'" in problem


def test_foto_repetida(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "fotos.csv", [row("a.jpg"), row("a.jpg")])

    with pytest.raises(
        photos_mod.PhotosError, match=r"fila 3: la foto está repetida \(ya figura en la fila 2\)"
    ):
        photos_mod.load_photos(path)


def test_con_carpeta_comprueba_que_las_fotos_existan(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.jpg").write_bytes(b"x")
    path = write_csv(tmp_path / "fotos.csv", [row("a.jpg"), row("b.jpg")])

    with pytest.raises(photos_mod.PhotosError, match="fila 3: la foto no está en la carpeta"):
        photos_mod.load_photos(path, raw)
    assert [p.file for p in photos_mod.load_photos(path)] == ["a.jpg", "b.jpg"]


def test_faltan_columnas_o_el_archivo(tmp_path: Path) -> None:
    with pytest.raises(photos_mod.PhotosError, match="faltan: parecido"):
        photos_mod.load_photos(write_csv(tmp_path / "f.csv", [], header=HEADER.replace(",parecido", "")))
    with pytest.raises(photos_mod.PhotosError, match="no existe"):
        photos_mod.load_photos(tmp_path / "nada.csv")


def test_filtra_por_tipo(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "fotos.csv", [row("a.jpg"), row("b.jpg", tipo="positivo_marginal")])

    assert [p.file for p in photos_mod.load_photos(path, keep={"positivo_marginal"})] == ["b.jpg"]


def test_el_grupo_es_la_escena_sin_distinguir_mayusculas_o_la_foto_misma() -> None:
    def photo(file: str, scene: str) -> object:
        return photos_mod.CurbPhoto(file, scene, "positivo_claro", "garage", "", "dia", "seco", "ninguna", "")

    a, b, c = photo("a", "E01"), photo("b", "e01"), photo("c", "e02")
    solo, other_solo = photo("x", ""), photo("y", "")

    assert a.group == b.group != c.group
    assert solo.group != other_solo.group and solo.group != a.group


# --- reparto -------------------------------------------------------------------------------------

KINDS = [
    "positivo_claro",
    "positivo_marginal",
    "negativo_auto_junto_rebaje",
    "negativo_auto_sin_rebaje",
    "negativo_rebaje_sin_auto",
    "negativo_parecido",
]


def make_photos(scenes: int = 30, per_scene: int = 4) -> list:
    """`scenes` escenas de `per_scene` fotos cada una."""
    out = []
    for scene in range(scenes):
        for index in range(per_scene):
            kind = KINDS[(scene + index) % len(KINDS)]
            out.append(
                photos_mod.CurbPhoto(
                    f"c{scene:02d}_{index}.jpg", f"e{scene:02d}", kind, "garage",
                    "boca_tormenta" if kind == "negativo_parecido" else "",
                    "dia", "seco", "ninguna", "",
                )
            )  # fmt: skip
    return out


def single_photos(count: int = 60) -> list:
    """Fotos sin escena."""
    return [
        photos_mod.CurbPhoto(
            f"u{index:03d}.jpg", "", KINDS[index % len(KINDS)], "garage",
            "boca_tormenta" if KINDS[index % len(KINDS)] == "negativo_parecido" else "",
            "dia", "seco", "ninguna", "",
        )
        for index in range(count)
    ]  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_ninguna_escena_esta_en_dos_conjuntos(seed: int) -> None:
    assignment = splits_mod.assign(make_photos(), seed)

    groups = {name: {p.group for p in photos} for name, photos in assignment.splits.items()}
    assert not (groups["train"] & groups["valid"]) and not (groups["train"] & groups["test"])
    assert not (groups["valid"] & groups["test"])
    assert sum(len(v) for v in assignment.splits.values()) == 120
    assert {p.file for v in assignment.splits.values() for p in v} == {p.file for p in make_photos()}


def test_las_fotos_de_una_escena_van_juntas_y_las_sueltas_se_reparten_solas() -> None:
    photos = [*make_photos(scenes=12, per_scene=5), *single_photos(60)]

    assignment = splits_mod.assign(photos, 0)

    where = {p.file: name for name, v in assignment.splits.items() for p in v}
    for scene in range(12):
        assert len({where[f"c{scene:02d}_{i}.jpg"] for i in range(5)}) == 1
    assert len({where[f"u{i:03d}.jpg"] for i in range(60)}) == 3  # sin escena: cada foto va por su cuenta
    sizes = {name: len(v) for name, v in assignment.splits.items()}
    assert 0.5 <= sizes["train"] / 120 <= 0.7 and sizes["valid"] > 0 and sizes["test"] > 0


def test_las_proporciones_son_aproximadamente_60_20_20() -> None:
    sizes = {name: len(photos) for name, photos in splits_mod.assign(make_photos(), 0).splits.items()}

    assert 0.52 <= sizes["train"] / 120 <= 0.68
    assert 0.12 <= sizes["valid"] / 120 <= 0.28
    assert 0.12 <= sizes["test"] / 120 <= 0.28


def test_cada_tipo_esta_repartido_en_todos_los_conjuntos() -> None:
    assignment = splits_mod.assign(make_photos(), 0)

    for photos in assignment.splits.values():
        assert {p.kind for p in photos} == set(KINDS)
    totals = {kind: sum(p.kind == kind for v in assignment.splits.values() for p in v) for kind in KINDS}
    for kind in KINDS:
        share = sum(p.kind == kind for p in assignment.splits["train"]) / totals[kind]
        assert 0.4 <= share <= 0.8  # estratificado: ningún tipo queda concentrado


def test_el_reparto_es_reproducible_y_cambia_con_la_semilla() -> None:
    first = splits_mod.assign(make_photos(), 3)
    again = splits_mod.assign(make_photos(), 3)
    other = splits_mod.assign(make_photos(), 4)

    names = lambda a: {k: [p.file for p in v] for k, v in a.splits.items()}  # noqa: E731
    assert names(first) == names(again)
    assert names(first) != names(other)


def test_con_menos_de_tres_grupos_no_se_puede_repartir() -> None:
    with pytest.raises(splits_mod.SplitError, match="al menos 3 grupos"):
        splits_mod.assign(make_photos(scenes=2), 0)


def test_el_test_se_guarda_la_primera_vez_y_no_se_pisa(tmp_path: Path) -> None:
    assignment = splits_mod.assign(make_photos(), 0)

    splits_mod.save_test(tmp_path, assignment, 0)
    saved = (tmp_path / "test.json").read_bytes()
    other = splits_mod.assign(make_photos(), 5)
    splits_mod.save_test(tmp_path, other, 5)

    assert (tmp_path / "test.json").read_bytes() == saved
    data = splits_mod.load_fixed_test(tmp_path)
    assert data is not None and set(data["archivos"]) == {p.file for p in assignment.splits["test"]}
    assert splits_mod.load_fixed_test(tmp_path / "nada") is None


def test_con_el_test_guardado_las_fotos_nuevas_no_lo_cambian(tmp_path: Path) -> None:
    first = splits_mod.assign(make_photos(), 0)
    splits_mod.save_test(tmp_path, first, 0)
    fixed = splits_mod.load_fixed_test(tmp_path)
    more = [*make_photos(scenes=40), *single_photos(5)]  # llegan 10 escenas y 5 fotos sueltas

    second = splits_mod.assign(more, 9, fixed)

    assert [p.file for p in second.splits["test"]] == [p.file for p in first.splits["test"]]
    new_groups = {p.group for p in more} - {p.group for p in make_photos()}
    assert len(new_groups) == 15 and not (new_groups & {p.group for p in second.splits["test"]})
    train_valid = {p.group for p in second.splits["train"]} & {p.group for p in second.splits["valid"]}
    assert not train_valid and not second.left_out


def test_una_foto_nueva_de_una_escena_del_test_queda_afuera_de_todo(tmp_path: Path) -> None:
    first = splits_mod.assign(make_photos(), 0)
    splits_mod.save_test(tmp_path, first, 0)
    test_scene = first.splits["test"][0].scene
    extra = photos_mod.CurbPhoto(
        "nueva.jpg", test_scene, "positivo_claro", "garage", "", "dia", "seco", "ninguna", ""
    )

    second = splits_mod.assign([*make_photos(), extra], 0, splits_mod.load_fixed_test(tmp_path))

    assert [p.file for p in second.left_out] == ["nueva.jpg"]
    assert "nueva.jpg" not in {p.file for v in second.splits.values() for p in v}


def test_el_test_guardado_nombra_los_grupos(tmp_path: Path) -> None:
    splits_mod.save_test(tmp_path, splits_mod.assign([*make_photos(10), *single_photos(30)], 0), 0)

    saved = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))

    assert set(saved) == {"semilla", "grupos", "archivos"}
    assert all(g.startswith(("escena:", "foto:")) for g in saved["grupos"])


def test_el_resumen_y_las_asignaciones(tmp_path: Path) -> None:
    assignment = splits_mod.assign(make_photos(), 0)

    summary = assignment.summary()
    splits_mod.write_assignments(tmp_path, assignment)

    assert summary["train"]["fotos"] + summary["valid"]["fotos"] + summary["test"]["fotos"] == 120
    assert summary["fuera_del_test"] == 0
    saved = json.loads((tmp_path / "assignments.json").read_text(encoding="utf-8"))
    assert set(saved) == {"train", "valid", "test", "fuera_del_test"}


# --- prepare.py ----------------------------------------------------------------------------------


def make_curb(
    tmp_path: Path,
    scenes: int = 6,
    reverse_ids: bool = False,
    rle: bool = False,
    skip_in_export: str | None = None,
) -> Path:
    """`<root>/curb/` con fotos.csv, raw/ y una exportación de CVAT sintética."""
    curb = tmp_path / "data" / "curb"
    (curb / "raw").mkdir(parents=True)
    (curb / "cvat-export" / "annotations").mkdir(parents=True)
    rows, images, annotations = [], [], []
    ids = {"curb": 2, "curb_cut": 1} if reverse_ids else {"curb": 1, "curb_cut": 2}
    for scene in range(scenes):
        for index in range(3):
            name = f"s{scene}_{index}.jpg"
            kind = KINDS[(scene + index) % len(KINDS)]
            cv2.imwrite(str(curb / "raw" / name), np.full((40, 60, 3), 30 * (scene + 1), np.uint8))
            rows.append(
                row(
                    name,
                    escena=f"e{scene}",
                    tipo=kind,
                    rebaje="garage",
                    parecido="boca_tormenta" if kind == "negativo_parecido" else "",
                )
            )
            if name == skip_in_export:
                continue
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": f"default/{name}", "width": 60, "height": 40})
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": ids["curb"],
                    "segmentation": [[2, 30, 58, 30, 58, 36, 2, 36]],
                    "bbox": [2, 30, 56, 6],
                    "area": 336,
                    "iscrowd": 0,
                }
            )
            if kind in (
                "positivo_claro",
                "positivo_marginal",
                "negativo_auto_junto_rebaje",
                "negativo_rebaje_sin_auto",
            ):
                segmentation = (
                    {"counts": [0, 10], "size": [40, 60]} if rle else [[10, 30, 30, 30, 30, 36, 10, 36]]
                )
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": ids["curb_cut"],
                        "segmentation": segmentation,
                        "bbox": [10, 30, 20, 6],
                        "area": 120,
                        "iscrowd": 1 if rle else 0,
                    }
                )
    write_csv(curb / "fotos.csv", rows)
    export = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": ids["curb"], "name": "curb"}, {"id": ids["curb_cut"], "name": "curb_cut"}],
    }
    (curb / "cvat-export" / "annotations" / "instances_default.json").write_text(
        json.dumps(export), encoding="utf-8"
    )
    return tmp_path / "data"


def test_prepare_arma_el_dataset_con_las_clases_en_el_orden_del_modelo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_curb(tmp_path, reverse_ids=True)  # CVAT numeró las clases al revés

    assert prepare.main(["--data-dir", str(root)]) == 0

    dataset = root / "curb" / "dataset"
    total = 0
    for split in ("train", "valid", "test"):
        coco = json.loads((dataset / split / prepare.ANNOTATIONS).read_text(encoding="utf-8"))
        assert [c["name"] for c in coco["categories"]] == ["curb", "curb_cut"] and [
            c["id"] for c in coco["categories"]
        ] == [1, 2]
        assert all((dataset / split / image["file_name"]).is_file() for image in coco["images"])
        assert all(isinstance(a["segmentation"], list) and a["iscrowd"] == 0 for a in coco["annotations"])
        total += len(coco["images"])
        curb_ids = {a["category_id"] for a in coco["annotations"] if a["area"] == 336}
        assert curb_ids == {1}  # el cordón es la clase 1 aunque CVAT lo haya numerado 2
    assert total == 18
    assert "Listo" in capsys.readouterr().out


def test_prepare_deja_el_test_guardado_y_no_cambia_al_repetirlo(tmp_path: Path) -> None:
    root = make_curb(tmp_path)
    assert prepare.main(["--data-dir", str(root)]) == 0
    saved = (root / "curb" / "splits" / "test.json").read_bytes()
    first = (root / "curb" / "dataset" / "test" / prepare.ANNOTATIONS).read_bytes()

    assert prepare.main(["--data-dir", str(root), "--seed", "99"]) == 0

    assert (root / "curb" / "splits" / "test.json").read_bytes() == saved
    assert (root / "curb" / "dataset" / "test" / prepare.ANNOTATIONS).read_bytes() == first
    assert (root / "curb" / "splits" / "assignments.json").is_file()


def test_prepare_no_toca_las_fotos_originales(tmp_path: Path) -> None:
    root = make_curb(tmp_path)
    before = {p.name: p.read_bytes() for p in (root / "curb" / "raw").iterdir()}

    prepare.main(["--data-dir", str(root)])

    assert {p.name: p.read_bytes() for p in (root / "curb" / "raw").iterdir()} == before


def test_las_mascaras_rle_se_descartan_y_se_cuentan(tmp_path: Path) -> None:
    root = make_curb(tmp_path, rle=True)

    assert prepare.main(["--data-dir", str(root)]) == 0

    info = json.loads((root / "curb" / "dataset" / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["anotaciones_descartadas"] == {"sin polígono": 12}
    assert info["fotos_sin_curb_cut_esperado"] == 12


def test_prepare_avisa_de_las_fotos_sin_curb_cut_esperado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_curb(tmp_path, rle=True)

    prepare.main(["--data-dir", str(root)])

    assert "no tienen ningún curb_cut anotado" in capsys.readouterr().out


def test_una_foto_del_csv_que_no_esta_en_cvat_es_un_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_curb(tmp_path, skip_in_export="s0_0.jpg")

    assert prepare.main(["--data-dir", str(root)]) == 1
    assert "no están en la exportación de CVAT: s0_0.jpg" in capsys.readouterr().err


def test_un_csv_con_errores_se_informa_fila_por_fila(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_curb(tmp_path)
    path = root / "curb" / "fotos.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",seco,", ",humedo,", 1), encoding="utf-8")

    assert prepare.main(["--data-dir", str(root)]) == 1
    assert "fila 2" in capsys.readouterr().err


def test_la_exportacion_sin_las_clases_es_un_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_curb(tmp_path)
    path = root / "curb" / "cvat-export" / "annotations" / "instances_default.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["categories"] = [{"id": 1, "name": "curb"}]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert prepare.main(["--data-dir", str(root)]) == 1
    assert "no tiene las clases curb_cut" in capsys.readouterr().err


def test_prepare_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare.main(["--data-dir", str(REPO_ROOT / "datos")]) == 1
    assert "repositorio" in capsys.readouterr().err


def test_find_export(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    target = tmp_path / "annotations" / "instances_default.json"
    target.write_text("{}", encoding="utf-8")

    assert prepare.find_export(tmp_path) == target
    assert prepare.find_export(target) == target
    with pytest.raises(prepare.ExportError):
        prepare.find_export(tmp_path / "otra")
