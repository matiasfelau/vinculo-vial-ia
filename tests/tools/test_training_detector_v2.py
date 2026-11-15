"""Detector v2: clases y jerarquía de Open Images, armado del COCO (con fotos de fondo) y exclusiones.

No necesita FiftyOne, PyTorch ni red: trabaja sobre estructuras chicas armadas en el test.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "detector_v2"))
import v2_classes as classes  # noqa: E402
import v2_download as dl  # noqa: E402

CACHE_HIERARCHY = Path("D:/vv-eval/cardd/fiftyone-cache/open-images-v7/train/metadata/hierarchy.json")


def box(label: str, **flags: bool) -> dl.Box:
    return dl.Box(label, 0.1, 0.1, 0.5, 0.5, **flags)


# --- clases -----------------------------------------------------------------------------------------


def test_las_clases_tienen_el_orden_del_manifest_y_sin_rostro_ni_rueda() -> None:
    assert tuple(classes.CLASSES[:5]) == ("car", "truck", "van", "bus", "plate")
    assert tuple(classes.PRESENCE) == ("car", "truck", "van", "bus")
    assert set(classes.CONFOUNDERS) == {
        "airplane",
        "boat",
        "bicycle",
        "motorcycle",
        "wheelchair",
        "traffic_sign",
        "clock",
    }
    assert "face" not in classes.CLASSES and "wheel" not in classes.CLASSES
    assert len(set(classes.CLASSES)) == len(classes.CLASSES) == 12


def test_toda_clase_de_open_images_mapea_a_una_clase_de_v2_y_toda_clase_de_v2_tiene_origen() -> None:
    assert set(classes.OPEN_IMAGES_CLASSES.values()) == set(classes.CLASSES)
    for names, *_ in classes.STAGES.values():
        assert all(name in classes.OPEN_IMAGES_CLASSES for name in names)


def test_las_etapas_cubren_cada_clase_de_v2() -> None:
    staged = {classes.OPEN_IMAGES_CLASSES[n] for names, *_ in classes.STAGES.values() for n in names}

    assert staged == set(classes.CLASSES)


@pytest.mark.skipif(not CACHE_HIERARCHY.is_file(), reason="no está el caché de Open Images de la máquina")
def test_las_subclases_de_la_jerarquia_de_open_images_estan_mapeadas() -> None:
    tree = json.loads(CACHE_HIERARCHY.read_text(encoding="utf-8"))

    def find(node: dict, name: str) -> dict | None:  # type: ignore[type-arg]
        if node["LabelName"] == name:
            return node
        for child in node.get("Subcategory", []):
            found = find(child, name)
            if found:
                return found
        return None

    def descendants(node: dict) -> list[str]:  # type: ignore[type-arg]
        children = node.get("Subcategory", [])
        return [c["LabelName"] for c in children] + [n for c in children for n in descendants(c)]

    for root in ("Car", "Truck", "Bus", "Boat", "Traffic sign", "Clock", "Airplane", "Bicycle", "Motorcycle"):
        node = find(tree, root)
        assert node is not None
        for name in (root, *descendants(node)):
            assert name in classes.OPEN_IMAGES_CLASSES, f"{name} (hijo de {root}) no está mapeado"
    for name in ("Taxi", "Ambulance", "Wheelchair"):  # hermanas de Car en `Land vehicle`
        assert name in classes.OPEN_IMAGES_CLASSES
    assert classes.OPEN_IMAGES_CLASSES["Van"] == "van" and classes.OPEN_IMAGES_CLASSES["Limousine"] == "car"


def test_la_familia_vehiculo_de_las_fotos_de_fondo_incluye_las_clases_de_v2() -> None:
    blocked = classes.BACKGROUND_EXCLUDED

    assert {"Car", "Truck", "Wheel", "Tire", "Vehicle registration plate", "Motorcycle", "Train"} <= blocked
    assert set(classes.OPEN_IMAGES_CLASSES) <= blocked


# --- COCO ----------------------------------------------------------------------------------------------


def test_to_coco_conserva_las_fotos_de_fondo_sin_anotaciones_y_descarta_las_de_etapa_sin_cajas_utiles() -> (
    None
):
    records = [
        dl.ImageRecord("a.jpg", 100, 100, [box("Truck"), box("Vehicle registration plate")]),
        dl.ImageRecord("fondo.jpg", 100, 100, []),
        dl.ImageRecord("grupo.jpg", 100, 100, [box("Car", group_of=True)]),  # solo cajas ambiguas
    ]

    coco = dl.to_coco(records)

    assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "fondo.jpg"]
    assert {a["image_id"] for a in coco["annotations"]} == {1}  # la de fondo no tiene cajas
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert [names[a["category_id"]] for a in coco["annotations"]] == ["truck", "plate"]
    assert [c["name"] for c in coco["categories"]] == list(classes.CLASSES)


def test_to_coco_descarta_cajas_diminutas_y_con_atributos_ambiguos() -> None:
    tiny = dl.Box("Car", 0.1, 0.1, 0.05, 0.5)  # 5 px de ancho en una foto de 100
    record = dl.ImageRecord(
        "a.jpg", 100, 100, [tiny, box("Clock", depiction=True), box("Bus", inside=True), box("Bicycle")]
    )

    coco = dl.to_coco([record])
    counts = dl.class_counts(coco)

    assert counts["bicycle"] == 1 and sum(counts.values()) == 1


def test_to_coco_asigna_ids_en_el_orden_de_las_clases() -> None:
    coco = dl.to_coco([dl.ImageRecord("a.jpg", 100, 100, [box("Taxi"), box("Ambulance"), box("Gondola")])])
    names = {c["id"]: c["name"] for c in coco["categories"]}

    assert [names[a["category_id"]] for a in coco["annotations"]] == ["car", "van", "boat"]
    assert coco["categories"][0]["id"] == 1


# --- elección de fotos -------------------------------------------------------------------------------


def annotations() -> dl.Annotations:
    boxes: dict[str, list[dl.Box]] = {}
    for stage, (names, train, _valid) in classes.STAGES.items():
        for i in range(train * 2 + 5):  # más candidatas que las pedidas
            boxes[f"{stage}-{i}"] = [box(names[0])]
    all_images = set(boxes) | {f"fondo-{i}" for i in range(2000)} | {f"auto-{i}" for i in range(50)}
    vehicles = {f"auto-{i}" for i in range(50)}  # tienen una rueda o un tren: no pueden ser fondo
    return dl.Annotations(boxes, vehicles, all_images)


def test_se_eligen_las_cantidades_de_cada_etapa_sin_repetir_fotos() -> None:
    chosen = dl.select_images(annotations(), set(), split_index=0, seed=0)

    for stage, (_names, train, _valid) in classes.STAGES.items():
        assert len(chosen[stage]) == train
    assert len(chosen["backgrounds"]) == classes.BACKGROUNDS[0]
    flat = [i for ids in chosen.values() for i in ids]
    assert len(flat) == len(set(flat))


def test_la_validacion_usa_sus_cantidades_y_la_escala_achica_todo() -> None:
    valid = dl.select_images(annotations(), set(), split_index=1, seed=0)
    small = dl.select_images(annotations(), set(), split_index=0, seed=0, scale=0.1)

    assert len(valid["plates"]) == classes.STAGES["plates"][2]
    assert len(small["plates"]) == round(classes.STAGES["plates"][1] * 0.1)
    assert all(len(ids) >= 1 for ids in small.values())


def test_las_fotos_excluidas_nunca_se_eligen() -> None:
    ann = annotations()
    banned = {f"plates-{i}" for i in range(0, 600)} | {f"fondo-{i}" for i in range(0, 1500)}

    chosen = dl.select_images(ann, banned, split_index=0, seed=0)

    assert not banned & {i for ids in chosen.values() for i in ids}
    assert all(i.startswith("fondo-") for i in chosen["backgrounds"])


def test_el_fondo_no_tiene_vehiculos_ni_clases_de_v2() -> None:
    ann = annotations()

    chosen = dl.select_images(ann, set(), split_index=0, seed=0)

    assert not set(chosen["backgrounds"]) & (ann.vehicle_images | set(ann.boxes))


def test_la_eleccion_es_reproducible_con_la_misma_semilla_y_cambia_con_otra() -> None:
    a = dl.select_images(annotations(), set(), 0, seed=1)
    b = dl.select_images(annotations(), set(), 0, seed=1)
    c = dl.select_images(annotations(), set(), 0, seed=2)

    assert a == b and a != c


def test_una_etapa_ignora_las_fotos_con_solo_cajas_ambiguas() -> None:
    ann = dl.Annotations({"x": [box("Truck", group_of=True)], "y": [box("Truck")]}, set(), {"x", "y"})

    assert dl.stage_candidates(ann, ("Truck",)) == ["y"]


# --- exclusiones -------------------------------------------------------------------------------------


def write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_las_negativas_de_v1_son_las_fotos_sin_auto_rueda_ni_patente(tmp_path: Path) -> None:
    for split, (annotated, images) in {
        "train": ("car", ["a.jpg", "b.jpg"]),
        "test": ("face", ["c.jpg"]),
    }.items():
        folder = tmp_path / split
        folder.mkdir()
        coco = {
            "images": [
                {"id": n, "file_name": name, "width": 10, "height": 10} for n, name in enumerate(images)
            ],
            "categories": [{"id": 1, "name": annotated}],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 1}
            ],  # solo la primera foto de cada split tiene su clase
        }
        (folder / dl.ANNOTATIONS).write_text(json.dumps(coco))

    assert dl.v1_negative_ids(tmp_path) == {"b", "c"}  # `a` tiene un auto; `c` solo un rostro


def test_la_verificacion_atrapa_la_misma_foto_por_id_por_contenido_y_por_md5(tmp_path: Path) -> None:
    excluded_file = write(tmp_path / "ex" / "e1.jpg", b"contenido-de-e1")
    same_bytes = write(tmp_path / "sel" / "otra-id.jpg", b"contenido-de-e1")  # otro id, mismo archivo
    same_id = write(tmp_path / "sel" / "e1b.jpg", b"distinto")
    md5_twin = write(tmp_path / "sel" / "gemela.jpg", b"re-encodeada")
    clean = write(tmp_path / "sel" / "limpia.jpg", b"nada-que-ver")
    excluded = {"prueba_de_v1": {"e1", "e1b"}}
    selected = {"otra-id": same_bytes, "e1b": same_id, "gemela": md5_twin, "limpia": clean}
    md5s = {"e1": "MD5-X", "gemela": "MD5-X", "limpia": "MD5-Y", "otra-id": "MD5-Z"}

    check = dl.verify_exclusion(selected, excluded, {"e1": excluded_file}, md5s)

    assert check["coincidencias_por_identificador"] == ["e1b"]
    assert check["coincidencias_por_sha256"] == ["otra-id"]
    assert check["coincidencias_por_md5_original"] == ["gemela"]
    assert check["ok"] is False
    assert check["excluidas_sin_archivo_cantidad"] == 1  # `e1b` no tiene archivo para comparar por contenido


def test_la_verificacion_da_ok_si_no_hay_coincidencias(tmp_path: Path) -> None:
    excluded_file = write(tmp_path / "ex" / "e1.jpg", b"uno")
    clean = write(tmp_path / "sel" / "x.jpg", b"dos")

    check = dl.verify_exclusion({"x": clean}, {"g": {"e1"}}, {"e1": excluded_file}, {"e1": "A", "x": "B"})

    assert (
        check["ok"] is True
        and check["elegidas"] == 1
        and check["archivos_excluidos_comparados_por_contenido"] == 1
    )


def test_excluded_ids_junta_los_tres_grupos(tmp_path: Path) -> None:
    data = tmp_path / "data"
    write(data / dl.V1_DATASET / "test" / "t1.jpg", b"1")
    fresh = tmp_path / "fresh"
    write(fresh / "f1.jpg", b"2")
    write(fresh / "f2.jpg", b"3")

    groups = dl.excluded_ids(data, fresh)

    assert groups["prueba_de_v1"] == {"t1"} and groups["negativas_nuevas"] == {"f1", "f2"}
    assert groups["negativas_de_v1"] == set()


def test_file_sha256_es_el_del_contenido(tmp_path: Path) -> None:
    path = write(tmp_path / "a.jpg", b"abc")

    assert dl.file_sha256(path) == hashlib.sha256(b"abc").hexdigest()
