"""tools/training/reid: ángulos de vista, partición agrupada, pares y métricas de verificación."""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDER = REPO_ROOT / "tools" / "training" / "reid"


def load(name: str) -> ModuleType:
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location(f"reid_{name}", FOLDER / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"reid_{name}"] = module
    spec.loader.exec_module(module)
    return module


dataset = load("dataset")
metrics = load("metrics")


def pose(x: float, z: float, look_x: float = 0.0, look_z: float = 0.0, y: float = 1.2) -> list[float]:
    """Pose cámara→mundo (por filas) en `(x, y, z)` mirando por su eje `-z` hacia `(look_x, look_z)`."""
    forward = np.array([look_x - x, 0.0, look_z - z])
    forward /= np.linalg.norm(forward)
    back = -forward  # el eje +z de la cámara apunta hacia atrás
    right = np.cross([0.0, 1.0, 0.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, up, back, [x, y, z]
    return [float(v) for v in matrix.reshape(-1)]


def ring(
    degrees: list[float], radius: float = 4.0, center: tuple[float, float] = (0.0, 0.0)
) -> list[list[float]]:
    return [
        pose(
            center[0] + radius * math.sin(math.radians(d)),
            center[1] + radius * math.cos(math.radians(d)),
            center[0],
            center[1],
        )
        for d in degrees
    ]


def car(
    name: str, lighting: str = "standard", kind: str = "SUV", brand: str = "A", color: str = "blanco"
) -> object:
    return dataset.Car(name, brand, "m", lighting, kind, color)


# --- ángulos --------------------------------------------------------------------------


def test_azimut_recupera_los_angulos_de_una_caminata_alrededor_del_auto() -> None:
    angles = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]

    found = dataset.azimuths(ring(angles, center=(3.0, -2.0)))

    # el azimut es relativo a un eje arbitrario, pero las diferencias se conservan
    for a, b, f, g in zip(angles, angles[1:], found, found[1:], strict=False):
        assert dataset.angle_difference(f, g) == pytest.approx(dataset.angle_difference(a, b), abs=0.5)


def test_azimut_aguanta_una_caminata_incompleta_y_ruido_en_la_mirada() -> None:
    angles = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    poses = ring(angles)

    found = dataset.azimuths(poses)

    assert dataset.angle_difference(found[0], found[-1]) == pytest.approx(100.0, abs=1.0)


def test_azimut_con_pocas_poses_cae_al_promedio_sin_fallar() -> None:
    found = dataset.azimuths(ring([0.0, 90.0]))

    assert len(found) == 2
    assert all(0.0 <= value < 360.0 for value in found)


def test_pose_con_largo_incorrecto() -> None:
    with pytest.raises(ValueError, match="16"):
        dataset.azimuths([[1.0, 2.0]])


def test_diferencia_de_angulos_da_la_vuelta() -> None:
    assert dataset.angle_difference(350.0, 10.0) == pytest.approx(20.0)
    assert dataset.angle_difference(0.0, 180.0) == pytest.approx(180.0)
    assert dataset.angle_difference(90.0, 90.0) == 0.0


# --- partición ------------------------------------------------------------------------


def make_cars(n: int = 200) -> list[object]:
    return [
        car(f"c{i:04d}", lighting=("standard", "reflective", "dark")[i % 3], kind=("SUV", "Sedan")[i % 2])
        for i in range(n)
    ]


def test_particion_agrupada_cada_auto_en_una_sola_y_reproducible() -> None:
    cars = make_cars()

    first = dataset.group_split(cars, seed=7)
    again = dataset.group_split(cars, seed=7)
    other = dataset.group_split(cars, seed=8)

    assert set(first) == {c.name for c in cars}  # todos asignados, una vez cada uno
    assert set(first.values()) == {"train", "val", "test"}
    assert first == again
    assert first != other


def test_particion_estratificada_cada_luz_esta_en_las_tres_particiones() -> None:
    cars = make_cars()
    split = dataset.group_split(cars, seed=0)

    for lighting in ("standard", "reflective", "dark"):
        seen = {split[c.name] for c in cars if c.lighting == lighting}
        assert seen == {"train", "val", "test"}
    sizes = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
    assert sizes["train"] > sizes["val"] and sizes["train"] > sizes["test"]


def test_particion_con_fracciones_invalidas() -> None:
    with pytest.raises(ValueError, match="sumar 1"):
        dataset.group_split(make_cars(10), fractions=(0.5, 0.2, 0.2))


# --- pares ----------------------------------------------------------------------------


def views_of(name: str, degrees: list[float]) -> list[object]:
    return [dataset.View(name, i, d) for i, d in enumerate(degrees)]


def test_pares_del_mismo_auto_respetan_la_separacion_minima() -> None:
    views = views_of("a", [0.0, 10.0, 40.0, 200.0])

    pairs = dataset.same_car_pairs(views, min_separation=30.0)

    assert all(angle >= 30.0 for _, _, angle in pairs)
    assert (0, 1, pytest.approx(10.0)) not in pairs  # 10° no llega
    assert {(i, j) for i, j, _ in pairs} == {(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_rangos_de_angulo() -> None:
    assert dataset.angle_bin(45.0) == "30-60"
    assert dataset.angle_bin(60.0) == "60-90"
    assert dataset.angle_bin(179.9) == "135-180"
    assert dataset.angle_bin(180.0) == "135-180"
    assert dataset.angle_bin(10.0) is None


def test_pares_de_autos_distintos_nunca_son_del_mismo_auto() -> None:
    views = views_of("a", [0, 90, 180]) + views_of("b", [0, 90, 180]) + views_of("c", [0, 90])
    cars = {n: car(n) for n in "abc"}

    pairs = dataset.different_car_pairs(views, cars, limit=20, seed=1)

    assert pairs and all(views[i].car != views[j].car for i, j in pairs)
    assert pairs == dataset.different_car_pairs(views, cars, limit=20, seed=1)


def test_pares_dificiles_son_de_la_misma_marca_y_el_mismo_color() -> None:
    views = [dataset.View(n, 0, 0.0) for n in ("a", "b", "c", "d")]
    cars = {
        "a": car("a", brand="X", color="rojo"),
        "b": car("b", brand="X", color="rojo"),
        "c": car("c", brand="X", color="azul"),
        "d": car("d", brand="Y", color="rojo"),
    }

    pairs = dataset.different_car_pairs(views, cars, limit=10, seed=0, hard=True)

    assert pairs and {frozenset((views[i].car, views[j].car)) for i, j in pairs} == {frozenset("ab")}


def test_sin_autos_dificiles_no_hay_pares() -> None:
    views = [dataset.View(n, 0, 0.0) for n in ("a", "b")]
    cars = {"a": car("a", brand="X"), "b": car("b", brand="Y")}

    assert dataset.different_car_pairs(views, cars, limit=5, hard=True) == []


# --- métricas -------------------------------------------------------------------------


def test_auc_perfecta_inversa_y_empates() -> None:
    assert metrics.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert metrics.auc([0.1], [0.9]) == 0.0
    assert metrics.auc([0.5], [0.5]) == 0.5
    assert metrics.auc([], [0.5]) is None


def test_cota_del_95_con_cero_fallas_es_3_sobre_n() -> None:
    assert metrics.upper_bound(0, 300) == pytest.approx(0.01)
    assert metrics.upper_bound(0, 0) is None
    assert metrics.upper_bound(5, 5) == 1.0


def test_cota_clopper_pearson_con_fallas() -> None:
    # valor de referencia exacto: 1 falla en 100, cota superior de una cola del 95 % = 4,66 %
    assert metrics.upper_bound(1, 100) == pytest.approx(0.0466, abs=5e-4)
    assert metrics.upper_bound(10, 1000) > 0.01


def test_u1_y_s4_a_un_umbral() -> None:
    same = np.array([0.9, 0.8, 0.3, 0.7])
    different = np.array([0.1, 0.2, 0.6, 0.0])

    row = metrics.at_threshold(same, different, 0.5)

    assert row["U1"] == pytest.approx(0.25) and row["U1_pares"] == 1
    assert row["S4"] == pytest.approx(0.75)  # 0,1, 0,2 y 0,0 quedan bajo 0,5


def test_calibracion_deja_la_cota_de_u1_bajo_el_maximo() -> None:
    rng = np.random.default_rng(0)
    same = rng.normal(0.8, 0.08, 5000)
    different = rng.normal(0.2, 0.15, 5000)

    result = metrics.calibrate(same, different, max_u1=0.01)

    assert result["U1_cota_95"] <= 0.01
    assert result["U1"] <= 0.01
    assert result["S4"] > 0.99  # clases muy separadas: casi todo lo distinto queda bajo el umbral


def test_calibracion_con_pocos_pares_no_puede_cumplir() -> None:
    result = metrics.calibrate(np.linspace(0.5, 0.9, 100), np.zeros(10), max_u1=0.01)

    assert result["umbral"] is None and "supera" in result["motivo"]


def test_calibracion_sin_pares() -> None:
    assert metrics.calibrate(np.array([]), np.array([0.1]))["umbral"] is None


def test_histograma_cuenta_todo() -> None:
    assert sum(metrics.histogram(np.linspace(-1, 1, 101))) == 101


# --- fetch.py (sin red) ---------------------------------------------------------------


fetch = load("fetch")


def test_spread_reparte_parejo_y_no_repite() -> None:
    frames = list(range(1, 201))

    chosen = fetch.spread(frames, 12)

    assert len(chosen) == 12 and chosen == sorted(set(chosen))
    gaps = np.diff(chosen)
    assert gaps.min() >= 14 and gaps.max() <= 18  # 200 / 12 ≈ 16,7
    assert fetch.spread([5, 3, 9], 12) == [3, 5, 9]  # menos fotos que las pedidas: todas, en orden


def test_select_cars_toma_todos_los_oscuros_y_reflectivos_y_una_muestra_de_estandar() -> None:
    lighting = {f"a{i}": "standard" for i in range(50)} | {f"b{i}": "dark" for i in range(5)}
    lighting |= {f"c{i}": "reflective" for i in range(7)}

    chosen = fetch.select_cars(lighting, list(lighting), standard=10, seed=1)

    assert sum(lighting[c] == "standard" for c in chosen) == 10
    assert (
        sum(lighting[c] == "dark" for c in chosen) == 5
        and sum(lighting[c] == "reflective" for c in chosen) == 7
    )
    assert chosen == fetch.select_cars(lighting, list(lighting), standard=10, seed=1)
    assert chosen != fetch.select_cars(lighting, list(lighting), standard=10, seed=2)


def sheet_xlsx(path: Path, header: dict[str, str], rows: list[dict[str, str]]) -> None:
    """Un xlsx mínimo con cadenas compartidas (lo que lee `parse_annotations`)."""
    import zipfile

    strings: list[str] = []

    def string(text: str) -> int:
        if text not in strings:
            strings.append(text)
        return strings.index(text)

    def row(cells: dict[str, str], number: int) -> str:
        parts = "".join(f'<c r="{c}{number}" t="s"><v>{string(v)}</v></c>' for c, v in cells.items())
        return f'<row r="{number}">{parts}</row>'

    body = "".join(row(r, n) for n, r in enumerate([header, *rows], 1))
    sheet = f"<worksheet><sheetData>{body}</sheetData></worksheet>"
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
        archive.writestr("xl/sharedStrings.xml", f"<sst>{shared}</sst>")


def test_parse_annotations_lee_luz_tipo_y_color(tmp_path: Path) -> None:
    rows = [
        {"B": "2024_04_11_11_06_05", "C": "BYD", "D": "Song", "E": "reflective", "F": "SUV", "G": "gris"},
        {"B": "2024_04_11_11_07_05", "C": "VW", "D": "Golf", "E": "standard", "F": "Sedan", "G": "blanco"},
        {"B": "no-es-un-auto", "C": "x", "D": "y", "E": "standard", "F": "SUV", "G": "z"},
        {"B": "2024_04_11_11_08_05", "C": "x", "D": "y", "E": "luz-rara", "F": "SUV", "G": "z"},
    ]
    sheet_xlsx(tmp_path / "a.xlsx", fetch.HEADER, rows)

    annotations = fetch.parse_annotations(tmp_path / "a.xlsx")

    assert set(annotations) == {"2024_04_11_11_06_05", "2024_04_11_11_07_05"}
    assert annotations["2024_04_11_11_06_05"] == {
        "marca": "BYD",
        "modelo": "Song",
        "luz": "reflective",
        "tipo": "Suv",
        "color": "gris",
    }


def test_parse_annotations_falla_si_el_encabezado_cambio(tmp_path: Path) -> None:
    sheet_xlsx(tmp_path / "b.xlsx", {"B": "otra cosa"}, [])

    with pytest.raises(fetch.FetchError, match="encabezado"):
        fetch.parse_annotations(tmp_path / "b.xlsx")


def test_read_entry_verifica_el_crc_y_descomprime(monkeypatch: pytest.MonkeyPatch) -> None:
    import struct
    import zlib

    payload = b"hola" * 100
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    packed = compressor.compress(payload) + compressor.flush()
    name = "auto/frame_00001.json"
    header = struct.pack(
        "<4s5H3L2H",
        b"PK\x03\x04",
        20,
        0,
        8,
        0,
        0,
        zlib.crc32(payload),
        len(packed),
        len(payload),
        len(name),
        4,
    )
    block = header + name.encode() + b"\0\0\0\0" + packed + b"RESTO"
    monkeypatch.setattr(fetch, "http_range", lambda *_: block)

    assert fetch.read_entry("u", 0, len(packed), 8, zlib.crc32(payload), name) == payload
    with pytest.raises(fetch.FetchError, match="CRC"):
        fetch.read_entry("u", 0, len(packed), 8, 123, name)
    monkeypatch.setattr(fetch, "http_range", lambda *_: b"XXXX" + block[4:])
    with pytest.raises(fetch.FetchError, match="encabezado"):
        fetch.read_entry("u", 0, len(packed), 8, zlib.crc32(payload), name)


def test_build_pairs_arma_los_tres_conjuntos() -> None:
    items = []
    for car, brand in (("a", "X"), ("b", "X"), ("c", "Y")):
        for frame, angle in enumerate((0.0, 100.0, 200.0, 300.0)):
            items.append(
                {
                    "auto": car,
                    "foto": frame,
                    "azimut": angle,
                    "marca": brand,
                    "modelo": "m",
                    "luz": "standard",
                    "tipo": "SUV",
                    "color": "rojo",
                }
            )

    pairs = dataset.build_pairs(items, different_limit=50, hard_limit=20, seed=0)

    assert pairs.same and all(items[i]["auto"] == items[j]["auto"] and a >= 30.0 for i, j, a in pairs.same)
    assert pairs.different and all(items[i]["auto"] != items[j]["auto"] for i, j in pairs.different)
    assert pairs.hard and {frozenset((items[i]["auto"], items[j]["auto"])) for i, j in pairs.hard} == {
        frozenset("ab")
    }


def test_la_particion_de_un_estrato_no_cambia_si_se_suman_autos_de_otros() -> None:
    standard = [car(f"s{i:03d}", lighting="standard") for i in range(60)]
    dark = [car(f"d{i:03d}", lighting="dark") for i in range(40)]
    reflective = [car(f"r{i:03d}", lighting="reflective") for i in range(40)]

    before = dataset.group_split(standard + dark, seed=3)
    after = dataset.group_split(standard + dark + reflective, seed=3)

    assert all(after[name] == split for name, split in before.items())
    assert {after[c.name] for c in reflective} == {"train", "val", "test"}


def test_is_complete_detecta_jpg_y_json_a_medias(tmp_path: Path) -> None:
    good, cut = tmp_path / "a.jpg", tmp_path / "b.jpg"
    good.write_bytes(b"\xff\xd8datos\xff\xd9")
    cut.write_bytes(b"\xff\xd8datos")
    ok_json, bad_json, empty = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    ok_json.write_text('{"x": 1}')
    bad_json.write_text('{"x": ')
    empty.write_bytes(b"")

    assert fetch.is_complete(good) and not fetch.is_complete(cut)
    assert fetch.is_complete(ok_json) and not fetch.is_complete(bad_json) and not fetch.is_complete(empty)
    assert not fetch.is_complete(tmp_path / "no-existe.jpg")


def test_azimut_sin_poses_es_una_lista_vacia() -> None:
    assert dataset.azimuths([]) == []
