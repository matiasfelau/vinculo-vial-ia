"""Baja de 3DRealCar solo las fotos RGB que hacen falta (sin depth, mallas ni mapas del escáner).

3DRealCar (Du et al., ICCV 2025; Apache-2.0) se distribuye en zips de 55-110 GB con todo el escaneo 3D. Acá
solo se necesita un puñado de fotos por auto: el script lee el índice de cada zip con pedidos por rango y baja
únicamente `frame_N.jpg` y `frame_N.json` (la pose de la cámara) de las vistas elegidas. Nada se guarda dentro
del repositorio. Los zips salen del folder público (`--source public`, sujeto a la cuota de Google) o de una
carpeta de Drive propia con copias (`--source drive`, ver `drive.py`).

    annotations  baja `annotation2000.xlsx` (marca, modelo, luz, tipo y color de cada auto) y lo pasa a
                 `annotations.json`
    index        lee el índice de los zips y arma `index.json` (auto → zip, fotos y posiciones en el zip)
    select       elige los autos (todos los oscuros y reflectivos, y una muestra de los estándar) y `K` fotos
                 de cada uno; no cambia las vistas de los autos ya elegidos
    drive-check  lista la carpeta de Drive y lee unos bytes de cada zip (valida el acceso con casi 0 tráfico)
    verify       compara cada zip de la fuente con `index.json` (cantidad de autos y posición de las vistas)
    pull         baja esas fotos a `raw/<auto>/frame_N.jpg|json`, con CRC y conteo verificados
    status       autos completos por zip y por luz (qué copia ya se puede borrar)

Uso: ver `tools/training/reid/README.md`.
"""

import argparse
import json
import os
import random
import re
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from drive import (
    FOLDER_ENV,
    KEY_ENV,
    REPO_ROOT,
    DriveSource,
    FetchError,
    QuotaExceeded,
    Reader,
    ServiceAccountTokens,
)

DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
# Los zips de autos del folder público de 3DRealCar (id de Google Drive). Los HQ y el de segmentación quedan
# afuera.
ZIPS: Mapping[str, str] = {
    "0-200": "1nsYDY9hdat9LA4rHR0yQUFVfxNUfdChN",
    "200-400": "1XNLu9ThkNYRSNHvWOBYwpc16C7lF7glC",
    "400-600": "1jyvwOD0xTXX0pKaouqHgEGqHZT2mDTDd",
    "600-800": "1pRkrOaJTKEZgV_0B47_wDDxdo711kZzJ",
    "800-1000": "1cOp5SucYoaMAqXThLV-86X8tcRrJhPJy",
    "1000-1200": "1rbHqryca7eJa2tTHlWEjLfJ0RN_zx31n",
    "1200-1400": "1_j-EoY4zj0MPJDTADAsI8xW2geYQUYZS",
    "1400-1600": "1CDpElrbObORD9_Qs1VxDAmHzxY7nLVzP",
    "1600-1800": "1pg9JiNKQm-JYlHZz2hsMYQGU-bOJxuHi",
    "1800-2045": "1uV83nEQLoFIi5wayApIhkqZYWP16l1Ho",
}
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
JPEG_END = bytes((0xFF, 0xD9))  # marca de fin de un JPEG entero
LOCAL_SIGNATURE = b"PK\x03\x04"
HEADER_SLACK = 512  # holgura para el campo "extra" del encabezado local (se desconoce hasta leerlo)
TIMEOUT_S = 120
RETRIES = 6
BACKOFF_S = 2.0  # espera creciente entre reintentos: Google limita los pedidos por rango muy seguidos
FRAME = re.compile(r"^(?P<car>[^/]+)/frame_(?P<n>\d+)\.(?P<ext>jpg|json)$")
CAR_NAME = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
LIGHTING = ("standard", "reflective", "dark")
ANNOTATIONS_ID = "1L5b3QK4MDjNGp_LdCefjC8h1xz4n1EnI"  # annotation2000.xlsx (888 MB: lleva una foto por auto)
# Encabezado de la primera hoja: número, carpeta, marca, modelo, brillo (luz), tipo, color, foto
HEADER = {"B": "数据文件夹名称", "C": "品牌", "D": "车型", "E": "光泽", "F": "车辆类型", "G": "颜色"}


def http_range(url: str, start: int, end: int) -> bytes:
    """Los bytes `[start, end]` (inclusive) de `url`, con reintentos."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})  # noqa: S310 - https fijo
            with urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
                if response.status != 206:
                    raise FetchError(f"el servidor no respondió por rango (HTTP {response.status})")
                return bytes(response.read())
        except Exception as exc:  # noqa: BLE001 - red inestable: se reintenta
            last = exc
            time.sleep(BACKOFF_S * 2**attempt)
    raise FetchError(f"falló {url.split('id=')[1][:8]}… [{start}-{end}]: {type(last).__name__}")


def total_size(url: str) -> int:
    request = Request(url, headers={"Range": "bytes=0-0"})  # noqa: S310
    with urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
        return int(response.headers["Content-Range"].split("/")[1])


class RangeFile:
    """Archivo de solo lectura sobre pedidos por rango (alcanza para que `zipfile` lea el índice)."""

    def __init__(self, read: Reader, size: int) -> None:
        self._read = read
        self.size = size
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, count: int = -1) -> bytes:
        if count < 0 or self.pos + count > self.size:
            count = self.size - self.pos
        if count <= 0:
            return b""
        data = self._read(self.pos, self.pos + count - 1)
        self.pos += len(data)
        return data


class ZipSource(Protocol):
    """De dónde salen los zips: el folder público o una carpeta de Drive propia."""

    bytes_read: int

    def labels(self) -> list[str]: ...

    def size(self, label: str) -> int: ...

    def reader(self, label: str) -> Reader: ...


class PublicSource:
    """Los zips del folder público de 3DRealCar (`ZIPS`), sujetos a la cuota de descarga de Google."""

    def __init__(self) -> None:
        self.bytes_read = 0
        self._lock = threading.Lock()

    def labels(self) -> list[str]:
        return list(ZIPS)

    def size(self, label: str) -> int:
        return total_size(DRIVE_URL.format(id=ZIPS[label]))

    def reader(self, label: str) -> Reader:
        url = DRIVE_URL.format(id=ZIPS[label])

        def read(start: int, end: int) -> bytes:
            data = http_range(url, start, end)
            with self._lock:
                self.bytes_read += len(data)
            return data

        return read


def read_entry(
    read: Reader, offset: int, compress_size: int, compress_type: int, crc: int, name: str
) -> bytes:
    """Contenido de una entrada del zip con un solo pedido: encabezado local y datos juntos."""
    block = read(offset, offset + LOCAL_HEADER.size + len(name) + HEADER_SLACK + compress_size)
    signature, *_, name_len, extra_len = LOCAL_HEADER.unpack_from(block)
    if signature != LOCAL_SIGNATURE:
        raise FetchError(f"encabezado local inválido en {name}")
    start = LOCAL_HEADER.size + name_len + extra_len
    if start + compress_size > len(block):
        block = read(offset, offset + start + compress_size)
    raw = block[start : start + compress_size]
    if compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    elif compress_type != zipfile.ZIP_STORED:
        raise FetchError(f"compresión no soportada ({compress_type}) en {name}")
    if zlib.crc32(raw) != crc:
        raise FetchError(f"CRC distinto en {name}")
    return raw


def index_zip(label: str, source: ZipSource) -> dict[str, Any]:
    """`auto → {zip, fotos: {n: {jpg: entrada, json: entrada}}}` de un zip, leyendo solo su índice."""
    archive = zipfile.ZipFile(RangeFile(source.reader(label), source.size(label)))  # type: ignore[arg-type]
    cars: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        parts = info.filename.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        match = FRAME.match(parts[1])
        if not match or not CAR_NAME.match(match["car"]):
            continue
        entry = [info.header_offset, info.compress_size, info.compress_type, info.CRC, info.filename]
        car = cars.setdefault(match["car"], {"zip": label, "id": ZIPS.get(label, ""), "fotos": {}})
        car["fotos"].setdefault(int(match["n"]), {})[match["ext"]] = entry
    for car in cars.values():  # solo las vistas que tienen foto (hay muchos más json que jpg)
        car["fotos"] = {n: e for n, e in car["fotos"].items() if "jpg" in e and "json" in e}
    return cars


def spread(frames: Sequence[int], count: int) -> list[int]:
    """`count` fotos repartidas parejo en el orden en que se sacaron (la caminata alrededor del auto)."""
    ordered = sorted(frames)
    if len(ordered) <= count:
        return ordered
    step = len(ordered) / count
    return [ordered[int(i * step + step / 2)] for i in range(count)]


def select_cars(
    annotated: Mapping[str, str], available: Sequence[str], standard: int, seed: int
) -> list[str]:
    """Todos los autos oscuros y reflectivos y `standard` de los estándar (sorteados con `seed`)."""
    rng = random.Random(seed)  # noqa: S311 - muestreo de datos, no criptografía
    chosen: list[str] = []
    for lighting in LIGHTING:
        pool = sorted(car for car in available if annotated.get(car) == lighting)
        if lighting == "standard":
            rng.shuffle(pool)
            pool = sorted(pool[:standard])
        chosen += pool
    return chosen


def parse_annotations(xlsx: Path) -> dict[str, dict[str, str]]:
    """`auto → {marca, modelo, luz, tipo, color}` de la primera hoja de `annotation2000.xlsx`.

    Se lee el XML a mano (con `zipfile`): el archivo trae miles de fotos incrustadas y `openpyxl` no lo abre.
    Si el encabezado no es el esperado, falla.
    """
    with zipfile.ZipFile(xlsx) as archive:
        shared = [
            re.sub(r"<[^>]+>", "", item)
            for item in re.findall(
                r"<si>(.*?)</si>", archive.read("xl/sharedStrings.xml").decode("utf-8"), re.S
            )
        ]
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows: list[dict[str, str]] = []
    for body in re.findall(r"<row [^>]*>(.*?)</row>", sheet, re.S):
        cells: dict[str, str] = {}
        for column, attributes, content in re.findall(
            r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', body, re.S
        ):
            value = re.search(r"<v>(.*?)</v>", content or "")
            if value:
                cells[column] = shared[int(value.group(1))] if 't="s"' in attributes else value.group(1)
        rows.append(cells)
    if not rows or any(rows[0].get(column) != title for column, title in HEADER.items()):
        raise FetchError("la primera hoja de annotation2000.xlsx no tiene el encabezado esperado")
    out: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        name = row.get("B", "")
        if CAR_NAME.match(name) and row.get("E") in LIGHTING:
            out[name] = {
                "marca": row.get("C", ""),
                "modelo": row.get("D", ""),
                "luz": row["E"],
                "tipo": row.get("F", "").strip().title(),
                "color": row.get("G", ""),
            }
    return out


def load_annotations(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def command_annotations(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    xlsx = root / "annotation2000.xlsx"
    if not xlsx.is_file():
        request = Request(DRIVE_URL.format(id=ANNOTATIONS_ID))  # noqa: S310 - https fijo
        with urlopen(request, timeout=TIMEOUT_S) as response, xlsx.open("wb") as target:  # noqa: S310
            shutil.copyfileobj(response, target)
    annotations = parse_annotations(xlsx)
    (root / "annotations.json").write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")
    counts = {light: sum(a["luz"] == light for a in annotations.values()) for light in LIGHTING}
    print(f"{len(annotations)} autos anotados: {counts}")
    return 0


def make_source(args: argparse.Namespace) -> ZipSource:
    """La fuente de los zips: el folder público o la carpeta de Drive propia (credenciales por entorno)."""
    if args.source == "public":
        return PublicSource()
    folder = args.drive_folder or os.environ.get(FOLDER_ENV)
    key = os.environ.get(KEY_ENV)
    if not folder or not key:
        raise FetchError(f"con --source drive indicá --drive-folder (o {FOLDER_ENV}) y definí {KEY_ENV}")
    return DriveSource(folder, ServiceAccountTokens(Path(key)))


def command_index(args: argparse.Namespace) -> int:
    source = make_source(args)
    index: dict[str, Any] = {}
    for label in source.labels():
        cars = index_zip(label, source)
        print(f"{label}: {len(cars)} autos, {sum(len(c['fotos']) for c in cars.values())} fotos")
        index.update(cars)
    Path(args.root).mkdir(parents=True, exist_ok=True)
    (Path(args.root) / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(index)} autos en total")
    return 0


def command_select(args: argparse.Namespace) -> int:
    root = Path(args.root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    annotations = load_annotations(root / "annotations.json")
    available = [car for car in index if car in annotations and len(index[car]["fotos"]) >= args.frames]
    lighting = {car: annotations[car]["luz"] for car in available}
    cars = select_cars(lighting, available, args.standard, args.seed)
    selection = {
        car: {
            "zip": index[car]["zip"],
            "id": index[car]["id"],
            "fotos": spread(list(map(int, index[car]["fotos"])), args.frames),
        }
        for car in cars
    }
    target = root / "selection.json"
    if target.is_file():  # sumar autos no puede cambiar las vistas (ni sacar autos) de los ya elegidos
        previous = json.loads(target.read_text(encoding="utf-8"))
        changed = [
            car
            for car, meta in previous.items()
            if car in selection and selection[car]["fotos"] != meta["fotos"]
        ]
        if changed:
            print(
                f"{len(changed)} autos cambiarían de vistas (¿otro --frames?); no se escribió nada.",
                file=sys.stderr,
            )
            return 1
        for car, meta in previous.items():
            selection.setdefault(car, meta)
    target.write_text(json.dumps(selection), encoding="utf-8")
    print(f"{len(selection)} autos de {len(available)} con anotación y fotos; {args.frames} vistas cada uno")
    return 0


def is_complete(path: Path) -> bool:
    """El archivo ya bajado está entero: un JPEG termina en `FFD9` y la pose es un JSON válido."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if path.suffix == ".jpg":
        return data.endswith(JPEG_END)
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def pull_frame(root: Path, car: str, number: int, entries: Mapping[str, Sequence[Any]], read: Reader) -> int:
    """Baja la pose y la foto de una vista (las que ya están enteras se saltean); devuelve cuántas bajó."""
    folder = root / "raw" / car
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for ext in ("json", "jpg"):
        target = folder / f"frame_{number:05d}.{ext}"
        if is_complete(target):
            continue
        offset, size, kind, crc, name = entries[ext]
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(read_entry(read, offset, size, kind, crc, name))
        partial.replace(target)  # nunca queda un archivo a medias con el nombre final
        done += 1
    return done


def car_complete(root: Path, car: str, frames: Sequence[int]) -> bool:
    return all(
        is_complete(root / "raw" / car / f"frame_{number:05d}.{ext}")
        for number in frames
        for ext in ("jpg", "json")
    )


def check_zip(
    label: str, source: ZipSource, index: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[str]:
    """Qué no coincide entre el zip de la fuente y `index.json` (vacío si está bien).

    Un zip cortado no abre (su directorio central está al final). Después se compara la cantidad de autos, la
    cantidad de vistas de cada uno y la posición y el CRC de las vistas elegidas.
    """
    try:
        cars = index_zip(label, source)
    except QuotaExceeded:
        raise  # no es una copia dañada: es la cuota de Google, y lo maneja quien llama
    except (zipfile.BadZipFile, FetchError, OSError, ValueError, KeyError) as exc:
        return [f"no se puede abrir {label}.zip ({type(exc).__name__}): ¿terminó de copiarse?"]
    expected = {car: meta for car, meta in index.items() if meta["zip"] == label}
    problems: list[str] = []
    if len(cars) != len(expected):
        problems.append(f"{len(cars)} autos en la copia y {len(expected)} en el índice")
    problems += [f"falta el auto {car}" for car in sorted(set(expected) - set(cars))]
    problems += [f"sobra el auto {car}" for car in sorted(set(cars) - set(expected))]
    for car, meta in expected.items():
        mine = cars.get(car)
        if mine is None:
            continue
        if len(mine["fotos"]) != len(meta["fotos"]):
            problems.append(
                f"{car}: {len(mine['fotos'])} vistas en la copia y {len(meta['fotos'])} en el índice"
            )
        for number in selection.get(car, {}).get("fotos", []):
            if mine["fotos"].get(number) != meta["fotos"].get(str(number)):
                problems.append(f"{car}: la vista {number} no coincide con el índice")
    return problems


def report_problems(label: str, problems: Sequence[str]) -> None:
    print(f"{label}.zip: la copia no coincide con el índice ({len(problems)} problemas)", file=sys.stderr)
    for problem in problems[:20]:
        print(f"  {problem}", file=sys.stderr)


def load_state(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    return index, selection


def command_drive_check(args: argparse.Namespace) -> int:
    """Valida el acceso: lista la carpeta y lee los últimos bytes de cada zip (donde está su índice)."""
    source = make_source(args)
    if not isinstance(source, DriveSource):
        print("drive-check es solo para --source drive", file=sys.stderr)
        return 1
    zips = source.zips()
    for label in source.labels():
        found = zips[label]
        tail = source.reader(label)(max(0, found.size - 1024), found.size - 1)
        state = "ok" if b"PK\x05\x06" in tail else "SIN FIN DE ZIP"
        print(f"{label:>10}  {found.size / 1e9:7.2f} GB  md5 {found.md5 or '-'}  {state}")
    missing = [label for label in ZIPS if label not in zips]
    print(
        f"{len(zips)} de {len(ZIPS)} zips en la carpeta"
        + (f"; faltan: {', '.join(missing)}" if missing else "")
    )
    print(f"{source.bytes_read / 1e6:.2f} MB leídos en {source.requests} pedidos")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    root = Path(args.root)
    source = make_source(args)
    index, selection = load_state(root)
    bad = 0
    for label in args.only or source.labels():
        problems = check_zip(label, source, index, selection)
        if problems:
            report_problems(label, problems)
            bad += 1
        else:
            cars = sum(meta["zip"] == label for meta in index.values())
            print(f"{label}.zip: {cars} autos y las vistas elegidas coinciden con el índice")
    print(f"{source.bytes_read / 1e9:.3f} GB leídos")
    return 1 if bad else 0


def command_pull(args: argparse.Namespace) -> int:
    root = Path(args.root)
    source = make_source(args)
    index, selection = load_state(root)
    available = source.labels()
    unknown = [label for label in args.only or [] if label not in available]
    if unknown:
        print(f"No están en la fuente: {', '.join(unknown)}", file=sys.stderr)
        return 1
    failed = 0
    for label in args.only or available:
        before = source.bytes_read
        problems = check_zip(label, source, index, selection)
        if problems:
            report_problems(label, problems)
            failed += 1
            continue
        read = source.reader(label)
        cars = [car for car, meta in selection.items() if meta["zip"] == label]
        jobs = [
            (car, number, index[car]["fotos"][str(number)])
            for car in cars
            for number in selection[car]["fotos"]
        ]
        if args.reverse:  # para sumar un segundo proceso que arranque por el final
            jobs.reverse()
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(pull_frame, root, *job, read) for job in jobs]
            for count, future in enumerate(futures, 1):
                try:
                    future.result()
                except QuotaExceeded as exc:
                    pool.shutdown(wait=False, cancel_futures=True)
                    print(f"{exc}. Lo ya bajado se conserva: se retoma después.", file=sys.stderr)
                    return 3
                except FetchError as exc:
                    print(f"  {exc}", file=sys.stderr)
                if count % 500 == 0:
                    print(f"{label}: {count}/{len(jobs)} vistas")
        done = sum(car_complete(root, car, selection[car]["fotos"]) for car in cars)
        gigabytes = (source.bytes_read - before) / 1e9
        print(f"{label}.zip: {done}/{len(cars)} autos completos; {gigabytes:.2f} GB leídos")
        if done == len(cars):
            print(f"  {label}.zip terminado y verificado: ya podés borrar su copia de Drive")
        else:
            failed += 1
    return 1 if failed else 0


def command_status(args: argparse.Namespace) -> int:
    root = Path(args.root)
    _, selection = load_state(root)
    annotations = load_annotations(root / "annotations.json")
    per_zip: dict[str, list[int]] = {}
    per_light: dict[str, list[int]] = {light: [0, 0] for light in LIGHTING}
    for car, meta in selection.items():
        done = car_complete(root, car, meta["fotos"])
        for counts in (per_zip.setdefault(meta["zip"], [0, 0]), per_light[annotations[car]["luz"]]):
            counts[0] += done
            counts[1] += 1
    for label in sorted(per_zip, key=lambda name: int(name.split("-")[0])):
        done, total = per_zip[label]
        print(f"{label:>10}  {done:>4}/{total:<4} autos completos" + ("  <- listo" if done == total else ""))
    for light, (done, total) in per_light.items():
        print(f"{light:>10}  {done:>4}/{total:<4} autos completos")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("annotations", "index", "select", "drive-check", "verify", "pull", "status")
    )
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--frames", type=int, default=12, help="vistas por auto")
    parser.add_argument(
        "--standard", type=int, default=600, help="autos de luz estándar (todos los demás entran)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reverse", action="store_true", help="pull: empezar por la última vista")
    parser.add_argument(
        "--source",
        choices=("public", "drive"),
        default="public",
        help="folder público o carpeta de Drive propia",
    )
    parser.add_argument("--drive-folder", default=None, help=f"id de la carpeta de Drive ({FOLDER_ENV})")
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="ZIP", help="solo estos zips (p. ej. 1400-1600)"
    )
    args = parser.parse_args(argv)
    if args.root is None:
        parser.error("indicá --root o definí REID_DATA_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(args.root).resolve().is_relative_to(REPO_ROOT):
        print("Los datos no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    commands = {
        "annotations": command_annotations,
        "index": command_index,
        "select": command_select,
        "drive-check": command_drive_check,
        "verify": command_verify,
        "pull": command_pull,
        "status": command_status,
    }
    try:
        return commands[args.command](args)
    except QuotaExceeded as exc:
        print(f"{exc}. Lo ya bajado se conserva: se retoma después.", file=sys.stderr)
        return 3
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
