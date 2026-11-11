"""Baja de 3DRealCar solo las fotos RGB que hacen falta (sin depth, mallas ni mapas del escáner).

3DRealCar (Du et al., ICCV 2025; Apache-2.0) se distribuye en zips de 55-110 GB con todo el escaneo 3D. Acá
solo
se necesita un puñado de fotos por auto, así que el script lee el índice de cada zip por HTTP (Google Drive
acepta
pedidos por rango) y baja únicamente `frame_N.jpg` y `frame_N.json` (la pose de la cámara) de las vistas
elegidas.
Nada se guarda dentro del repositorio.

    annotations  baja `annotation2000.xlsx` (marca, modelo, luz, tipo y color de cada auto) y lo pasa a
    `annotations.json`
    index   lee el índice de los 11 zips y arma `index.json` (auto → zip, fotos y posiciones dentro del zip)
    select  elige los autos (todos los oscuros y reflectivos, y una muestra de los estándar) y `K` fotos de
    cada uno
    pull    baja esas fotos a `raw/<auto>/frame_N.jpg|json`

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
import time
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
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


class FetchError(RuntimeError):
    """No se pudo leer algo del zip remoto."""


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
    """Archivo de solo lectura sobre HTTP con pedidos por rango (alcanza para que `zipfile` lea el índice)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.size = total_size(url)
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
        data = http_range(self.url, self.pos, self.pos + count - 1)
        self.pos += len(data)
        return data


def read_entry(url: str, offset: int, compress_size: int, compress_type: int, crc: int, name: str) -> bytes:
    """Contenido de una entrada del zip con un solo pedido: encabezado local y datos juntos."""
    block = http_range(url, offset, offset + LOCAL_HEADER.size + len(name) + HEADER_SLACK + compress_size)
    signature, *_, name_len, extra_len = LOCAL_HEADER.unpack_from(block)
    if signature != LOCAL_SIGNATURE:
        raise FetchError(f"encabezado local inválido en {name}")
    start = LOCAL_HEADER.size + name_len + extra_len
    if start + compress_size > len(block):
        block = http_range(url, offset, offset + start + compress_size)
    raw = block[start : start + compress_size]
    if compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    elif compress_type != zipfile.ZIP_STORED:
        raise FetchError(f"compresión no soportada ({compress_type}) en {name}")
    if zlib.crc32(raw) != crc:
        raise FetchError(f"CRC distinto en {name}")
    return raw


def index_zip(zip_name: str, file_id: str) -> dict[str, Any]:
    """`auto → {zip, fotos: {n: {jpg: entrada, json: entrada}}}` de un zip, leyendo solo su índice."""
    url = DRIVE_URL.format(id=file_id)
    archive = zipfile.ZipFile(RangeFile(url))  # type: ignore[arg-type]
    cars: dict[str, dict[str, Any]] = {}
    for info in archive.infolist():
        parts = info.filename.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        match = FRAME.match(parts[1])
        if not match or not CAR_NAME.match(match["car"]):
            continue
        entry = [info.header_offset, info.compress_size, info.compress_type, info.CRC, info.filename]
        car = cars.setdefault(match["car"], {"zip": zip_name, "id": file_id, "fotos": {}})
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


def command_index(args: argparse.Namespace) -> int:
    index: dict[str, Any] = {}
    for zip_name, file_id in ZIPS.items():
        cars = index_zip(zip_name, file_id)
        print(f"{zip_name}: {len(cars)} autos, {sum(len(c['fotos']) for c in cars.values())} fotos")
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
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
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


def pull_frame(root: Path, car: str, number: int, entries: Mapping[str, Sequence[Any]], file_id: str) -> int:
    """Baja la pose y la foto de una vista (las que ya están enteras se saltean); devuelve cuántas bajó."""
    folder = root / "raw" / car
    folder.mkdir(parents=True, exist_ok=True)
    url = DRIVE_URL.format(id=file_id)
    done = 0
    for ext in ("json", "jpg"):
        target = folder / f"frame_{number:05d}.{ext}"
        if is_complete(target):
            continue
        offset, size, kind, crc, name = entries[ext]
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(read_entry(url, offset, size, kind, crc, name))
        partial.replace(target)  # nunca queda un archivo a medias con el nombre final
        done += 1
    return done


def command_pull(args: argparse.Namespace) -> int:
    root = Path(args.root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    jobs = [
        (car, number, index[car]["fotos"][str(number)], meta["id"])
        for car, meta in selection.items()
        for number in meta["fotos"]
    ]
    if args.reverse:  # para sumar un segundo proceso que arranque por el final
        jobs.reverse()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(pull_frame, root, *job) for job in jobs]
        for count, future in enumerate(futures, 1):
            try:
                future.result()
            except FetchError as exc:
                failed += 1
                print(f"  {exc}", file=sys.stderr)
            if count % 500 == 0:
                print(f"{count}/{len(jobs)} vistas")
    print(f"Listo: {len(jobs) - failed} vistas bajadas, {failed} con error")
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("annotations", "index", "select", "pull"))
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--frames", type=int, default=12, help="vistas por auto")
    parser.add_argument(
        "--standard", type=int, default=600, help="autos de luz estándar (todos los demás entran)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reverse", action="store_true", help="pull: empezar por la última vista")
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
        "pull": command_pull,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
