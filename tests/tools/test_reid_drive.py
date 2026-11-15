"""tools/training/reid/drive.py y la fuente `drive` de fetch.py, contra un Drive simulado (sin red real)."""

import argparse
import io
import json
import random
import sys
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

FOLDER_REID = Path(__file__).resolve().parents[2] / "tools" / "training" / "reid"
sys.path.insert(0, str(FOLDER_REID))  # los módulos se importan por nombre, como cuando corren como script
import drive  # noqa: E402
import fetch  # noqa: E402

LABEL = "1400-1600"
FOLDER = "carpeta-de-prueba-123"
CARS = [f"2024_05_0{i}_10_00_00" for i in range(1, 5)]
VIEWS = 30
TOKEN = "token-1"


def build_zip(*, label: str = LABEL, cars: list[str] = CARS) -> bytes:
    """Un zip como el de 3DRealCar: por auto, `frame_N.jpg|json` y mucho ruido que no se necesita."""
    rng = random.Random(0)  # noqa: S311 - datos de prueba
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for car in cars:
            archive.writestr(f"{label}/{car}/", b"")
            for n in range(1, VIEWS + 1):
                jpg = zipfile.ZipInfo(f"{label}/{car}/frame_{n:05d}.jpg")
                archive.writestr(jpg, rng.randbytes(20_000) + b"\xff\xd9", zipfile.ZIP_STORED)
                pose = json.dumps({"cameraPoseARFrame": [float(n)] * 16, "relleno": "x" * 300})
                archive.writestr(f"{label}/{car}/frame_{n:05d}.json", pose, zipfile.ZIP_DEFLATED)
                archive.writestr(f"{label}/{car}/depth_{n:05d}.png", rng.randbytes(30_000))
                archive.writestr(f"{label}/{car}/conf_{n:05d}.png", rng.randbytes(30_000))
    return out.getvalue()


class FakeDrive:
    """Servidor local con la parte de la API de Drive v3 que usa `DriveSource` (lista y rangos)."""

    def __init__(self, files: dict[str, tuple[str, bytes]]) -> None:
        self.files = files  # id -> (nombre, contenido)
        self.token = TOKEN
        self.script: list[tuple[int, str]] = []  # (HTTP, razón) que se sirven antes de responder bien
        self.page_size = 1000
        self.media_requests = 0
        self.served = 0
        self.ignore_range = False
        self.quota_from: int | None = None  # a partir de este pedido, Google corta por cuota
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def reply(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    for key, value in (headers or {}).items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:  # el cliente cortó sin leer (el caso del 200 sin rango)
                    pass

            def error(self, status: int, reason: str) -> None:
                detail = {"error": {"code": status, "errors": [{"reason": reason}]}}
                self.reply(status, json.dumps(detail).encode())

            def do_GET(self) -> None:  # noqa: N802
                if self.headers.get("Authorization") != f"Bearer {outer.token}":
                    return self.error(401, "authError")
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path == "/drive/v3/files":
                    return self.listing(query)
                file_id = url.path.rsplit("/", 1)[1]
                if file_id not in outer.files:
                    return self.error(404, "notFound")
                outer.media_requests += 1
                if outer.quota_from is not None and outer.media_requests > outer.quota_from:
                    return self.error(403, "downloadQuotaExceeded")
                if outer.script:
                    return self.error(*outer.script.pop(0))
                data = outer.files[file_id][1]
                header = self.headers.get("Range")
                if header is None or outer.ignore_range:
                    outer.served += len(data)
                    return self.reply(200, data)
                start, end = header.removeprefix("bytes=").split("-")
                first, last = int(start), min(int(end), len(data) - 1)
                outer.served += last - first + 1
                self.reply(
                    206, data[first : last + 1], {"Content-Range": f"bytes {first}-{last}/{len(data)}"}
                )

            def listing(self, query: dict[str, list[str]]) -> None:
                if f"'{FOLDER}' in parents" not in query["q"][0]:
                    return self.reply(200, json.dumps({"files": []}).encode())
                items = [
                    {"id": i, "name": n, "size": str(len(d)), "md5Checksum": "m" * 32}
                    for i, (n, d) in sorted(outer.files.items())
                ]
                start = int(query.get("pageToken", ["0"])[0])
                page = items[start : start + outer.page_size]
                body: dict[str, object] = {"files": page}
                if start + outer.page_size < len(items):
                    body["nextPageToken"] = str(start + outer.page_size)
                self.reply(200, json.dumps(body).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/drive/v3"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class FakeTokens:
    def __init__(self, server: FakeDrive, current: str = TOKEN) -> None:
        self.server = server
        self.current = current
        self.refreshes = 0

    def get(self, *, refresh: bool = False) -> str:
        if refresh:
            self.refreshes += 1
            self.current = self.server.token
        return self.current


@pytest.fixture
def zip_bytes() -> bytes:
    return build_zip()


@pytest.fixture
def server(zip_bytes: bytes) -> Iterator[FakeDrive]:
    fake = FakeDrive(
        {
            "id-zip": (f"Copia de {LABEL}.zip", zip_bytes),
            "id-hq": ("HQ200.zip", b"no es de autos"),
            "id-txt": ("notas.txt", b"hola"),
        }
    )
    yield fake
    fake.stop()


def source_for(server: FakeDrive, tokens: FakeTokens | None = None) -> Any:
    sleeps: list[float] = []
    source = drive.DriveSource(FOLDER, tokens or FakeTokens(server), api=server.api, sleep=sleeps.append)
    source.sleeps = sleeps
    return source


def make_state(root: Path, source: Any) -> None:
    """`index.json` (de la fuente, como lo haría `index`) y `selection.json` (12 vistas por auto)."""
    cars = fetch.index_zip(LABEL, source)
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(cars))
    selection = {
        car: {"zip": LABEL, "id": "", "fotos": fetch.spread(list(meta["fotos"]), 12)}
        for car, meta in cars.items()
    }
    (root / "selection.json").write_text(json.dumps(selection))


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, server: FakeDrive) -> Any:
    """`fetch.make_source` apunta al Drive simulado."""
    sources: list[Any] = []

    def make(_: argparse.Namespace) -> Any:
        source = source_for(server)
        sources.append(source)
        return source

    monkeypatch.setattr(fetch, "make_source", make)
    return sources  # type: ignore[return-value]


def run(command: str, root: Path, *extra: str) -> int:
    return int(fetch.main([command, "--root", str(root), "--source", "drive", *extra]))


# --- carpeta ------------------------------------------------------------------------------------------


def test_zips_encuentra_la_copia_aunque_se_llame_copia_de(server: FakeDrive) -> None:
    server.page_size = 1  # fuerza varias páginas

    found = source_for(server).zips()

    assert list(found) == [LABEL]
    assert found[LABEL].file_id == "id-zip" and found[LABEL].name == f"Copia de {LABEL}.zip"
    assert found[LABEL].size == len(server.files["id-zip"][1])


def test_dos_copias_del_mismo_zip_es_un_error(server: FakeDrive) -> None:
    server.files["id-otra"] = (f"{LABEL}.zip", b"x")

    with pytest.raises(drive.FetchError, match="dos archivos"):
        source_for(server).zips()


def test_el_id_de_carpeta_se_valida() -> None:
    with pytest.raises(drive.FetchError, match="carpeta"):
        drive.DriveSource("x' or '1'='1", FakeTokens(None))  # type: ignore[arg-type]


# --- lectura parcial -----------------------------------------------------------------------------------


def test_index_zip_lee_solo_el_indice_y_coincide_con_el_zip_local(
    server: FakeDrive, zip_bytes: bytes
) -> None:
    source = source_for(server)

    cars = fetch.index_zip(LABEL, source)

    assert sorted(cars) == sorted(CARS)
    assert all(len(meta["fotos"]) == VIEWS for meta in cars.values())
    local = {i.filename: (i.header_offset, i.CRC) for i in zipfile.ZipFile(io.BytesIO(zip_bytes)).infolist()}
    first = cars[CARS[0]]["fotos"][1]["jpg"]
    assert (first[0], first[3]) == local[first[4]]
    assert source.bytes_read < 0.1 * len(zip_bytes)  # el índice es una fracción del zip
    assert 0 <= source.bytes_read - server.served < 1024  # lo demás es el JSON de la lista de la carpeta


def test_pull_baja_solo_las_vistas_elegidas_con_pocos_bytes_y_es_idempotente(
    tmp_path: Path, server: FakeDrive, zip_bytes: bytes, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "datos"
    make_state(root, source_for(server))
    selection = json.loads((root / "selection.json").read_text())
    local = zipfile.ZipFile(io.BytesIO(zip_bytes))

    assert run("pull", root, "--only", LABEL) == 0

    out = capsys.readouterr().out
    assert f"{len(CARS)}/{len(CARS)} autos completos" in out and "ya podés borrar" in out
    for car, meta in selection.items():
        for n in meta["fotos"]:
            for ext in ("jpg", "json"):
                name = f"frame_{n:05d}.{ext}"
                assert (root / "raw" / car / name).read_bytes() == local.read(f"{LABEL}/{car}/{name}")
        assert len(list((root / "raw" / car).glob("*.jpg"))) == 12  # solo las elegidas, no las 30
    assert not list(root.rglob("*.part"))
    pulled = wired[-1].bytes_read
    ratio = pulled / len(zip_bytes)
    print(f"\nzip sintético {len(zip_bytes) / 1e6:.1f} MB; leídos {pulled / 1e6:.2f} MB ({ratio:.1%})")
    assert pulled < 0.25 * len(zip_bytes)

    before = {p: p.stat().st_mtime_ns for p in (root / "raw").rglob("frame_*")}
    assert run("pull", root, "--only", LABEL) == 0  # segunda corrida: no vuelve a bajar nada
    assert before == {p: p.stat().st_mtime_ns for p in (root / "raw").rglob("frame_*")}
    assert wired[-1].bytes_read < 0.1 * len(zip_bytes)  # solo el índice


def test_pull_de_un_zip_que_no_esta_en_la_fuente_falla(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))

    assert run("pull", tmp_path, "--only", "200-400") == 1
    assert "200-400" in capsys.readouterr().err


# --- verificación --------------------------------------------------------------------------------------


def test_verify_acepta_la_copia_buena(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))

    assert run("verify", tmp_path) == 0
    assert f"{LABEL}.zip: {len(CARS)} autos" in capsys.readouterr().out


def test_verify_detecta_un_auto_de_menos_y_pull_no_baja_nada(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    server.files["id-zip"] = (f"{LABEL}.zip", build_zip(cars=CARS[:3]))  # la copia quedó con un auto menos

    assert run("verify", tmp_path) == 1
    assert run("pull", tmp_path, "--only", LABEL) == 1

    err = capsys.readouterr().err
    assert "3 autos en la copia y 4 en el índice" in err and f"falta el auto {CARS[3]}" in err
    assert not (tmp_path / "raw").exists()


def test_verify_acepta_autos_repetidos_en_otro_zip_del_indice(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """3DRealCar repite autos entre zips (17 de 200-400 están en 1200-1400): el índice los asigna a uno."""
    make_state(tmp_path, source_for(server))
    index = json.loads((tmp_path / "index.json").read_text())
    index[CARS[3]]["zip"] = "1200-1400"  # el índice lo asignó al otro zip; la copia de este zip lo trae igual
    (tmp_path / "index.json").write_text(json.dumps(index))

    assert run("verify", tmp_path) == 0
    assert "sobra" not in capsys.readouterr().err


def test_verify_sigue_rechazando_un_auto_de_mas_que_no_esta_en_el_indice(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    extra = "2024_05_09_10_00_00"
    server.files["id-zip"] = (f"{LABEL}.zip", build_zip(cars=[*CARS, extra]))

    assert run("verify", tmp_path) == 1
    assert f"sobra el auto {extra}" in capsys.readouterr().err


def test_verify_rechaza_un_repetido_con_otras_vistas(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    index = json.loads((tmp_path / "index.json").read_text())
    index[CARS[3]]["zip"] = "1200-1400"
    index[CARS[3]]["fotos"].popitem()  # en el otro zip tiene una vista menos: no es el mismo auto
    (tmp_path / "index.json").write_text(json.dumps(index))

    assert run("verify", tmp_path) == 1
    assert f"sobra el auto {CARS[3]}" in capsys.readouterr().err


def test_verify_detecta_una_vista_que_cambio_de_crc(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    index = json.loads((tmp_path / "index.json").read_text())
    selected = json.loads((tmp_path / "selection.json").read_text())[CARS[0]]["fotos"][0]
    index[CARS[0]]["fotos"][str(selected)]["jpg"][3] ^= 1
    (tmp_path / "index.json").write_text(json.dumps(index))

    assert run("verify", tmp_path) == 1
    assert f"{CARS[0]}: la vista {selected} no coincide" in capsys.readouterr().err


def test_un_zip_cortado_no_abre(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    data = server.files["id-zip"][1]
    server.files["id-zip"] = (f"{LABEL}.zip", data[: len(data) // 2])  # copia interrumpida

    assert run("verify", tmp_path) == 1
    assert "terminó de copiarse" in capsys.readouterr().err


def test_un_byte_dañado_en_una_vista_elegida_falla_por_crc_y_las_demas_se_bajan(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    selection = json.loads((tmp_path / "selection.json").read_text())
    index = json.loads((tmp_path / "index.json").read_text())
    car = CARS[0]
    number = selection[car]["fotos"][0]
    offset, _, _, _, name = index[car]["fotos"][str(number)]["jpg"]
    data = bytearray(server.files["id-zip"][1])
    data[offset + 30 + len(name) + 100] ^= 0xFF  # un byte adentro de los datos del JPEG (zip sin compresión)
    server.files["id-zip"] = (f"{LABEL}.zip", bytes(data))

    assert run("pull", tmp_path, "--only", LABEL) == 1

    assert "CRC distinto" in capsys.readouterr().err
    assert not (tmp_path / "raw" / car / f"frame_{number:05d}.jpg").exists()
    assert not list(tmp_path.rglob("*.part"))
    assert (tmp_path / "raw" / CARS[1] / f"frame_{selection[CARS[1]]['fotos'][0]:05d}.jpg").is_file()


# --- errores de Drive ---------------------------------------------------------------------------------


def test_un_token_vencido_se_renueva_una_vez(server: FakeDrive) -> None:
    tokens = FakeTokens(server, current="viejo")

    found = source_for(server, tokens).zips()

    assert LABEL in found and tokens.refreshes == 1


def test_el_limite_de_velocidad_se_reintenta_con_espera(server: FakeDrive) -> None:
    source = source_for(server)
    server.script = [(403, "userRateLimitExceeded"), (429, "rateLimitExceeded")]

    data = source.reader(LABEL)(0, 3)

    assert data[:2] == b"PK" and server.media_requests == 3
    assert source.sleeps == [2.0, 4.0]


def test_la_cuota_de_descarga_corta_sin_reintentar(server: FakeDrive) -> None:
    source = source_for(server)
    server.script = [(403, "downloadQuotaExceeded")] * 10

    with pytest.raises(drive.QuotaExceeded, match="cuota"):
        source.reader(LABEL)(0, 3)

    assert server.media_requests == 1 and source.sleeps == []


def test_sin_permiso_el_error_dice_que_hay_que_compartir_la_carpeta(server: FakeDrive) -> None:
    source = source_for(server)
    server.script = [(403, "forbidden")]

    with pytest.raises(drive.FetchError, match="compartí la carpeta"):
        source.reader(LABEL)(0, 3)

    assert server.media_requests == 1


def test_un_error_de_servidor_se_reintenta_y_despues_se_rinde(server: FakeDrive) -> None:
    source = source_for(server)
    server.script = [(503, "backendError")] * 20

    with pytest.raises(drive.FetchError, match="6 intentos"):
        source.reader(LABEL)(0, 3)

    assert server.media_requests == drive.RETRIES and len(source.sleeps) == drive.RETRIES


def test_una_respuesta_sin_rango_no_se_lee_entera(server: FakeDrive) -> None:
    source = source_for(server)
    read = source.reader(LABEL)  # lista la carpeta una vez
    before = source.bytes_read
    server.ignore_range = True

    with pytest.raises(drive.FetchError, match="por rango"):
        read(0, 3)

    assert source.bytes_read == before  # el cuerpo (el zip entero) no se leyó


def test_la_cuota_en_medio_del_pull_corta_con_codigo_3_y_conserva_lo_bajado(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    before = server.media_requests
    make_state(tmp_path, source_for(server))
    index_requests = server.media_requests - before  # lo que cuesta leer el índice del zip
    server.quota_from = server.media_requests + index_requests + 3  # la cuota llega con la 2.ª foto

    assert run("pull", tmp_path, "--only", LABEL, "--threads", "1") == 3

    assert "cuota" in capsys.readouterr().err
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2  # lo bajado antes del corte se conserva
    assert not list(tmp_path.rglob("*.part"))


def test_la_cuota_al_leer_el_indice_no_se_confunde_con_una_copia_rota(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    server.quota_from = server.media_requests

    assert run("verify", tmp_path) == 3

    err = capsys.readouterr().err
    assert "cuota" in err and "terminó de copiarse" not in err


# --- credenciales y configuración ----------------------------------------------------------------------


def test_la_clave_no_puede_estar_dentro_del_repositorio() -> None:
    with pytest.raises(drive.FetchError, match="repositorio"):
        drive.ServiceAccountTokens(drive.REPO_ROOT / "pyproject.toml")


def test_la_clave_tiene_que_existir(tmp_path: Path) -> None:
    with pytest.raises(drive.FetchError, match="no existe"):
        drive.ServiceAccountTokens(tmp_path / "no-esta.json")


def test_make_source_drive_exige_carpeta_y_clave(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(drive.FOLDER_ENV, raising=False)
    monkeypatch.delenv(drive.KEY_ENV, raising=False)
    args = argparse.Namespace(source="drive", drive_folder=None)

    with pytest.raises(drive.FetchError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        fetch.make_source(args)
    assert isinstance(fetch.make_source(argparse.Namespace(source="public")), fetch.PublicSource)


def test_main_convierte_el_error_de_configuracion_en_un_mensaje(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(drive.FOLDER_ENV, raising=False)
    monkeypatch.delenv(drive.KEY_ENV, raising=False)

    assert run("drive-check", tmp_path) == 1
    assert "--source drive" in capsys.readouterr().err


# --- comandos auxiliares ---------------------------------------------------------------------------------


def test_drive_check_lista_los_zips_y_dice_cuales_faltan(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("drive-check", tmp_path) == 0

    out = capsys.readouterr().out
    assert LABEL in out and "ok" in out and "1 de 10 zips" in out and "faltan: 0-200" in out
    assert wired[-1].bytes_read < 4096  # la lista de la carpeta y los últimos 1 KiB de cada zip


def test_status_cuenta_autos_completos_por_zip_y_por_luz(
    tmp_path: Path, server: FakeDrive, wired: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    make_state(tmp_path, source_for(server))
    annotations = {car: {"luz": "dark" if i % 2 else "standard"} for i, car in enumerate(CARS)}
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    assert run("status", tmp_path) == 0
    assert f"{LABEL}     0/4" in capsys.readouterr().out

    run("pull", tmp_path, "--only", LABEL)
    capsys.readouterr()
    assert run("status", tmp_path) == 0

    out = capsys.readouterr().out
    assert "4/4" in out and "<- listo" in out and "standard     2/2" in out and "reflective     0/0" in out


def test_select_no_cambia_las_vistas_de_los_autos_ya_elegidos(
    tmp_path: Path, server: FakeDrive, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path
    cars = fetch.index_zip(LABEL, source_for(server))
    (root / "index.json").write_text(json.dumps(cars))
    (root / "annotations.json").write_text(json.dumps({c: {"luz": "standard"} for c in CARS}))
    assert fetch.main(["select", "--root", str(root), "--standard", "2"]) == 0
    first = json.loads((root / "selection.json").read_text())

    assert fetch.main(["select", "--root", str(root), "--standard", "100"]) == 0  # suma autos
    second = json.loads((root / "selection.json").read_text())
    assert len(first) == 2 and len(second) == 4
    assert all(second[car] == meta for car, meta in first.items())

    assert fetch.main(["select", "--root", str(root), "--standard", "1"]) == 0  # pedir menos no saca autos
    assert set(json.loads((root / "selection.json").read_text())) == set(second)

    assert fetch.main(["select", "--root", str(root), "--standard", "100", "--frames", "5"]) == 1
    assert "cambiarían de vistas" in capsys.readouterr().err
    assert json.loads((root / "selection.json").read_text()) == second
