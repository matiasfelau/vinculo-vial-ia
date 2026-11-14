"""Lectura por rangos de los zips de 3DRealCar desde una carpeta de Google Drive propia, solo lectura.

La copia de cada zip vive en el Drive de quien entrena (con "Crear una copia"): ahí la cuota de descarga es
la de esa cuenta y no la del folder público. Se accede con una cuenta de servicio a la que se le comparte
**esa carpeta** como lector: solo ve lo que se le comparta y no puede escribir ni borrar. La clave JSON se
queda fuera del repositorio (`GOOGLE_APPLICATION_CREDENTIALS`) y nunca se loguea.

Cada lectura es un pedido `Range` sobre `files/{id}?alt=media`: nunca se baja un zip entero.
"""

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
API = "https://www.googleapis.com/drive/v3"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
KEY_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
FOLDER_ENV = "REID_DRIVE_FOLDER_ID"
TIMEOUT_S = 120
RETRIES = 6
BACKOFF_S = 2.0  # espera creciente entre reintentos
ZIP_NAME = re.compile(r"(\d+-\d+)\.zip$", re.IGNORECASE)  # tolera "Copia de 1400-1600.zip"
FOLDER_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
# Razones de 403 que se resuelven esperando unos segundos (se reintenta) y las que no (se frena)
RATE_LIMITED = frozenset({"userRateLimitExceeded", "rateLimitExceeded"})
QUOTA = frozenset({"downloadQuotaExceeded", "dailyLimitExceeded", "quotaExceeded", "storageQuotaExceeded"})

Reader = Callable[[int, int], bytes]  # (desde, hasta inclusive) -> bytes


class FetchError(RuntimeError):
    """No se pudo leer algo del zip remoto."""


class QuotaExceeded(FetchError):
    """Google cortó por cuota: insistir no sirve. Hay que esperar a que se libere."""


class Tokens(Protocol):
    def get(self, *, refresh: bool = False) -> str: ...


class ServiceAccountTokens:
    """Token de acceso de una cuenta de servicio (`google-auth`, extra `tools`)."""

    def __init__(self, key_file: Path) -> None:
        if key_file.resolve().is_relative_to(REPO_ROOT):
            raise FetchError("la clave de la cuenta de servicio no puede estar dentro del repositorio")
        if not key_file.is_file():
            raise FetchError(f"no existe la clave indicada en {KEY_ENV}")
        try:
            from google.auth.transport.requests import Request as AuthRequest
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise FetchError("falta `google-auth` (pip install -e '.[tools]')") from exc
        self._credentials = service_account.Credentials.from_service_account_file(
            str(key_file), scopes=[SCOPE]
        )
        self._request = AuthRequest()
        self._lock = threading.Lock()

    def get(self, *, refresh: bool = False) -> str:
        with self._lock:
            if refresh or not self._credentials.valid:
                self._credentials.refresh(self._request)
            return str(self._credentials.token)


@dataclass(frozen=True)
class DriveZip:
    label: str
    file_id: str
    name: str
    size: int
    md5: str


class DriveSource:
    """Los zips de una carpeta de Drive, leídos por rangos con una cuenta de solo lectura."""

    def __init__(
        self,
        folder_id: str,
        tokens: Tokens,
        *,
        api: str = API,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not FOLDER_ID.match(folder_id):
            raise FetchError("el id de la carpeta de Drive no tiene el formato esperado")
        self.folder_id = folder_id
        self.tokens = tokens
        self.api = api.rstrip("/")
        self._sleep = sleep
        self._lock = threading.Lock()
        self._zips: dict[str, DriveZip] | None = None
        self.bytes_read = 0
        self.requests = 0

    # --- HTTP -------------------------------------------------------------------------------------------

    def _get(self, url: str, headers: Mapping[str, str], *, ranged: bool) -> bytes:
        refreshed = False
        last = "sin respuesta"
        for attempt in range(RETRIES):
            token = self.tokens.get(refresh=False)
            request = Request(url, headers={**headers, "Authorization": f"Bearer {token}"})  # noqa: S310
            try:
                with urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
                    # Sin Range, Drive contesta 200 con el archivo entero (decenas de GB): no se lee el cuerpo
                    if ranged and response.status != 206:
                        raise FetchError(f"Drive no respondió por rango (HTTP {response.status})")
                    data = bytes(response.read())
                with self._lock:
                    self.requests += 1
                    self.bytes_read += len(data)
                return data
            except HTTPError as exc:
                reason = _reason(exc)
                if exc.code == 401 and not refreshed:
                    refreshed = True  # el token venció: se pide uno nuevo una vez
                    self.tokens.get(refresh=True)
                    continue
                if exc.code == 403 and reason in QUOTA:
                    raise QuotaExceeded(
                        f"Google cortó por cuota ({reason}): esperá a que se libere"
                    ) from None
                if exc.code in (403, 404) and reason not in RATE_LIMITED:
                    raise FetchError(
                        f"HTTP {exc.code} ({reason or 'sin detalle'}): la cuenta de servicio no ve "
                        "el archivo; compartí la carpeta con ella como lector"
                    ) from None
                last = f"HTTP {exc.code} {reason}".strip()
            except (URLError, TimeoutError, OSError) as exc:
                last = type(exc).__name__
            self._sleep(BACKOFF_S * 2**attempt)
        raise FetchError(f"falló el pedido a Drive tras {RETRIES} intentos ({last})")

    # --- carpeta ----------------------------------------------------------------------------------------

    def zips(self) -> dict[str, DriveZip]:
        """`etiqueta → zip` de los archivos `<a>-<b>.zip` de la carpeta (la etiqueta es la de `ZIPS`)."""
        if self._zips is not None:
            return self._zips
        found: dict[str, DriveZip] = {}
        page = ""
        while True:
            query = {
                "q": f"'{self.folder_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,size,md5Checksum)",
                "pageSize": "1000",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page:
                query["pageToken"] = page
            body = json.loads(
                self._get(f"{self.api}/files?{urlencode(query, quote_via=quote)}", {}, ranged=False)
            )
            for item in body.get("files", []):
                match = ZIP_NAME.search(item["name"])
                if not match:
                    continue
                label = match.group(1)
                if label in found:
                    raise FetchError(f"hay dos archivos para {label}.zip en la carpeta: dejá solo uno")
                found[label] = DriveZip(
                    label, item["id"], item["name"], int(item.get("size", 0)), item.get("md5Checksum", "")
                )
            page = body.get("nextPageToken", "")
            if not page:
                break
        self._zips = found
        return found

    def labels(self) -> list[str]:
        return sorted(self.zips(), key=lambda label: int(label.split("-")[0]))

    def size(self, label: str) -> int:
        return self.zips()[label].size

    def reader(self, label: str) -> Reader:
        file_id = self.zips()[label].file_id
        url = f"{self.api}/files/{quote(file_id)}?alt=media&supportsAllDrives=true"

        def read(start: int, end: int) -> bytes:
            return self._get(url, {"Range": f"bytes={start}-{end}"}, ranged=True)

        return read


def _reason(error: HTTPError) -> str:
    """La razón que da Drive en el cuerpo del error (`downloadQuotaExceeded`, `forbidden`...)."""
    try:
        detail = json.loads(error.read().decode("utf-8", "replace"))
        return str(detail["error"]["errors"][0]["reason"])
    except (ValueError, KeyError, IndexError, TypeError, OSError):
        return ""
