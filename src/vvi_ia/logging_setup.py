"""Logs estructurados en JSON, una línea por registro (evita la inyección de líneas)."""

import json
import logging
import sys
from datetime import UTC, datetime

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Serializa cada registro como un objeto JSON; los campos de `extra` pasan tal cual."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                entry[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            # Solo el tipo de la excepción: el mensaje y el traceback pueden arrastrar contenido.
            entry["exc_type"] = record.exc_info[0].__name__
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Reemplaza los handlers del logger raíz por uno JSON hacia stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
