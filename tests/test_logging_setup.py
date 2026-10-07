import json
import logging

from vvi_ia.logging_setup import JsonFormatter


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("vvi", logging.INFO, __file__, 1, msg, None, None)
    record.__dict__.update(extra)
    return record


def test_una_linea_json_con_extras() -> None:
    line = JsonFormatter().format(_record("hola", alert_id="a-1"))

    data = json.loads(line)
    assert data["msg"] == "hola"
    assert data["alert_id"] == "a-1"
    assert data["level"] == "INFO"


def test_los_saltos_de_linea_no_rompen_el_registro() -> None:
    line = JsonFormatter().format(_record("uno\ndos\r\ntres"))

    assert "\n" not in line
    assert json.loads(line)["msg"] == "uno\ndos\r\ntres"


def test_la_excepcion_registra_solo_el_tipo() -> None:
    try:
        raise ValueError("contenido sensible")
    except ValueError:
        import sys

        record = _record("falló")
        record.exc_info = sys.exc_info()

    line = JsonFormatter().format(record)

    assert json.loads(line)["exc_type"] == "ValueError"
    assert "contenido sensible" not in line
