import base64
import hashlib
import uuid
from typing import Any

import pytest

from vvi_ia.domain.model import AlertType, EvidenceRole
from vvi_ia.service.payload import InvalidPayloadError, parse_evidence_uploaded, readable_alert_id

BUCKET = "bucket-de-prueba"


def _item(alert_id: str, role: str, index: int = 0) -> dict[str, Any]:
    return {
        "role": role,
        "index": index,
        "key": f"alerts/{alert_id}/original/{role.lower()}-{index}.bin",
        "sha256": hashlib.sha256(f"{role}{index}".encode()).hexdigest(),
    }


def _raw(alert_type: str = "CRASHED", roles: list[tuple[str, int]] | None = None) -> dict[str, Any]:
    alert_id = str(uuid.uuid4())
    if roles is None:
        roles = [("GENERAL", 0), ("PLATE", 0)]
        roles += [("DETAIL", 0)] if alert_type == "CRASHED" else [("CONTEXT", 0)]
    return {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "pat_enc_ia": base64.b64encode(b"x" * 48).decode(),
        "bucket": BUCKET,
        "evidence": [_item(alert_id, role, index) for role, index in roles],
    }


def _parse(raw: dict[str, Any]) -> None:
    parse_evidence_uploaded(raw, BUCKET)


def test_crashed_valido() -> None:
    payload = parse_evidence_uploaded(_raw("CRASHED"), BUCKET)

    assert payload.alert_type == AlertType.CRASHED
    assert [i.role for i in payload.evidence] == [
        EvidenceRole.GENERAL,
        EvidenceRole.PLATE,
        EvidenceRole.DETAIL,
    ]


def test_garage_valido() -> None:
    assert parse_evidence_uploaded(_raw("PARKED_ON_GARAGE"), BUCKET).alert_type == AlertType.PARKED_ON_GARAGE


def test_crashed_acepta_hasta_tres_detalles() -> None:
    roles = [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0), ("DETAIL", 1), ("DETAIL", 2)]

    assert len(parse_evidence_uploaded(_raw("CRASHED", roles), BUCKET).items(EvidenceRole.DETAIL)) == 3


def test_los_campos_extra_se_ignoran() -> None:
    raw = _raw()
    raw["actor"] = "no-deberia-estar"
    raw["evidence"][0]["extra"] = 1

    _parse(raw)


@pytest.mark.parametrize(
    ("alert_type", "roles"),
    [
        ("CRASHED", [("GENERAL", 0), ("PLATE", 0)]),  # falta DETAIL
        (
            "CRASHED",
            [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0), ("DETAIL", 1), ("DETAIL", 2), ("DETAIL", 3)],
        ),
        ("CRASHED", [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0), ("CONTEXT", 0)]),  # rol ajeno
        ("PARKED_ON_GARAGE", [("GENERAL", 0), ("PLATE", 0)]),  # falta CONTEXT
        ("PARKED_ON_GARAGE", [("GENERAL", 0), ("PLATE", 0), ("CONTEXT", 0), ("DETAIL", 0)]),
        ("PARKED_ON_GARAGE", [("GENERAL", 0), ("PLATE", 0), ("CONTEXT", 0), ("CONTEXT", 1)]),
        ("CRASHED", [("PLATE", 0), ("DETAIL", 0)]),  # falta GENERAL
        ("CRASHED", [("GENERAL", 0), ("GENERAL", 1), ("PLATE", 0), ("DETAIL", 0)]),
        ("CRASHED", [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0), ("DETAIL", 0)]),  # índice repetido
        ("CRASHED", [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 1)]),  # índice no contiguo
        ("CRASHED", []),
    ],
)
def test_roles_y_cantidades_segun_el_tipo(alert_type: str, roles: list[tuple[str, int]]) -> None:
    with pytest.raises(InvalidPayloadError):
        _parse(_raw(alert_type, roles))


def test_tipo_deshabilitado_o_desconocido() -> None:
    for alert_type in ("PARKED_TOO_CLOSE", "OTRO"):
        raw = _raw()
        raw["alert_type"] = alert_type
        with pytest.raises(InvalidPayloadError):
            _parse(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(alert_id="no-es-uuid"),
        lambda r: r.update(pat_enc_ia="***no-base64***"),
        lambda r: r.update(pat_enc_ia=""),
        lambda r: r.update(bucket="otro-bucket"),
        lambda r: r["evidence"][0].update(sha256="ABC"),
        lambda r: r["evidence"][0].update(sha256="A" * 64),
        lambda r: r["evidence"][0].update(key="alerts/otra-alerta/original/general-0.bin"),
        lambda r: r["evidence"][0].update(key=f"alerts/{r['alert_id']}/original/../../x.bin"),
        lambda r: r["evidence"][0].update(index=-1),
        lambda r: r["evidence"][0].update(role="OTRO"),
        lambda r: r.pop("pat_enc_ia"),
        lambda r: r.pop("evidence"),
    ],
)
def test_payload_invalido(mutate: Any) -> None:
    raw = _raw()
    mutate(raw)

    with pytest.raises(InvalidPayloadError):
        _parse(raw)


def test_payload_que_no_es_objeto() -> None:
    with pytest.raises(InvalidPayloadError):
        parse_evidence_uploaded([], BUCKET)  # type: ignore[arg-type]


def test_el_error_no_filtra_los_valores_recibidos() -> None:
    raw = _raw()
    raw["pat_enc_ia"] = "valor-secreto-no-base64!!"

    with pytest.raises(InvalidPayloadError) as exc:
        _parse(raw)

    assert "valor-secreto" not in str(exc.value)


def test_readable_alert_id_devuelve_el_uuid_canonico_aunque_el_resto_sea_invalido() -> None:
    alert_id = uuid.uuid4()

    assert readable_alert_id({"alert_id": str(alert_id)}) == str(alert_id)
    assert readable_alert_id({"alert_id": str(alert_id).upper(), "bucket": 3, "evidence": "x"}) == str(
        alert_id
    )
    assert readable_alert_id({"alert_id": alert_id.hex}) == str(alert_id)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"alert_id": None},
        {"alert_id": ""},
        {"alert_id": "x"},
        {"alert_id": 123},
        {"alert_id": ["a"]},
        {"alert_id": {"a": 1}},
        {"alert_id": str(uuid.uuid4())[:-1]},
    ],
)
def test_readable_alert_id_es_none_si_falta_o_no_es_un_uuid(raw: dict[str, Any]) -> None:
    assert readable_alert_id(raw) is None
