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
    return {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "pat_enc_ia": base64.b64encode(b"x" * 48).decode(),
        "bucket": BUCKET,
        "evidence": [_item(alert_id, role, index) for role, index in roles],
    }


def _parse(raw: dict[str, Any]) -> None:
    parse_evidence_uploaded(raw, BUCKET)


@pytest.mark.parametrize("alert_type", [AlertType.CRASHED, AlertType.PARKED_ON_GARAGE])
def test_los_dos_tipos_aceptan_exactamente_general_y_plate(alert_type: AlertType) -> None:
    payload = parse_evidence_uploaded(_raw(alert_type.value), BUCKET)

    assert payload.alert_type == alert_type
    assert [i.role for i in payload.evidence] == [EvidenceRole.GENERAL, EvidenceRole.PLATE]
    assert [i.key for i in payload.items(EvidenceRole.PLATE)] == [payload.evidence[1].key]


def test_el_orden_de_las_partes_no_importa() -> None:
    payload = parse_evidence_uploaded(_raw("CRASHED", [("PLATE", 0), ("GENERAL", 0)]), BUCKET)

    assert {i.role for i in payload.evidence} == {EvidenceRole.GENERAL, EvidenceRole.PLATE}


def test_los_roles_detail_y_context_ya_no_existen() -> None:
    assert {role.value for role in EvidenceRole} == {"GENERAL", "PLATE"}


def test_los_campos_extra_se_ignoran() -> None:
    raw = _raw()
    raw["actor"] = "no-deberia-estar"
    raw["evidence"][0]["extra"] = 1

    _parse(raw)


@pytest.mark.parametrize("alert_type", ["CRASHED", "PARKED_ON_GARAGE"])
@pytest.mark.parametrize(
    "roles",
    [
        [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0)],  # la app vieja: un detalle de más
        [("GENERAL", 0), ("PLATE", 0), ("CONTEXT", 0)],  # la app vieja: un contexto de más
        [("GENERAL", 0), ("DETAIL", 0)],  # un detalle en lugar de la patente
        [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0), ("DETAIL", 1), ("DETAIL", 2)],
        [("GENERAL", 0), ("PLATE", 0), ("GENERAL", 1)],  # tres partes
        [("GENERAL", 0), ("PLATE", 0), ("PLATE", 1)],
        [("GENERAL", 0), ("GENERAL", 1), ("PLATE", 0)],
        [("GENERAL", 0)],  # falta PLATE
        [("PLATE", 0)],  # falta GENERAL
        [("GENERAL", 1), ("PLATE", 0)],  # índice no contiguo
        [("GENERAL", 0), ("PLATE", 1)],
        [],
    ],
)
def test_cualquier_otra_cosa_que_general_y_plate_es_invalida(
    alert_type: str, roles: list[tuple[str, int]]
) -> None:
    with pytest.raises(InvalidPayloadError):
        _parse(_raw(alert_type, roles))


def test_el_rechazo_de_una_parte_detail_no_filtra_sus_valores() -> None:
    raw = _raw("CRASHED", [("GENERAL", 0), ("PLATE", 0), ("DETAIL", 0)])

    with pytest.raises(InvalidPayloadError) as exc:
        _parse(raw)

    assert raw["evidence"][2]["key"] not in str(exc.value) and raw["alert_id"] not in str(exc.value)


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
