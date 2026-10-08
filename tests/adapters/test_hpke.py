import base64
import json
from pathlib import Path

import pytest

from vvi_ia.adapters.hpke import ENC_SIZE, HpkeDecryptor
from vvi_ia.service.ports import DecryptionError

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict[str, str]:
    data: dict[str, str] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


# --- vector de prueba de RFC 9180 (modo base, X25519/HKDF-SHA256/AES-256-GCM) -----------


@pytest.fixture
def rfc() -> dict[str, object]:
    vector: dict[str, object] = json.loads(
        (FIXTURES / "hpke_rfc9180_vector.json").read_text(encoding="utf-8")
    )
    return vector


def _rfc_parts(vector: dict[str, object]) -> tuple[HpkeDecryptor, bytes, bytes, bytes, bytes]:
    first = vector["encryptions"][0]  # type: ignore[index]
    decryptor = HpkeDecryptor(_b64(bytes.fromhex(str(vector["skRm"]))))
    ciphertext = bytes.fromhex(str(vector["enc"])) + bytes.fromhex(first["ct"])
    return (
        decryptor,
        ciphertext,
        bytes.fromhex(str(vector["info"])),
        bytes.fromhex(first["aad"]),
        bytes.fromhex(first["pt"]),
    )


def test_vector_rfc9180(rfc: dict[str, object]) -> None:
    assert (rfc["mode"], rfc["kem_id"], rfc["kdf_id"], rfc["aead_id"]) == (0, 32, 1, 2)
    decryptor, ciphertext, info, aad, expected = _rfc_parts(rfc)

    assert decryptor.open(ciphertext, info, aad=aad) == expected


def test_vector_rfc9180_con_otro_info_falla(rfc: dict[str, object]) -> None:
    decryptor, ciphertext, _, aad, _ = _rfc_parts(rfc)

    with pytest.raises(DecryptionError):
        decryptor.open(ciphertext, b"otro info", aad=aad)


def test_vector_rfc9180_con_otro_aad_falla(rfc: dict[str, object]) -> None:
    decryptor, ciphertext, info, _, _ = _rfc_parts(rfc)

    with pytest.raises(DecryptionError):
        decryptor.open(ciphertext, info, aad=b"otro aad")


# --- fixture generado con el Encryptor de Vehicle (circl) -------------------------------


@pytest.fixture
def vehicle() -> dict[str, str]:
    return _load("hpke_vehicle_fixture.json")


def test_abre_pat_enc_ia_generado_por_vehicle(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])

    plaintext = decryptor.open(base64.b64decode(vehicle["pat_enc_ia_b64"]), vehicle["alert_id"].encode())

    assert plaintext.decode() == vehicle["pat_plaintext"]


def test_abre_una_foto_cifrada_por_vehicle(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])

    plaintext = decryptor.open(base64.b64decode(vehicle["photo_enc_b64"]), vehicle["alert_id"].encode())

    assert plaintext == base64.b64decode(vehicle["photo_plain_b64"])


def test_con_un_info_distinto_falla(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])

    with pytest.raises(DecryptionError):
        decryptor.open(base64.b64decode(vehicle["pat_enc_ia_b64"]), b"otro-alert-id")


def test_cifrado_sin_info_solo_abre_con_info_vacio(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])
    ciphertext = base64.b64decode(vehicle["empty_info_enc_b64"])

    assert decryptor.open(ciphertext, b"").decode() == vehicle["pat_plaintext"]
    with pytest.raises(DecryptionError):
        decryptor.open(ciphertext, vehicle["alert_id"].encode())


def test_criptograma_alterado_falla(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])
    raw = bytearray(base64.b64decode(vehicle["pat_enc_ia_b64"]))
    raw[-1] ^= 0x01

    with pytest.raises(DecryptionError):
        decryptor.open(bytes(raw), vehicle["alert_id"].encode())


def test_enc_alterado_falla(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])
    raw = bytearray(base64.b64decode(vehicle["pat_enc_ia_b64"]))
    raw[0] ^= 0x01

    with pytest.raises(DecryptionError):
        decryptor.open(bytes(raw), vehicle["alert_id"].encode())


def test_otra_clave_privada_no_abre(vehicle: dict[str, str]) -> None:
    decryptor = HpkeDecryptor(_b64(bytes(range(32))))

    with pytest.raises(DecryptionError):
        decryptor.open(base64.b64decode(vehicle["pat_enc_ia_b64"]), vehicle["alert_id"].encode())


@pytest.mark.parametrize("size", [0, 1, ENC_SIZE, ENC_SIZE + 15])
def test_criptograma_demasiado_corto(vehicle: dict[str, str], size: int) -> None:
    decryptor = HpkeDecryptor(vehicle["private_key_b64"])

    with pytest.raises(DecryptionError):
        decryptor.open(b"\x01" * size, b"x")


@pytest.mark.parametrize("key", ["no-base64!!", _b64(b"corta"), _b64(bytes(33)), ""])
def test_clave_privada_invalida(key: str) -> None:
    with pytest.raises(ValueError, match="clave privada"):
        HpkeDecryptor(key)
