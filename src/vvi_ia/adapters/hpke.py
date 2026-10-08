"""HPKE RFC 9180, modo base: DHKEM(X25519, HKDF-SHA256) / HKDF-SHA256 / AES-256-GCM.

Formato del criptograma: `enc (32 bytes) ‖ ct`, el mismo que producen la app (Tink) y Vehicle (circl).
"""

import base64
import binascii

from pyhpke import AEADId, CipherSuite, KDFId, KEMId, KEMKeyInterface, PyHPKEError

from vvi_ia.service.ports import DecryptionError

ENC_SIZE = 32  # Nenc de DHKEM(X25519, HKDF-SHA256)
TAG_SIZE = 16  # AES-256-GCM
KEY_SIZE = 32  # clave privada X25519 cruda


class HpkeDecryptor:
    """Abre criptogramas dirigidos a IA con `IA_HPKE_PRIVATE_KEY`, atados a un `info`."""

    def __init__(self, private_key_b64: str) -> None:
        try:
            raw = base64.b64decode(private_key_b64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("la clave privada HPKE no es base64 estándar") from None
        if len(raw) != KEY_SIZE:
            raise ValueError(f"la clave privada HPKE debe tener {KEY_SIZE} bytes")
        self._suite = CipherSuite.new(KEMId.DHKEM_X25519_HKDF_SHA256, KDFId.HKDF_SHA256, AEADId.AES256_GCM)
        self._private_key: KEMKeyInterface = self._suite.kem.deserialize_private_key(raw)

    def open(self, ciphertext: bytes, info: bytes, aad: bytes = b"") -> bytes:
        """Descifra `enc‖ct` con el `info` dado (y `aad`, vacío en el flujo real).

        Cualquier falla se informa como `DecryptionError`.
        """
        if len(ciphertext) < ENC_SIZE + TAG_SIZE:
            raise DecryptionError("criptograma demasiado corto")
        enc, sealed = ciphertext[:ENC_SIZE], ciphertext[ENC_SIZE:]
        try:
            context = self._suite.create_recipient_context(enc, self._private_key, info)
            return context.open(sealed, aad)
        except (PyHPKEError, ValueError):
            raise DecryptionError("no se pudo descifrar el criptograma") from None
