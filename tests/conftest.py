import base64
import os
from pathlib import Path

import pytest

VALID_HPKE_KEY = base64.b64encode(bytes(range(32))).decode()

BASE_ENV = {
    "RABBITMQ_URL": "amqp://user:secret@broker:5672/",
    "S3_REGION": "us-east-1",
    "EVIDENCE_BUCKET": "bucket-de-prueba",
    "IA_HPKE_PRIVATE_KEY": VALID_HPKE_KEY,
    "ENV": "local",
}

_CONFIG_KEYS = {
    *BASE_ENV,
    "REDACTION_REQUIRED",
    "REID_T_LOW",
    "REID_T_HIGH",
    "WORKERS",
    "HEALTH_PORT",
    "MAX_RETRIES",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """Entorno aislado: sin variables heredadas ni .env del directorio de trabajo."""
    for key in list(os.environ):
        if key.upper() in _CONFIG_KEYS:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


@pytest.fixture
def base_env(clean_env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key, value in BASE_ENV.items():
        clean_env.setenv(key, value)
    return clean_env
