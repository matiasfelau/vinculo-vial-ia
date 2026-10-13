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
    "EVIDENCE_MAX_PART_BYTES",
    "SEGMENTER_MIN_CONF",
    "DETECTOR_MIN_CONF",
    "DAMAGE_T_LOW",
    "DAMAGE_T_HIGH",
    "CURB_MIN_CONF",
    "PRESENCE_MIN_AREA",
    "RAMP_T_LOW",
    "RAMP_T_HIGH",
    "FACE_TILING",
    "PLATE_TILING",
    "TILING_TILE_PX",
    "TILING_OVERLAP",
    "MODEL_DETECTOR",
    "MODEL_FACE",
    "MODEL_SEGMENTER",
    "MODEL_CURB",
    "MODEL_EMBEDDER",
    "MODEL_OCR",
    "MODEL_DAMAGE",
    "MODELS_DIR",
    "MODELS_MANIFEST",
    "ONNX_INTRA_OP_THREADS",
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Los tests marcados `models` usan pesos reales: se omiten si no hay MODELS_DIR."""
    if os.environ.get("MODELS_DIR"):
        return
    skip = pytest.mark.skip(reason="MODELS_DIR no definido")
    for item in items:
        if item.get_closest_marker("models") is not None:
            item.add_marker(skip)
