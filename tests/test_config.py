from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.conftest import BASE_ENV
from vvi_ia.config import Settings, load_settings


def test_carga_valida_con_valores_por_defecto(base_env: pytest.MonkeyPatch) -> None:
    settings = load_settings()

    assert settings.input_queue == "ia.alert-evidence-uploaded.q"
    assert settings.workers == 2
    assert settings.redaction_required is True
    assert settings.reid_t == 0.075 and settings.damage_presence_t == 0.97
    assert settings.model_curb == "" and settings.model_selection()["curb"] == "null"


@pytest.mark.parametrize("missing", sorted(BASE_ENV))
def test_falla_si_falta_una_variable_obligatoria(base_env: pytest.MonkeyPatch, missing: str) -> None:
    base_env.delenv(missing)

    with pytest.raises(ValidationError):
        load_settings()


@pytest.mark.parametrize("key", ["reemplazar", "###", "AAAA"])
def test_rechaza_clave_hpke_invalida(base_env: pytest.MonkeyPatch, key: str) -> None:
    base_env.setenv("IA_HPKE_PRIVATE_KEY", key)

    with pytest.raises(ValidationError):
        load_settings()


def test_redaction_required_false_solo_con_env_local(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("REDACTION_REQUIRED", "false")
    base_env.setenv("ENV", "prod")

    with pytest.raises(ValidationError):
        load_settings()


def test_redaction_required_false_con_env_local_arranca(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("REDACTION_REQUIRED", "false")

    assert load_settings().redaction_required is False


def test_el_umbral_de_reid_debe_estar_entre_0_y_1(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("REID_T", "1.5")

    with pytest.raises(ValidationError):
        load_settings()


def test_el_modelo_de_rebajes_vacio_no_carga_pesos_y_con_nombre_exige_models_dir(
    base_env: pytest.MonkeyPatch,
) -> None:
    assert load_settings().model_selection()["curb"] == "null"

    base_env.setenv("MODEL_CURB", "rfdetr_seg_curb")
    base_env.delenv("MODELS_DIR", raising=False)
    with pytest.raises(ValidationError):
        load_settings()


def test_workers_debe_ser_positivo(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("WORKERS", "0")

    with pytest.raises(ValidationError):
        load_settings()


def test_los_secretos_no_se_exponen_en_repr(base_env: pytest.MonkeyPatch) -> None:
    rendered = repr(load_settings())

    assert "secret" not in rendered
    assert BASE_ENV["IA_HPKE_PRIVATE_KEY"] not in rendered


def test_main_sale_con_codigo_distinto_de_cero_si_la_config_es_invalida(
    clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vvi_ia.main import EXIT_BAD_CONFIG, main

    assert main() == EXIT_BAD_CONFIG
    assert "configuración inválida" in capsys.readouterr().out


def test_settings_se_construye_con_kwargs() -> None:
    settings = Settings(  # type: ignore[call-arg]
        rabbitmq_url="amqp://x",  # type: ignore[arg-type]
        s3_region="us-east-1",
        evidence_bucket="b",
        ia_hpke_private_key=BASE_ENV["IA_HPKE_PRIVATE_KEY"],  # type: ignore[arg-type]
        env="local",
    )
    assert settings.evidence_bucket == "b"


def test_el_arranque_falla_con_redaction_required_false_fuera_de_local(
    base_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vvi_ia.main import EXIT_BAD_CONFIG, main

    base_env.setenv("REDACTION_REQUIRED", "false")
    base_env.setenv("ENV", "dev")

    assert main() == EXIT_BAD_CONFIG
    assert "REDACTION_REQUIRED=false solo se acepta con ENV=local" in capsys.readouterr().out


def test_evidence_max_part_bytes_por_defecto_es_1_mib_y_es_configurable(base_env: pytest.MonkeyPatch) -> None:
    assert load_settings().evidence_max_part_bytes == 1_048_576

    base_env.setenv("EVIDENCE_MAX_PART_BYTES", "2048")
    assert load_settings().evidence_max_part_bytes == 2048

    base_env.setenv("EVIDENCE_MAX_PART_BYTES", "0")
    with pytest.raises(ValidationError):
        load_settings()


# --- selección de modelos ------------------------------------------------------------------


def test_por_defecto_todos_los_modelos_son_null(base_env: pytest.MonkeyPatch) -> None:
    settings = load_settings()

    assert set(settings.model_selection()) == {
        "detector",
        "face",
        "segmenter",
        "curb",
        "embedder",
        "ocr",
        "damage",
    }
    assert set(settings.model_selection().values()) == {"null"}
    assert settings.models_dir is None


def test_un_modelo_seleccionado_exige_models_dir(base_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_env.setenv("MODEL_OCR", "parseq")
    with pytest.raises(ValidationError):
        load_settings()

    base_env.setenv("MODELS_DIR", str(tmp_path))
    assert load_settings().model_selection()["ocr"] == "parseq"


@pytest.mark.parametrize("value", ["", "Con Espacios", "../otro", "MAYUS"])
def test_nombre_de_modelo_invalido(base_env: pytest.MonkeyPatch, value: str) -> None:
    base_env.setenv("MODEL_FACE", value)

    with pytest.raises(ValidationError):
        load_settings()


def test_el_arranque_falla_si_un_modelo_seleccionado_no_se_puede_verificar(
    base_env: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vvi_ia.main import EXIT_BAD_MODELS, main

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.json").write_text(
        '{"schema_version": 1, "models": []}', encoding="utf-8"
    )
    base_env.setenv("MODEL_EMBEDDER", "dinov2_vits14")
    base_env.setenv("MODELS_DIR", str(tmp_path))

    assert main() == EXIT_BAD_MODELS
    assert "modelos inválidos" in capsys.readouterr().out


def test_con_null_load_models_no_necesita_manifest(base_env: pytest.MonkeyPatch) -> None:
    from vvi_ia.main import load_models

    assert set(load_models(load_settings()).versions().values()) == {"null"}


# --- la imagen no lleva herramientas de entrenamiento ---------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ONLY = ("torch", "torchvision", "transformers", "rfdetr", "fiftyone", "sam3", "timm")


def test_las_dependencias_base_no_incluyen_frameworks_de_entrenamiento() -> None:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base = [dep.split(">")[0].split("=")[0].split("[")[0].strip().lower() for dep in project["dependencies"]]

    assert "onnxruntime" in base
    assert not set(base) & set(TRAINING_ONLY)


def test_la_imagen_no_instala_extras_ni_copia_pesos() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--extra" not in dockerfile
    assert "--all-extras" not in dockerfile
    assert "COPY models/manifest.json" in dockerfile
    assert "COPY models " not in dockerfile and "COPY . " not in dockerfile
