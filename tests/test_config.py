import pytest
from pydantic import ValidationError

from tests.conftest import BASE_ENV
from vvi_ia.config import Settings, load_settings


def test_carga_valida_con_valores_por_defecto(base_env: pytest.MonkeyPatch) -> None:
    settings = load_settings()

    assert settings.input_queue == "ia.alert-evidence-uploaded.q"
    assert settings.workers == 2
    assert settings.redaction_required is True
    assert settings.reid_t_low < settings.reid_t_high


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


def test_umbrales_reid_deben_estar_ordenados(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("REID_T_LOW", "0.8")
    base_env.setenv("REID_T_HIGH", "0.7")

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
