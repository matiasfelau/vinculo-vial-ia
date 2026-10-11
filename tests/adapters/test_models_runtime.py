"""Manifest, verificación de sha256, selección por entorno y sesiones ONNX (con un modelo diminuto)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from vvi_ia.adapters.models.manifest import (
    PORTS,
    Commercial,
    ManifestError,
    load_manifest,
)
from vvi_ia.adapters.models.null import NullFaceDetector, NullVehicleDetector
from vvi_ia.adapters.models.onnx_session import OnnxSession
from vvi_ia.adapters.models.registry import (
    ADAPTERS,
    AdapterFactory,
    ModelContext,
    ModelSetupError,
    build_models,
    register,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def tiny_onnx() -> bytes:
    """Modelo ONNX mínimo generado en el test: y = x + 1."""
    node = helper.make_node("Add", ["x", "one"], ["y"])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 3])],
        [one],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def entry(name: str = "tiny", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": name,
        "version": "v1",
        "ports": ["detector", "face"],
        "file": f"{name}.onnx",
        "sha256": hashlib.sha256(tiny_onnx()).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2026},
        "source": "generado en el test",
        "commercial": "yes",
    }
    data.update(overrides)
    return data


class Env:
    """Manifest y MODELS_DIR temporales."""

    def __init__(self, tmp_path: Path, entries: list[dict[str, Any]] | None = None) -> None:
        self.models_dir = tmp_path / "weights"
        self.models_dir.mkdir()
        self.manifest = tmp_path / "manifest.json"
        self.write(entries if entries is not None else [entry()])
        (self.models_dir / "tiny.onnx").write_bytes(tiny_onnx())

    def write(self, entries: list[dict[str, Any]]) -> None:
        self.manifest.write_text(json.dumps({"schema_version": 1, "models": entries}), encoding="utf-8")


class Probe:
    """Adaptador de prueba: guarda el contexto con el que se lo creó."""

    def __init__(self, ctx: ModelContext) -> None:
        self.ctx = ctx
        self.version = ctx.model.version

    def detect(self, img: Any) -> None:
        return None


FAKE_ADAPTERS: dict[tuple[str, str], AdapterFactory] = {("detector", "tiny"): Probe, ("face", "tiny"): Probe}


def _build(env: Env, selection: dict[str, str], **kwargs: Any) -> Any:
    return build_models(selection, env.manifest, env.models_dir, adapters=FAKE_ADAPTERS, **kwargs)


# --- manifest ----------------------------------------------------------------------------


def test_el_manifest_versionado_del_repo_es_valido() -> None:
    manifest = load_manifest(REPO_ROOT / "models" / "manifest.json")

    assert manifest.schema_version == 1
    for model in manifest.models:
        assert set(model.ports) <= set(PORTS)


def test_carga_un_manifest_valido(tmp_path: Path) -> None:
    manifest = load_manifest(Env(tmp_path).manifest)

    model = manifest.get("tiny")
    assert model is not None
    assert model.commercial == Commercial.YES
    assert model.paper.venue == "ICLR"
    assert manifest.get("otro") is None


def test_version_es_nombre_arroba_doce_caracteres_del_sha256(tmp_path: Path) -> None:
    model = load_manifest(Env(tmp_path).manifest).get("tiny")
    assert model is not None

    assert model.version_label("abcdef0123456789" * 4) == "tiny@abcdef012345"


@pytest.mark.parametrize(
    "overrides",
    [
        {"commercial": "tal vez"},
        {"sha256": "ABC"},
        {"sha256": "A" * 64},
        {"file": "../fuera.onnx"},
        {"file": "sub/dir.onnx"},
        {"file": "C:\\\\x.onnx"},
        {"ports": []},
        {"ports": ["inexistente"]},
        {"name": "Con Mayúsculas"},
        {"paper": {"title": "x", "venue": "ICLR"}},
        {"paper": {"title": "x", "venue": "", "year": 2026}},
        {"license": ""},
        {"source": ""},
        {"campo_extra": 1},
    ],
)
def test_manifest_invalido(tmp_path: Path, overrides: dict[str, Any]) -> None:
    env = Env(tmp_path, [entry(**overrides)])

    with pytest.raises(ManifestError, match="manifest inválido"):
        load_manifest(env.manifest)


@pytest.mark.parametrize("field", ["name", "version", "sha256", "license", "paper", "source", "commercial"])
def test_campos_obligatorios_del_manifest(tmp_path: Path, field: str) -> None:
    data = entry()
    del data[field]

    with pytest.raises(ManifestError):
        load_manifest(Env(tmp_path, [data]).manifest)


def test_manifest_con_nombres_repetidos(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="repetidos"):
        load_manifest(Env(tmp_path, [entry(), entry()]).manifest)


def test_manifest_inexistente_o_corrupto(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no existe"):
        load_manifest(tmp_path / "nada.json")
    broken = tmp_path / "roto.json"
    broken.write_text("{no es json", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(broken)


# --- selección por defecto: null ---------------------------------------------------------


def test_sin_seleccion_todo_es_null_y_no_hace_falta_manifest_ni_models_dir(tmp_path: Path) -> None:
    models = build_models({}, tmp_path / "no-existe.json", None)

    assert set(models.versions().values()) == {"null"}
    assert isinstance(models.vehicle, NullVehicleDetector)


def test_seleccion_explicita_null(tmp_path: Path) -> None:
    models = build_models(dict.fromkeys(PORTS, "null"), tmp_path / "no-existe.json", None)

    assert set(models.versions().values()) == {"null"}


# --- selección de un modelo real ---------------------------------------------------------


def test_modelo_seleccionado_se_verifica_y_expone_su_version(tmp_path: Path) -> None:
    env = Env(tmp_path)

    models = _build(env, {"detector": "tiny"})

    sha = hashlib.sha256(tiny_onnx()).hexdigest()
    assert models.versions()["detector"] == f"tiny@{sha[:12]}"
    assert models.versions()["face"] == "null"
    assert isinstance(models.face, NullFaceDetector)


def test_la_sesion_cargada_es_la_del_archivo_verificado(tmp_path: Path) -> None:
    models = _build(Env(tmp_path), {"detector": "tiny"})

    session = models.vehicle.ctx.model.session  # type: ignore[attr-defined]
    (out,) = session.run(None, {"x": np.zeros((2, 3), np.float32)})
    assert out.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_un_modelo_en_dos_puertos_comparte_la_sesion(tmp_path: Path) -> None:
    models = _build(Env(tmp_path), {"detector": "tiny", "face": "tiny"})

    assert models.vehicle.ctx.model.session is models.face.ctx.model.session  # type: ignore[attr-defined]
    assert models.versions()["detector"] == models.versions()["face"]


def test_los_parametros_del_adaptador_llegan_a_la_fabrica(tmp_path: Path) -> None:
    models = _build(Env(tmp_path), {"detector": "tiny"}, params={"min_conf": 0.4})

    assert models.vehicle.ctx.params == {"min_conf": 0.4}  # type: ignore[attr-defined]


# --- el servicio no arranca --------------------------------------------------------------


def test_sha256_que_no_coincide(tmp_path: Path) -> None:
    env = Env(tmp_path)
    (env.models_dir / "tiny.onnx").write_bytes(tiny_onnx() + b"\x00")

    with pytest.raises(ModelSetupError, match="sha256"):
        _build(env, {"detector": "tiny"})


def test_archivo_faltante(tmp_path: Path) -> None:
    env = Env(tmp_path)
    (env.models_dir / "tiny.onnx").unlink()

    with pytest.raises(ModelSetupError, match="no se pudo leer"):
        _build(env, {"detector": "tiny"})


def test_sha256_ausente_en_el_manifest(tmp_path: Path) -> None:
    env = Env(tmp_path, [entry(sha256=None)])

    with pytest.raises(ModelSetupError, match="no tiene sha256"):
        _build(env, {"detector": "tiny"})


def test_models_dir_sin_definir(tmp_path: Path) -> None:
    env = Env(tmp_path)

    with pytest.raises(ModelSetupError, match="MODELS_DIR"):
        build_models({"detector": "tiny"}, env.manifest, None, adapters=FAKE_ADAPTERS)


def test_modelo_que_no_esta_en_el_manifest(tmp_path: Path) -> None:
    with pytest.raises(ModelSetupError, match="no está en el manifest"):
        _build(Env(tmp_path), {"detector": "fantasma"})


def test_commercial_no_no_se_puede_seleccionar(tmp_path: Path) -> None:
    env = Env(tmp_path, [entry(commercial="no")])

    with pytest.raises(ModelSetupError, match="commercial=no"):
        _build(env, {"detector": "tiny"})


@pytest.mark.parametrize("commercial", ["yes", "escalable"])
def test_yes_y_escalable_si_se_pueden_seleccionar(tmp_path: Path, commercial: str) -> None:
    env = Env(tmp_path, [entry(commercial=commercial)])

    assert _build(env, {"detector": "tiny"}).versions()["detector"].startswith("tiny@")


def test_modelo_que_no_sirve_a_ese_puerto(tmp_path: Path) -> None:
    env = Env(tmp_path, [entry(ports=["face"])])

    with pytest.raises(ModelSetupError, match="no sirve al puerto detector"):
        _build(env, {"detector": "tiny"})


def test_modelo_sin_adaptador_registrado(tmp_path: Path) -> None:
    env = Env(tmp_path, [entry(ports=["ocr"])])

    with pytest.raises(ModelSetupError, match="no hay un adaptador"):
        _build(env, {"ocr": "tiny"})


def test_manifest_faltante_con_un_modelo_seleccionado(tmp_path: Path) -> None:
    with pytest.raises(ModelSetupError, match="no existe el manifest"):
        build_models({"detector": "tiny"}, tmp_path / "nada.json", tmp_path, adapters=FAKE_ADAPTERS)


def test_archivo_que_no_es_onnx_aunque_el_sha256_coincida(tmp_path: Path) -> None:
    garbage = b"esto no es un modelo"
    env = Env(tmp_path, [entry(sha256=hashlib.sha256(garbage).hexdigest())])
    (env.models_dir / "tiny.onnx").write_bytes(garbage)

    with pytest.raises(ModelSetupError, match="ONNX Runtime no pudo cargar"):
        _build(env, {"detector": "tiny"})


def test_una_fabrica_que_falla_no_deja_arrancar(tmp_path: Path) -> None:
    def broken(ctx: ModelContext) -> Any:
        raise ValueError("salida inesperada")

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models(
            {"detector": "tiny"},
            Env(tmp_path).manifest,
            tmp_path / "weights",
            adapters={("detector", "tiny"): broken},
        )


def test_puerto_desconocido_en_la_seleccion(tmp_path: Path) -> None:
    with pytest.raises(ModelSetupError, match="puertos desconocidos"):
        _build(Env(tmp_path), {"radar": "tiny"})


def test_los_mensajes_de_error_no_incluyen_rutas_del_host(tmp_path: Path) -> None:
    env = Env(tmp_path)
    (env.models_dir / "tiny.onnx").unlink()

    with pytest.raises(ModelSetupError) as exc:
        _build(env, {"detector": "tiny"})

    assert str(tmp_path) not in str(exc.value)


# --- registro y sesión -------------------------------------------------------------------


def test_register_agrega_y_valida_el_puerto() -> None:
    try:
        register("ocr", "de_prueba")(Probe)
        assert ADAPTERS[("ocr", "de_prueba")] is Probe
    finally:
        ADAPTERS.pop(("ocr", "de_prueba"), None)
    with pytest.raises(ValueError, match="puerto desconocido"):
        register("radar", "x")


def test_onnx_session_corre_en_cpu_y_expone_entradas_y_salidas() -> None:
    session = OnnxSession(tiny_onnx(), intra_op_threads=1)

    assert session.input_names == ("x",)
    assert session.output_names == ("y",)
    assert session.input_shape() == ("n", 3)
    assert session._session.get_providers() == ["CPUExecutionProvider"]
    (out,) = session.run(["y"], {"x": np.array([[1, 2, 3]], np.float32)})
    assert out.tolist() == [[2.0, 3.0, 4.0]]


# --- telemetría de ONNX Runtime apagada ---------------------------------------------------

SRC = REPO_ROOT / "src" / "vvi_ia"


def test_la_telemetria_se_apaga_antes_de_importar_onnxruntime() -> None:
    import os

    source = (SRC / "adapters" / "models" / "onnx_session.py").read_text(encoding="utf-8")

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert source.index('os.environ["ORT_DISABLE_TELEMETRY"] = "1"') < source.index("import onnxruntime")
    assert "ort.disable_telemetry_events()" in source


def test_solo_onnx_session_importa_onnxruntime() -> None:
    importers = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "import onnxruntime" in path.read_text(encoding="utf-8")
        or "from onnxruntime" in path.read_text(encoding="utf-8")
    ]

    assert importers == ["adapters/models/onnx_session.py"]


def test_la_imagen_apaga_la_telemetria_por_entorno() -> None:
    assert "ORT_DISABLE_TELEMETRY=1" in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
