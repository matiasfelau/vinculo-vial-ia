"""tools/training/reid/train.py: preproceso igual al del servicio, aumentos, CosFace y métricas."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # solo en la máquina de entrenamiento: CI no instala los extras de tools
pytest.importorskip("transformers")

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDER = REPO_ROOT / "tools" / "training" / "reid"


def load(name: str) -> ModuleType:
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location(f"reid_train_{name}", FOLDER / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"reid_train_{name}"] = module
    spec.loader.exec_module(module)
    return module


train = load("train")
dataset = load("dataset")


def crop(h: int = 120, w: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_el_preproceso_de_evaluacion_es_el_del_servicio() -> None:
    from vvi_ia.adapters.models.dinov2 import preprocess

    image = crop()

    expected = preprocess(image, np.ones(image.shape[:2], np.bool_))

    assert expected is not None
    np.testing.assert_allclose(train.eval_tensor(image).numpy(), expected[0])


def test_la_normalizacion_del_entrenamiento_coincide_con_la_del_servicio() -> None:
    from vvi_ia.adapters.models.dinov2 import preprocess

    image = crop(224, 224)  # ya es de 224 x 224: no hay achique que diferencie los caminos
    expected = preprocess(image, np.ones(image.shape[:2], np.bool_))

    assert expected is not None
    np.testing.assert_allclose(train.normalize(image).numpy(), expected[0], atol=1e-6)


def test_augment_devuelve_224_con_semilla_reproducible_y_cambia_la_imagen() -> None:
    image = crop(300, 180)

    a = train.augment(image, np.random.default_rng(1))
    b = train.augment(image, np.random.default_rng(1))
    c = train.augment(image, np.random.default_rng(2))

    assert a.shape == (224, 224, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augment_aguanta_recortes_chicos() -> None:
    assert train.augment(crop(9, 9), np.random.default_rng(0)).shape == (224, 224, 3)


def test_cosface_baja_la_perdida_cuando_el_embedding_esta_en_su_prototipo() -> None:
    prototypes = torch.eye(4)
    head = train.CosFace(prototypes)
    labels = torch.tensor([0, 1, 2, 3])

    aligned = head(torch.eye(4), labels)
    shuffled = head(torch.eye(4)[[1, 2, 3, 0]], labels)

    assert aligned.item() < shuffled.item()


def test_cosface_el_margen_endurece_la_pertenencia() -> None:
    head = train.CosFace(torch.eye(2))
    labels = torch.tensor([0])
    embedding = torch.tensor([[0.8, 0.6]])

    with_margin = head(embedding, labels).item()
    cosine_only = torch.nn.functional.cross_entropy(train.SCALE * (embedding @ torch.eye(2).T), labels).item()

    assert with_margin > cosine_only


def test_validation_metrics_separa_bien_a_embeddings_bien_separados() -> None:
    rng = np.random.default_rng(0)
    centers = np.eye(8)
    items, vectors = [], []
    for car in range(8):
        for frame in range(6):
            items.append(
                {
                    "auto": f"c{car}",
                    "foto": frame,
                    "azimut": frame * 60.0,
                    "marca": "X",
                    "modelo": "m",
                    "luz": "standard",
                    "tipo": "SUV",
                    "color": "blanco",
                }
            )
            vector = centers[car] + 0.05 * rng.normal(size=8)
            vectors.append(vector / np.linalg.norm(vector))
    pairs = dataset.build_pairs(items, different_limit=500, hard_limit=100, seed=0)

    metrics = train.validation_metrics(np.array(vectors), pairs)

    assert metrics["auc"] > 0.99
    assert (
        metrics["umbral"] is None or metrics["S4"] > 0.9
    )  # con pocos pares la cota de U1 puede no alcanzarse
