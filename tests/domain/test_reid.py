import numpy as np
import pytest
from numpy.typing import NDArray

from vvi_ia.domain.reid import GRAY_FILL, ReidOutcome, best_match, cosine, masked_crop, same_vehicle

T_LOW, T_HIGH = 0.5, 0.7


def _unit(angle_deg: float) -> NDArray[np.float64]:
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)])


def test_masked_crop_recorta_la_caja_y_rellena_el_fondo_en_gris() -> None:
    img = np.full((10, 10, 3), 200, np.uint8)
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3:7] = True
    mask[2, 3] = False  # esquina fuera de la máscara pero dentro de la caja

    crop = masked_crop(img, mask)

    assert crop is not None
    assert crop.shape == (3, 4, 3)
    assert (crop[0, 0] == GRAY_FILL).all()
    assert (crop[1, 1] == 200).all()


def test_masked_crop_no_modifica_la_imagen_original() -> None:
    img = np.full((6, 6, 3), 7, np.uint8)
    mask = np.zeros((6, 6), bool)
    mask[1:3, 1:3] = True
    mask[1, 1] = False

    masked_crop(img, mask)

    assert (img == 7).all()


def test_masked_crop_con_mascara_vacia_es_none() -> None:
    assert masked_crop(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), bool)) is None


def test_cosine_de_vectores_conocidos() -> None:
    assert cosine(_unit(0), _unit(0)) == pytest.approx(1.0)
    assert cosine(_unit(0), _unit(90)) == pytest.approx(0.0, abs=1e-9)
    assert cosine(_unit(0), _unit(180)) == pytest.approx(-1.0)


def test_cosine_con_vector_nulo_es_cero() -> None:
    assert cosine(np.zeros(2), _unit(0)) == 0.0


def test_cosine_exige_vectores_1d_del_mismo_largo() -> None:
    with pytest.raises(ValueError, match="1-D"):
        cosine(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="1-D"):
        cosine(np.ones((2, 2)), np.ones((2, 2)))


def test_same_vehicle_sobre_t_high_es_same() -> None:
    assert same_vehicle(_unit(0), _unit(30), T_LOW, T_HIGH).outcome == ReidOutcome.SAME  # cos 0.87


def test_same_vehicle_exactamente_en_t_high_es_same() -> None:
    a, b = np.array([3.0, 4.0]), np.array([1.0, 0.0])  # cos = 0.6 exacto

    assert same_vehicle(a, b, 0.5, 0.6).outcome == ReidOutcome.SAME


def test_same_vehicle_entre_umbrales_es_uncertain() -> None:
    result = same_vehicle(_unit(0), _unit(55), T_LOW, T_HIGH)  # cos 0.57

    assert result.outcome == ReidOutcome.UNCERTAIN
    assert result.similarity == pytest.approx(np.cos(np.deg2rad(55)))


def test_same_vehicle_exactamente_en_t_low_es_uncertain() -> None:
    a, b = np.array([3.0, 4.0]), np.array([1.0, 0.0])  # cos = 0.6 exacto

    assert same_vehicle(a, b, 0.6, 0.9).outcome == ReidOutcome.UNCERTAIN


def test_same_vehicle_bajo_t_low_es_different() -> None:
    assert same_vehicle(_unit(0), _unit(80), T_LOW, T_HIGH).outcome == ReidOutcome.DIFFERENT


def test_best_match_elige_el_candidato_mas_parecido_a_alguna_referencia() -> None:
    candidates = [_unit(80), _unit(10), _unit(70)]
    references = [_unit(0), _unit(45)]

    index, score = best_match(candidates, references) or (-1, 0.0)

    assert index == 1
    assert score == pytest.approx(np.cos(np.deg2rad(10)))


def test_best_match_sin_candidatos_o_sin_referencias_es_none() -> None:
    assert best_match([], [_unit(0)]) is None
    assert best_match([_unit(0)], []) is None
