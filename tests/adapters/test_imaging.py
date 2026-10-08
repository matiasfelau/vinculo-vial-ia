import struct

import cv2
import numpy as np
import pytest

from vvi_ia.adapters.imaging import (
    CvImageDecoder,
    encode_jpeg,
    fill_boxes,
    jpeg_dimensions,
    resize_max_side,
)
from vvi_ia.domain.model import Box, Pixels
from vvi_ia.service.ports import ImageDecodeError


def _gradient(height: int, width: int) -> Pixels:
    """Imagen RGB con canales distintos, para detectar confusiones de orden."""
    img = np.zeros((height, width, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]  # R crece a la derecha
    img[..., 1] = 40
    img[..., 2] = 200
    return img


def _jpeg(height: int = 40, width: int = 60) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(_gradient(height, width), cv2.COLOR_RGB2BGR))
    assert ok
    return bytes(buf.tobytes())


def _exif_app1(orientation: int) -> bytes:
    """Segmento APP1/Exif mínimo con la etiqueta de orientación."""
    tiff = b"MM\x00*\x00\x00\x00\x08" + struct.pack(">H", 1) + struct.pack(">HHI", 0x0112, 3, 1)
    tiff += struct.pack(">H", orientation) + b"\x00\x00" + b"\x00\x00\x00\x00"
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _with_exif(data: bytes, orientation: int) -> bytes:
    return data[:2] + _exif_app1(orientation) + data[2:]


def _patch_dimensions(data: bytes, width: int, height: int) -> bytes:
    sof = data.index(b"\xff\xc0")
    return data[: sof + 5] + struct.pack(">HH", height, width) + data[sof + 9 :]


# --- decodificación segura ---------------------------------------------------------------


def test_decodifica_un_jpeg_a_rgb() -> None:
    pixels = CvImageDecoder().decode(_jpeg(40, 60))

    assert pixels.shape == (40, 60, 3)
    assert pixels[0, -1, 0] > pixels[0, 0, 0]  # el canal R sigue creciendo hacia la derecha
    assert pixels[0, 0, 2] > pixels[0, 0, 1]  # B > G: no quedó en BGR


def test_lee_el_tamano_del_encabezado() -> None:
    assert jpeg_dimensions(_jpeg(40, 60)) == (60, 40)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        b"GIF89a" + b"\x00" * 50,
        b"\xff\xd8",
        b"\xff\xd8\xff",
        b"texto plano",
    ],
)
def test_rechaza_lo_que_no_es_jpeg_por_su_firma(data: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        CvImageDecoder().decode(data)


def test_rechaza_jpeg_truncado() -> None:
    data = _jpeg()

    with pytest.raises(ImageDecodeError):
        CvImageDecoder().decode(data[: len(data) // 2])


def test_rechaza_jpeg_sin_encabezado_de_tamano() -> None:
    with pytest.raises(ImageDecodeError):
        CvImageDecoder().decode(b"\xff\xd8\xff\xe0\x00\x04ab" + b"\xff\xda\x00\x02")


def test_rechaza_imagenes_sobre_el_limite_de_pixeles_sin_decodificarlas() -> None:
    bomba = _patch_dimensions(_jpeg(), 20_000, 20_000)  # 400 MP declarados

    with pytest.raises(ImageDecodeError, match="fuera de rango"):
        CvImageDecoder().decode(bomba)


def test_el_limite_de_pixeles_es_configurable_y_se_aplica_en_el_borde() -> None:
    data = _jpeg(10, 10)

    assert CvImageDecoder(max_pixels=100).decode(data).shape == (10, 10, 3)
    with pytest.raises(ImageDecodeError):
        CvImageDecoder(max_pixels=99).decode(data)


def test_rechaza_dimensiones_cero() -> None:
    with pytest.raises(ImageDecodeError):
        CvImageDecoder().decode(_patch_dimensions(_jpeg(), 0, 10))


def test_aplica_la_orientacion_exif_al_decodificar() -> None:
    upright = CvImageDecoder().decode(_jpeg(40, 60))
    rotated = CvImageDecoder().decode(_with_exif(_jpeg(40, 60), 6))

    assert upright.shape[:2] == (40, 60)
    assert rotated.shape[:2] == (60, 40)


# --- redimensionado ----------------------------------------------------------------------


def test_resize_reduce_el_lado_mayor_manteniendo_la_proporcion() -> None:
    out = resize_max_side(_gradient(1000, 2000), 1600)

    assert out.shape[:2] == (800, 1600)


def test_resize_vertical() -> None:
    assert resize_max_side(_gradient(3000, 1500), 1600).shape[:2] == (1600, 800)


def test_resize_no_agranda_ni_toca_lo_que_ya_entra() -> None:
    img = _gradient(300, 400)

    assert resize_max_side(img, 1600) is img
    assert resize_max_side(_gradient(1600, 1000), 1600).shape[:2] == (1600, 1000)


# --- relleno opaco -----------------------------------------------------------------------


def test_fill_cubre_la_caja_con_relleno_opaco_y_no_toca_el_resto() -> None:
    img = _gradient(50, 80)

    out = fill_boxes(img, [Box(10, 5, 30, 25)])

    assert (out[5:25, 10:30] == 0).all()
    mask = np.ones(img.shape[:2], bool)
    mask[5:25, 10:30] = False
    assert (out[mask] == img[mask]).all()


def test_fill_no_modifica_la_imagen_original() -> None:
    img = _gradient(20, 20)
    before = img.copy()

    fill_boxes(img, [Box(0, 0, 20, 20)])

    assert (img == before).all()


def test_fill_cubre_los_bordes_fraccionarios_hacia_afuera() -> None:
    out = fill_boxes(np.full((20, 20, 3), 255, np.uint8), [Box(4.5, 4.5, 9.2, 9.2)])

    assert (out[4:10, 4:10] == 0).all()
    assert out[3, 4, 0] == 255
    assert out[10, 4, 0] == 255


def test_fill_recorta_cajas_que_se_salen_de_la_imagen() -> None:
    out = fill_boxes(np.full((10, 10, 3), 255, np.uint8), [Box(-5, -5, 4, 4), Box(8, 8, 50, 50)])

    assert (out[:4, :4] == 0).all()
    assert (out[8:, 8:] == 0).all()
    assert out[5, 5, 0] == 255


def test_fill_ignora_cajas_vacias_o_fuera() -> None:
    img = np.full((10, 10, 3), 255, np.uint8)

    assert (fill_boxes(img, [Box(5, 5, 5, 9), Box(20, 20, 30, 30)]) == 255).all()


def test_fill_con_valor_opaco_no_es_un_blur() -> None:
    noisy = np.random.default_rng(0).integers(0, 256, (30, 30, 3), dtype=np.uint8)

    out = fill_boxes(noisy, [Box(5, 5, 25, 25)], value=0)

    assert np.unique(out[5:25, 5:25]).tolist() == [0]


# --- codificación sin EXIF ---------------------------------------------------------------


def test_encode_produce_un_jpeg_valido_que_se_puede_decodificar() -> None:
    data = encode_jpeg(_gradient(40, 60))

    assert data.startswith(b"\xff\xd8\xff")
    assert CvImageDecoder().decode(data).shape == (40, 60, 3)


def test_encode_no_conserva_el_exif_de_la_entrada() -> None:
    original = _with_exif(_jpeg(40, 60), 6)
    assert b"Exif" in original
    pixels = CvImageDecoder().decode(original)

    out = encode_jpeg(pixels)

    assert b"Exif" not in out
    assert b"\xff\xe1" not in out  # ningún segmento APP1


def test_encode_preserva_el_orden_de_canales() -> None:
    img = np.zeros((16, 16, 3), np.uint8)
    img[..., 0] = 250  # rojo puro

    back = CvImageDecoder().decode(encode_jpeg(img))

    assert back[8, 8, 0] > 200
    assert back[8, 8, 2] < 60
