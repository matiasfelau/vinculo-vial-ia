"""OCR de patentes con PARSeq (preentrenado, sin reentrenar) sobre ONNX Runtime.

Preproceso como el del entrenamiento: el recorte RGB se escala a 32×128 (alto×ancho) y se normaliza
con media 0,5 y desvío 0,5. El ONNX (`tools/export/parseq.py`) devuelve los logits de los 26 pasos de
la decodificación greedy autorregresiva con una iteración de refinamiento: `(1, 26, 95)`, donde la
clase 0 es fin de texto y las 94 restantes son los caracteres de `CHARSET`. Acá se aplica softmax, se
toma el argmax de cada paso hasta el fin de texto y la confianza es el mínimo de las probabilidades de
los caracteres (sin contar el fin de texto).

Postproceso: si el texto contiene una patente argentina (Mercosur `AA123AA` o anterior `ABC123`),
se devuelve solo esa subcadena, tolerando las confusiones O/0, I/1, B/8... que maneja
`domain/plate.py`, y su confianza es la de sus propios caracteres. Si no, se devuelve el texto crudo.
Las patentes de dos líneas (motos) están fuera de alcance: el modelo lee una sola línea.
"""

import logging
import string
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.domain.model import Pixels
from vvi_ia.domain.plate import TO_DIGIT, TO_LETTER

logger = logging.getLogger(__name__)

MODEL_NAME = "parseq"
OUTPUT_NAMES = ("logits",)
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation
EOS_ID = 0
NUM_CLASSES = len(CHARSET) + 1  # el fin de texto va primero
IMAGE_NDIM = 3
RGB_CHANNELS = 3
LOGITS_NDIM = 3  # (lote, pasos, clases)
NCHW_NDIM = 4
MAX_PIXEL = 255.0
HALF = 0.5
WARMUP_SIZE = (64, 200)  # alto y ancho de la imagen gris de prueba

# Formatos argentinos como secuencias de letras (L) y dígitos (D)
MERCOSUR = "LLDDDLL"
OLD_FORMAT = "LLLDDD"
_LETTER_LIKE = frozenset(chr(code) for code in TO_LETTER)  # dígitos que el OCR confunde con letras
_DIGIT_LIKE = frozenset(chr(code) for code in TO_DIGIT)  # letras que el OCR confunde con dígitos


def preprocess(crop: Pixels, height: int, width: int) -> NDArray[np.float32]:
    """Tensor `(1, 3, height, width)` con los valores en [-1, 1]."""
    if crop.ndim != IMAGE_NDIM or crop.shape[2] != RGB_CHANNELS or min(crop.shape[:2]) < 1:
        raise ValueError("el recorte debe ser RGB de alto y ancho positivos")
    shrinks = crop.shape[0] * crop.shape[1] > height * width
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if shrinks else cv2.INTER_CUBIC)
    normalized = (resized.astype(np.float32) / MAX_PIXEL - HALF) / HALF
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def decode_greedy(logits: NDArray[Any]) -> tuple[str, list[float]]:
    """Texto y probabilidad de cada carácter, hasta el primer fin de texto."""
    if logits.ndim != LOGITS_NDIM or logits.shape[0] != 1 or logits.shape[1] < 1:
        raise ValueError("forma inesperada de logits")
    if logits.shape[2] != NUM_CLASSES:
        raise ValueError("forma inesperada de logits")
    if not np.isfinite(logits).all():
        raise ValueError("la salida del modelo no es finita")
    steps = logits[0].astype(np.float64)
    exp = np.exp(steps - steps.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    ids = probs.argmax(axis=-1)
    chars: list[str] = []
    char_probs: list[float] = []
    for step, token in enumerate(ids):
        if token == EOS_ID:
            break
        chars.append(CHARSET[token - 1])
        char_probs.append(float(probs[step, token]))
    return "".join(chars), char_probs


def _substitutions(window: str, pattern: str) -> int | None:
    """Cuántos caracteres de `window` hay que reinterpretar para que sea `pattern`; `None` si no entra."""
    cost = 0
    for char, kind in zip(window, pattern, strict=True):
        if kind == "L":
            if char.isalpha():
                continue
            if char not in _LETTER_LIKE:
                return None
        else:
            if char.isdigit():
                continue
            if char not in _DIGIT_LIKE:
                return None
        cost += 1
    return cost


def extract_plate(text: str, probs: Sequence[float]) -> tuple[str, float]:
    """Patente argentina dentro de `text` y su confianza; si no hay, el texto crudo y su confianza.

    Gana la ventana con menos caracteres reinterpretados más caracteres que quedan fuera de ella: así
    una ventana corta del formato anterior no le gana a una patente Mercosur con una confusión. Ante
    un empate, Mercosur antes que el formato anterior, y la más a la izquierda.
    """
    kept = [index for index, char in enumerate(text) if char.isascii() and char.isalnum()]
    canonical = "".join(text[index].upper() for index in kept)
    best: tuple[tuple[int, int, int], int, int] | None = None
    for pattern in (MERCOSUR, OLD_FORMAT):
        size = len(pattern)
        for start in range(len(canonical) - size + 1):
            cost = _substitutions(canonical[start : start + size], pattern)
            if cost is None:
                continue
            key = (cost + len(canonical) - size, 0 if pattern == MERCOSUR else 1, start)
            if best is None or key < best[0]:
                best = (key, start, size)
    if best is None:
        return text, min(probs, default=0.0)
    _, start, size = best
    return canonical[start : start + size], min(probs[index] for index in kept[start : start + size])


class ParseqPlateReader:
    """Implementa `PlateReader`. La sesión es segura entre hilos: corre en el pool de inferencia."""

    def __init__(
        self, session: InferenceSessionLike, input_name: str, input_size: tuple[int, int], version: str
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._input_size = input_size  # (alto, ancho) del modelo
        self.version = version

    def _decode(self, crop: Pixels) -> tuple[str, list[float]]:
        tensor = preprocess(crop, *self._input_size)
        (logits,) = self._session.run(OUTPUT_NAMES, {self._input_name: tensor})
        return decode_greedy(logits)

    def read(self, crop: Pixels) -> tuple[str, float] | None:
        try:
            return extract_plate(*self._decode(crop))
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar la patente o la imagen.
            logger.warning("falló la inferencia del OCR", extra={"error": type(exc).__name__})
            return None

    def raw_text(self, crop: Pixels) -> str | None:
        """Texto leído antes de extraer la patente. Es solo para diagnóstico en `tools/eval`."""
        try:
            return self._decode(crop)[0]
        except Exception as exc:
            logger.warning("falló la inferencia del OCR", extra={"error": type(exc).__name__})
            return None


@register("ocr", MODEL_NAME)
def build(ctx: ModelContext) -> ParseqPlateReader:
    session = ctx.model.session
    if session.output_names != OUTPUT_NAMES or len(session.input_names) != 1:
        raise ValueError("el modelo no tiene la interfaz de PARSeq")
    shape = session.input_shape()
    if len(shape) != NCHW_NDIM:
        raise ValueError("la entrada del modelo debe ser NCHW")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("la entrada del modelo debe tener alto y ancho fijos")
    reader = ParseqPlateReader(session, session.input_names[0], (height, width), ctx.model.version)
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo
    # esperado, el servicio no arranca.
    if reader.read(np.full((*WARMUP_SIZE, RGB_CHANNELS), 128, np.uint8)) is None:
        raise ValueError("el modelo no devolvió una salida válida")
    return reader
