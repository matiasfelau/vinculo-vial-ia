"""Sesiones de ONNX Runtime en CPU.

Solo se usa `CPUExecutionProvider`: ningún proveedor que haga red o escriba a disco. La sesión se
crea desde los bytes ya verificados contra el manifest, así lo que corre es exactamente lo que se
verificó.

Telemetría: los binarios oficiales de ONNX Runtime traen la telemetría 1DS **encendida**, que envía
eventos a Microsoft por HTTPS y guarda un identificador de dispositivo en disco. Acá se apaga con
`ORT_DISABLE_TELEMETRY=1` **antes** de importar `onnxruntime` (después ya es tarde: se inicializa al
importar) y, por las dudas, también con la API. Este es el único módulo que importa `onnxruntime`.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from collections.abc import Mapping, Sequence  # noqa: E402
from typing import Any, Protocol  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

ort.disable_telemetry_events()

CPU_PROVIDER = "CPUExecutionProvider"
ORT_LOG_ERROR = 3


class InferenceSessionLike(Protocol):
    """Lo que un adaptador necesita de una sesión; los tests la reemplazan por una falsa."""

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]: ...


class OnnxSession:
    """Envuelve `onnxruntime.InferenceSession` y expone los nombres de entrada y salida."""

    def __init__(self, model: bytes, intra_op_threads: int) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads  # 0 = lo decide ONNX Runtime
        options.inter_op_num_threads = 1
        options.log_severity_level = ORT_LOG_ERROR
        self._session = ort.InferenceSession(model, sess_options=options, providers=[CPU_PROVIDER])
        self.input_names: tuple[str, ...] = tuple(i.name for i in self._session.get_inputs())
        self.output_names: tuple[str, ...] = tuple(o.name for o in self._session.get_outputs())

    def input_shape(self, index: int = 0) -> tuple[int | str | None, ...]:
        """Forma declarada de una entrada (las dimensiones dinámicas vienen como texto o `None`)."""
        return tuple(self._session.get_inputs()[index].shape)

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        outputs = self._session.run(list(output_names) if output_names else None, dict(input_feed))
        return [np.asarray(output) for output in outputs]
