"""Circuit breaker por dependencia (uno por destino; nunca se comparte entre dependencias)."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

from vvi_ia.service.ports import TransientError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    """Estados del circuito."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(TransientError):
    """Llamada rechazada de inmediato, sin tocar la red, porque el circuito está abierto."""


class CircuitBreaker:
    """Abre tras `failure_threshold` fallas dentro de `window_s`; sondea con `half_open_max_calls`.

    Seguro entre hilos (el pool de inferencia llama desde varios workers).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        window_s: float,
        open_duration_s: float,
        half_open_max_calls: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._threshold = failure_threshold
        self._window = window_s
        self._open_duration = open_duration_s
        self._max_probes = half_open_max_calls
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures: deque[float] = deque()
        self._opened_at = 0.0
        self._probes_in_flight = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        """Estado actual; un circuito abierto cuyo tiempo venció se informa como semiabierto."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._open_elapsed():
                return CircuitState.HALF_OPEN
            return self._state

    def call(self, operation: Callable[[], T], is_failure: Callable[[Exception], bool]) -> T:
        """Ejecuta `operation` bajo el breaker.

        Si el circuito está abierto lanza `CircuitOpenError` sin ejecutarla. Las excepciones se
        relanzan tal cual; cuentan como falla solo si `is_failure` lo indica.
        """
        probe = self._acquire()
        try:
            result = operation()
        except Exception as exc:
            self._release(probe, failed=is_failure(exc))
            raise
        self._release(probe, failed=False)
        return result

    def _open_elapsed(self) -> bool:
        return self._clock() - self._opened_at >= self._open_duration

    def _acquire(self) -> bool:
        """Devuelve True si la llamada es un sondeo del estado semiabierto."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if not self._open_elapsed():
                    raise CircuitOpenError(f"circuito abierto: {self._name}")
                self._transition(CircuitState.HALF_OPEN)
                self._probes_in_flight = 0
            if self._state == CircuitState.HALF_OPEN:
                if self._probes_in_flight >= self._max_probes:
                    raise CircuitOpenError(f"circuito semiabierto sin cupo de sondeo: {self._name}")
                self._probes_in_flight += 1
                return True
            return False

    def _release(self, probe: bool, failed: bool) -> None:
        with self._lock:
            now = self._clock()
            if probe:
                self._probes_in_flight = max(0, self._probes_in_flight - 1)
                if self._state != CircuitState.HALF_OPEN:
                    return
                if failed:
                    self._open(now)
                else:
                    self._failures.clear()
                    self._transition(CircuitState.CLOSED)
                return
            if not failed or self._state != CircuitState.CLOSED:
                return
            self._failures.append(now)
            while self._failures and now - self._failures[0] > self._window:
                self._failures.popleft()
            if len(self._failures) >= self._threshold:
                self._open(now)

    def _open(self, now: float) -> None:
        self._opened_at = now
        self._failures.clear()
        self._transition(CircuitState.OPEN)

    def _transition(self, new: CircuitState) -> None:
        old, self._state = self._state, new
        logger.warning(
            "circuit breaker cambió de estado",
            extra={"breaker": self._name, "from_state": old.value, "to_state": new.value},
        )
