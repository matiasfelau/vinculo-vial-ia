"""Clases del detector v2 y su correspondencia con la jerarquía de Open Images V7.

v2 detecta **presencia de un vehículo** (`car`, `truck`, `van`, `bus`), **patente** y siete **confusoras**
que no cuentan como presencia (avión, barco, bicicleta, moto, silla de ruedas, cartel de tránsito y reloj):
existen para que el modelo aprenda a no confundirlas con un auto o con una patente. No tiene rostro ni rueda.
El orden de `CLASSES` es el de las columnas de salida del ONNX y va en el `classes` del manifest: cambiarlo
obliga a reentrenar.

La jerarquía sale de `hierarchy.json` de Open Images V7 (anotada acá a mano, verificada contra ese archivo):

- `Car` tiene dos subclases, `Limousine` y `Van`. `Van` es su propia clase (`van`); `Limousine` es `car`.
- `Truck` y `Bus` no tienen subclases.
- `Taxi` y `Ambulance` cuelgan de `Land vehicle` (hermanas de `Car`, no hijas). Se mapean igual, porque son
  vehículos de calle que se ven como un auto o una van: si no, una foto con un taxi anotado solo como `Taxi`
  enseñaría que ese taxi no es un vehículo. `Taxi` → `car`, `Ambulance` → `van`.
- Qué cuenta como vehículo en el servicio (14/11/2026) lo decide el adaptador (`ROLES`), no este archivo:
  auto (con taxi), van y camión. `bus` es una salida del modelo que el adaptador no devuelve, así que el
  colectivo no cuenta. `Ambulance` se entrenó como `van` y sigue contando (para que no cuente habría que
  reentrenar). Open Images anota muchas pickups como `Truck`: el camión se mantiene completo.
- Confusoras: `Boat` (con `Barge`, `Gondola`, `Canoe`), `Airplane`, `Bicycle`, `Motorcycle`, `Wheelchair`,
  `Traffic sign` (con `Stop sign`) y `Clock` (con `Alarm clock`, `Digital clock`, `Wall clock`).
"""

from collections.abc import Mapping, Sequence

CLASSES: Sequence[str] = (
    "car",
    "truck",
    "van",
    "bus",
    "plate",
    "airplane",
    "boat",
    "bicycle",
    "motorcycle",
    "wheelchair",
    "traffic_sign",
    "clock",
)
PRESENCE: Sequence[str] = ("car", "truck", "van", "bus")
CONFOUNDERS: Sequence[str] = tuple(c for c in CLASSES if c not in (*PRESENCE, "plate"))

# Nombre de Open Images → clase de v2
OPEN_IMAGES_CLASSES: Mapping[str, str] = {
    "Car": "car",
    "Limousine": "car",
    "Taxi": "car",
    "Truck": "truck",
    "Van": "van",
    "Ambulance": "van",
    "Bus": "bus",
    "Vehicle registration plate": "plate",
    "Airplane": "airplane",
    "Boat": "boat",
    "Barge": "boat",
    "Gondola": "boat",
    "Canoe": "boat",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Wheelchair": "wheelchair",
    "Traffic sign": "traffic_sign",
    "Stop sign": "traffic_sign",
    "Clock": "clock",
    "Alarm clock": "clock",
    "Digital clock": "clock",
    "Wall clock": "clock",
}
# Una foto de fondo no puede tener nada de esto anotado: todo lo que cuelga de `Vehicle` (la clase y todos sus
# descendientes en la jerarquía), más las partes que delatan un vehículo.
VEHICLE_FAMILY: Sequence[str] = (
    "Vehicle",
    "Land vehicle",
    "Ambulance",
    "Cart",
    "Bicycle",
    "Bus",
    "Snowmobile",
    "Golf cart",
    "Motorcycle",
    "Segway",
    "Tank",
    "Train",
    "Truck",
    "Unicycle",
    "Car",
    "Limousine",
    "Van",
    "Taxi",
    "Wheelchair",
    "Watercraft",
    "Boat",
    "Barge",
    "Gondola",
    "Canoe",
    "Jet ski",
    "Submarine",
    "Aircraft",
    "Helicopter",
    "Airplane",
    "Rocket",
)
VEHICLE_PARTS: Sequence[str] = ("Wheel", "Tire", "Vehicle registration plate")
BACKGROUND_EXCLUDED: frozenset[str] = frozenset((*VEHICLE_FAMILY, *VEHICLE_PARTS, *OPEN_IMAGES_CLASSES))

# Cuántas fotos por etapa: (entrenamiento, validación). Cada etapa busca fotos donde aparece su clase;
# cada foto conserva todas sus cajas de las doce clases. Las etapas van en este orden y una foto no se repite.
STAGES: Mapping[str, tuple[Sequence[str], int, int]] = {
    "plates": (("Vehicle registration plate",), 500, 100),
    "cars": (("Car", "Limousine", "Taxi"), 400, 80),
    "trucks": (("Truck",), 150, 30),
    "vans": (("Van", "Ambulance"), 150, 30),
    "buses": (("Bus",), 150, 30),
    "airplanes": (("Airplane",), 70, 15),
    "boats": (("Boat", "Barge", "Gondola", "Canoe"), 70, 15),
    "bicycles": (("Bicycle",), 70, 15),
    "motorcycles": (("Motorcycle",), 70, 15),
    "wheelchairs": (("Wheelchair",), 70, 15),
    "traffic_signs": (("Traffic sign", "Stop sign"), 70, 15),
    "clocks": (("Clock", "Alarm clock", "Digital clock", "Wall clock"), 70, 15),
}
BACKGROUNDS = (300, 60)  # fotos de fondo: ninguna clase de v2 ni vehículo anotado
