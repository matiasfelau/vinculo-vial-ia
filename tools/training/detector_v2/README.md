# Detector v2: presencia de vehículo, patente y confusoras (sin rostro ni rueda)

El **detector v2** (`rfdetr_openimages_v2`) es **el detector por defecto del servicio** desde el 14/11/2026
(`MODEL_DETECTOR=rfdetr_openimages_v2` en `.env.example`) y reemplaza al v1 (`rfdetr_openimages`) en el puerto `detector`.
v1 queda en `models/manifest.json` con su SHA-256 como respaldo y se vuelve a él cambiando `MODEL_DETECTOR`; ya no se mide.
Los pesos no van a git.

| | v1 | v2 |
|---|---|---|
| Clases | `car`, `wheel`, `plate`, `face` | `car`, `truck`, `van`, `bus`, `plate` y siete confusoras |
| Puertos | `detector` y `face` | solo `detector` |
| Presencia de vehículo | `car` | `car` (con taxi), `truck` o `van` (el adaptador devuelve una caja `car`); `bus` no cuenta |
| Rostro, rueda | sí | no (decisión del 11/11/2026) |
| Confusoras | no | colectivo (salida propia, no se devuelve), avión, barco, bicicleta, moto, silla de ruedas, cartel de tránsito, reloj: no se devuelven |
| Fotos de fondo | no | sí: sin ninguna anotación |

Las confusoras no cuentan como presencia: existen para que el modelo aprenda que un avión, una moto o un reloj no
son un auto ni una patente. El orden de las columnas de salida, que fija `v2_classes.CLASSES`, va en el campo
`classes` del manifest y el adaptador lo toma de ahí (v1 no lo declara y usa el de su adaptador). El camino
`MODEL_FACE=rfdetr_openimages` queda **solo para v1**: `MODEL_FACE=rfdetr_openimages_v2` no arranca.

## Datos: solo Open Images V7

Ningún dato de COCO ni de otra fuente. (Los pesos de partida son los COCO que publica Roboflow para RF-DETR, igual que
en v1: el preentrenamiento no se cambia, lo que se ajusta es el detector.)

**Correspondencia con la jerarquía de Open Images V7** (verificada contra su `hierarchy.json`, `v2_classes.py`):

| Clase de v2 | Clases de Open Images |
|---|---|
| `car` | `Car`, `Limousine` (hija de `Car`) y `Taxi` |
| `van` | `Van` (hija de `Car`) y `Ambulance` |
| `truck`, `bus` | `Truck`, `Bus` (sin subclases) |
| `plate` | `Vehicle registration plate` |
| `boat` | `Boat`, `Barge`, `Gondola`, `Canoe` |
| `traffic_sign` | `Traffic sign`, `Stop sign` |
| `clock` | `Clock`, `Alarm clock`, `Digital clock`, `Wall clock` |
| `airplane`, `bicycle`, `motorcycle`, `wheelchair` | la clase del mismo nombre |

`Taxi` y `Ambulance` cuelgan de `Land vehicle` (son hermanas de `Car`, no hijas): se mapearon igual porque son
vehículos de calle; si no, una foto con un taxi anotado solo como `Taxi` enseñaría que ese taxi no es un vehículo.
Es una decisión mía; se cambia en `OPEN_IMAGES_CLASSES`.

**Qué cuenta como vehículo en el servicio (14/11/2026): auto (con taxi), van y camión.** Es el mapeo del adaptador
(`ROLES` en `rfdetr_detector.py`), no el del entrenamiento: `bus` es una salida propia del modelo y el adaptador no la
devuelve, así que un colectivo no cuenta. **Limitaciones:** (1) **la ambulancia se entrenó como `van`**, no es una
salida propia, así que **sigue contando**; que no cuente exige reentrenar (por ejemplo, mandarla a las fotos de fondo
o a una confusora). No se hizo. (2) **Open Images no distingue un flete de un camión pesado y anota muchas pickups como
`Truck`**, así que `truck` se mantiene completo: una pickup válida cuenta, y también un camión pesado que nadie
fotografiaría como un auto. Las confusoras se verificaron en `classes.csv` y en la
jerarquía: las siete existen como clases con cajas en Open Images V7.

**Qué se bajó** (`v2_download.py`, semilla 0): por etapa se eligen fotos donde aparece la clase y cada foto conserva
todas sus cajas de las doce clases; se descartan las cajas de grupo (`IsGroupOf`), de dibujos (`IsDepiction`), de
fotos tomadas desde adentro (`IsInside`) y de menos de 8 px.

| Etapa | Entrenamiento | Validación |
|---|---|---|
| patentes | 500 | 100 |
| autos | 400 | 80 |
| camiones, vans, colectivos | 150 c/u | 30 c/u |
| cada confusora (7) | 70 c/u | 15 c/u |
| **fondo** (sin ninguna clase) | 300 | 60 |

Resultado: 2 139 fotos de entrenamiento y 435 de validación. Cajas de entrenamiento: `car` 2 649, `truck` 300,
`van` 254, `bus` 355, `plate` 734, `airplane` 127, `boat` 208, `bicycle` 182, `motorcycle` 174, `wheelchair` 112,
`traffic_sign` 159, `clock` 119. Las fotos de fondo no tienen ninguna caja de las clases de v2 ni de la familia
`Vehicle` (ni ruedas, neumáticos o patentes); Open Images no es exhaustivo, así que alguna puede tener un auto sin
anotar (como pasó con 3 de las 1 200 negativas de la evaluación).

**Exclusiones.** Nunca entran: la **partición de prueba de v1** (204 fotos), las **negativas de la evaluación**
(1 700 fotos de `presence-negativas-nuevas`, que incluyen las 1 200 que se miden) y las **489 negativas de v1**: 2 326
identificadores. Se excluyen **antes** de elegir y se **verifican después**, sobre los archivos bajados
(`exclusion_check.json`, junto al dataset):

| Verificación | Resultado |
|---|---|
| Por identificador de Open Images (nombre del archivo) | 0 coincidencias sobre 2 574 fotos elegidas |
| Por SHA-256 del contenido (2 326 archivos excluidos comparados, ninguno faltaba) | 0 coincidencias |
| Por `OriginalMD5` de Open Images (atrapa la misma foto bajo otro identificador; conocido para las 2 574 y las 2 326) | 0 coincidencias |

Las particiones de Open Images (`train`, `validation`, `test`) son disjuntas por construcción: el entrenamiento sale
de `train` y la validación de `validation`, las exclusiones cubren además las fotos que v1 usó para evaluar.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
uv run --frozen --extra dev --extra tools python tools/training/detector_v2/v2_download.py --fresh-negatives D:\vv-eval\presence-negativas-nuevas
```

Hace falta que v1 haya bajado antes los CSV de anotaciones (`TRAIN_DATA_DIR\fiftyone-cache`, 4,4 GB). Las fotos se bajan
del bucket público de Open Images; `--scale 0.05` arma una prueba rápida.

## Entrenar y exportar (mismo procedimiento que v1)

```powershell
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/training/detector_v2/v2_train.py
uv run --frozen --extra dev --extra tools python tools/training/detector_v2/v2_export.py
```

RF-DETR Small a 512 px, partiendo de los pesos COCO de Roboflow, FP32, lote 2 con acumulación de 8, hasta 20 épocas con
parada temprana de 5 sobre la validación: **los mismos parámetros que v1** (`tools/training/common.py`). Las fotos de
fondo van en el COCO sin anotaciones. `v2_export.py` pide que el ONNX reproduzca a PyTorch (al menos el 95 % de las
consultas, en tres semillas) antes de guardarlo y muestra el SHA-256 que va en el manifest.

### Resultado de referencia (GTX 1660 de 6 GB)

| | |
|---|---|
| Duración | 117 minutos (6 997 s), 14 épocas; paró solo, la mejor fue la 8 |
| Memoria de la GPU | pico de 2,8 GB asignados (3,3 GB reservados) |
| Validación (mejor época, 435 fotos propias de v2) | mAP50 0,818 · mAP50:95 0,677 · recall 0,749 · precisión 0,801 |
| ONNX contra PyTorch | **100 %** de las consultas coinciden en el peor caso de tres semillas |
| Archivo | `rfdetr_openimages_v2.onnx`, 114,3 MB, SHA-256 `6bcad50b33c56c8f8a2dbb54a404688a34e7d68d6c19eba8063dfa9c36c3ee13` |

**El entrenamiento no es reproducible bit a bit**: otro entrenamiento da otros pesos y otro SHA-256; el archivo que vale
es el registrado en el manifest. La comparación con v1 sobre la prueba de v1 está en
[tools/eval/README.md](../../eval/README.md#detector-v2-contra-v1).

## Usarlo

Es el detector por defecto de `.env.example` (decisión del 14/11/2026):

```dotenv
MODEL_DETECTOR=rfdetr_openimages_v2
MODEL_FACE=scrfd_10g     # v2 no tiene rostros: los rostros los cubre otro modelo
MODELS_DIR=./models/weights
```

Para volver a v1 (respaldo, ya no se mide): `MODEL_DETECTOR=rfdetr_openimages`.
