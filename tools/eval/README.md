# Evaluar modelos

Evaluar un modelo es probarlo con ejemplos cuya respuesta correcta ya conocemos y contar cuántas
veces acierta. Sirve para decidir si un modelo está listo para activarse y para fijar los umbrales
del servicio.

Todo lo que se evalúa acá usa **fotos con patentes y caras de terceros**. Por eso:

- los datasets y los resultados viven **fuera del repo** (los scripts se niegan a leer o escribir
  dentro de él);
- la consola muestra solo cifras, nunca patentes;
- lo que contiene patentes (`audit.html`, `audit.csv` y `audit/`) queda en tu carpeta de salida:
  no lo subas ni lo compartas.

Antes de empezar hay que tener hecho "Antes de empezar" de [tools/export/README.md](../export/README.md)
(dependencias instaladas) y el modelo exportado en `MODELS_DIR`.

## OCR de patentes (`ocr.py`)

Mide qué tan bien lee `parseq` las patentes del dataset "patentes" de Roboflow
(`patentes-obs9r/patentes-n81go`, CC BY 4.0, 135 fotos).

### 1. Conseguir el dataset

`EVAL_DATA_DIR` es la carpeta raíz de todas las evaluaciones (por ejemplo `D:\vv-eval`). Cada una tiene
su subcarpeta. Para el OCR:

```text
D:\vv-eval\                      <- EVAL_DATA_DIR
  ocr-roboflow\                  <- el export COCO, tal como lo bajó Roboflow
    train\  valid\  test\
  ocr\
    label_overrides.csv          <- correcciones de etiquetas (opcional, ver más abajo)
```

1. Abrí <https://universe.roboflow.com/patentes-obs9r/patentes-n81go>.
2. **Download Dataset** → formato **COCO** → **Download zip to computer**.
3. Descomprimilo en `ocr-roboflow`, fuera del repo. Tienen que quedar las carpetas `train`, `valid`
   y `test`, cada una con su `_annotations.coco.json`. El export **no se modifica nunca**.

### 2. Correrlo

Elegí dónde se guardan los resultados (otra carpeta fuera del repo) y avisale a la terminal:

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval"
$env:EVAL_OUT = "D:\vv-eval\out\ocr"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/ocr.py
```

Tarda unos segundos. Las variables se pierden al cerrar la terminal. También se pueden pasar como
`--data-dir`, `--out-dir` y `--models-dir`; `--dataset-dir` y `--overrides` cambian dónde están el
dataset y el archivo de correcciones.

### 3. Qué hace el script

1. Lee las cajas de las patentes del COCO. La etiqueta de cada caja es el **nombre de su clase**, sin
   espacios y en mayúsculas.
2. Aplica las **correcciones de etiquetas** (ver abajo).
3. Descarta los **marcadores de posición**: etiquetas cuyos dígitos son todos ceros, como `AA000AA`.
4. Recorta cada caja de la foto igual que el servicio y se la pasa al OCR.
5. Compara la lectura con la etiqueta usando la misma función que el servicio (`plate_check`).
6. A cada lectura sin coincidencia exacta le asigna una **causa** (ver abajo).

### Correcciones de etiquetas

Si una etiqueta del dataset está mal, o la patente no se puede leer ni a ojo, no se edita el export:
se anota en `EVAL_DATA_DIR\ocr\label_overrides.csv`, que tiene tres columnas:

```csv
archivo,etiqueta,motivo
foto_1.jpg,AB123CD,etiqueta incorrecta en el dataset
foto_2.jpg,EXCLUIR,patente ilegible
```

- `archivo` es el nombre de la foto, tal como aparece en el export.
- `etiqueta` es el texto correcto de la patente o la palabra `EXCLUIR`.
- Vale para **todas las cajas de esa foto**.
- Las fotos excluidas se informan aparte como `ilegibles`: **no son una falla del modelo** y no
  cuentan en ninguna métrica.
- Si nombrás una foto que no existe, el script se detiene: casi seguro es un error de tipeo.
- Las lecturas con etiqueta corregida llevan `etiqueta_corregida = si` en la auditoría.

Este archivo contiene nombres de fotos del dataset: queda fuera del repo.

### 4. Cómo leer el resultado

La consola muestra algo así (los números dependen del dataset):

```text
Modelo parseq@7ff186b1f743 · 137 cajas · descartadas {'marcador de posición': 3}
Ilegibles (excluidas): 1 · etiquetas corregidas: 1
Lecturas: 133 · fallas del modelo: 0
Coincidencia exacta: 105 (78.9 %)
Distribución con OCR_MIN_CONF=0.6: {'PLATE_MATCH': 106, 'PLATE_FUZZY': 10, 'PLATE_MISMATCH': 3, 'PLATE_UNREADABLE': 14}
Propuesta (sin 4 lecturas de otros formatos): OCR_MIN_CONF=0.78 (75.2 % de lecturas aceptadas)
```

Debajo de eso imprime el **barrido de umbrales** (cuántas lecturas se aceptan con cada
`OCR_MIN_CONF` y cuántas de ellas son `MATCH`, `FUZZY` o `MISMATCH`) y la causa de los errores.

| Resultado | Qué significa en el servicio |
|---|---|
| `PLATE_MATCH` | La lectura coincide con la patente (permitiendo las confusiones típicas O/0, I/1, B/8…). |
| `PLATE_FUZZY` | Difiere en un solo carácter: alerta amarilla. |
| `PLATE_MISMATCH` | Difiere en más: **alerta roja de fraude**. Es el error caro. |
| `PLATE_UNREADABLE` | Lectura vacía o con confianza menor a `OCR_MIN_CONF`: alerta amarilla. |

*Coincidencia exacta* es más estricta que `MATCH`: no permite ninguna corrección por confusión.

En `EVAL_OUT` queda además `report.json`, con las mismas cifras desglosadas por split, por formato de
patente (Mercosur, anterior, otro) y un barrido de umbrales.

### 5. Causas de los errores

A cada lectura sin coincidencia exacta se le asigna **una causa principal**, la primera de esta lista que
se cumple, y se anotan **todos los factores** presentes (un recorte puede estar a la vez inclinado y
borroso). En `report.json`, `causas` tiene las cifras; los recortes solo están en `audit.html`.

| Causa | Cómo se detecta |
|---|---|
| `dos líneas (moto)` | La etiqueta no tiene formato Mercosur ni el anterior (en este dataset son motos). Fuera de alcance. |
| `lee la franja` | El texto que leyó el modelo, antes de quedarse con la patente, contiene fragmentos de `REPUBLICA`, `ARGENTINA` o `MERCOSUR`. |
| `recorte cortado` | Faltan 2 o más caracteres al principio o al final, o la caja toca el borde de la foto y falta alguno. |
| `ángulo` | Los caracteres forman una línea inclinada 10° o más. |
| `desenfoque o baja resolución` | El recorte mide menos de 100×45 px, o es poco nítido. |
| `reflejo` | Recorte claro y casi sin contraste (lavado por un reflejo o sobreexposición). |
| `etiqueta dudosa` | El modelo leyó con confianza 0,90 o más una patente válida que difiere en 2 o más caracteres de la etiqueta. |
| `confusión de caracteres` | Misma longitud y cada diferencia es entre caracteres parecidos (O/0, I/1, B/8, W/M, 0/G...). |
| `otras` | Ninguna de las anteriores: hay que mirarlas a ojo (suciedad, desgaste, algo tapando la patente...). |

Son reglas automáticas y aproximadas. Sirven para ver **dónde** están los errores, no para
diagnosticar una foto en particular: para eso está la auditoría. Dos aclaraciones:

- `confusión de caracteres` incluye pares que el servicio **no** corrige hoy (como `0` leído `G`);
  `plate_check` solo corrige las confusiones entre letra y dígito según la posición.
- `etiqueta dudosa` no prueba que la etiqueta esté mal: un modelo también puede equivocarse con mucha
  confianza. Por eso la corrección de una etiqueta la decide una persona, mirando el recorte.

### 6. Auditar a ojo

Abrí `audit.html` en el navegador. Muestra cada recorte con su etiqueta, la lectura, la confianza y el
resultado. Incluye **todo lo que no fue un acierto exacto** y una muestra al azar de aciertos
(`--match-samples`, 20 por defecto). Sirve para dos cosas:

- ver *por qué* falla el modelo (foto borrosa, patente cortada, moto de dos líneas…);
- detectar **etiquetas mal cargadas** en el dataset: si el modelo lee bien y la etiqueta está mal, el
  error es del dataset, no del modelo.

### 7. El umbral `OCR_MIN_CONF`

El servicio acepta una lectura solo si su confianza es mayor o igual a `OCR_MIN_CONF`. Si es más baja,
la patente queda como ilegible (amarillo). El riesgo está en el otro lado: una lectura equivocada con
confianza alta termina en `PLATE_MISMATCH` y marca como fraude una patente que era correcta.

Por eso la propuesta es el umbral más bajo que deja **por debajo** a todos los `MISMATCH` del
dataset: la confianza del peor `MISMATCH`, más 0,01, redondeada hacia arriba, entre 0,50 y 0,99. Sin
ningún `MISMATCH`, propone 0,50.

Tomala como orientativa:

- con 134 lecturas, un solo caso raro mueve mucho el resultado: mirá el barrido en `report.json` y los
  recortes del `MISMATCH` en la auditoría antes de cambiar nada;
- la propuesta **no incluye las lecturas de otros formatos**, que en este dataset son motos de dos
  líneas (fuera de alcance). El reporte muestra también qué saldría con todas;
- para aplicarla, poné `OCR_MIN_CONF=<valor>` en el `.env` del servicio.

### Opciones

| Opción | Para qué |
|---|---|
| `--model NOMBRE` | Otro modelo del puerto `ocr` del manifest (por defecto `parseq`). |
| `--splits train valid test` | Solo algunos splits (por defecto, los que existan). |
| `--min-conf 0.7` | Umbral con el que se arma la distribución (por defecto `OCR_MIN_CONF` del entorno o 0,78). |
| `--match-samples N` | Aciertos a incluir en la auditoría. |
| `--seed N` | Semilla de esa muestra al azar. |

### Problemas frecuentes

- **"indicá --data-dir o definí su variable de entorno"**: te faltó definir `EVAL_DATA_DIR`, `EVAL_OUT`
  o `MODELS_DIR` en esta terminal.
- **"no puede estar dentro del repositorio"**: usá una carpeta fuera del repo.
- **"falta train/_annotations.coco.json"**: el zip quedó dentro de otra carpeta; subí `train`, `valid` y
  `test` un nivel, o indicá la carpeta con `--dataset-dir`.
- **"las correcciones nombran fotos que no están en el dataset"**: un nombre de `label_overrides.csv`
  está mal escrito. El script te dice cuál.
- **"No se pudo empezar: ... sha256 ..."**: el archivo de `MODELS_DIR` no es el del manifest; ver
  [tools/export/README.md](../export/README.md).

## Re-identificación de autos (`reid.py`)

El servicio decide si las dos fotos de una alerta (la general y la de la patente) muestran **el mismo
auto** comparando sus *embeddings*: un embedding es una lista de números que describe cómo se ve el auto,
y dos fotos del mismo auto deberían dar listas parecidas. La **similitud** (coseno) va de -1 a 1 y se
compara con dos umbrales:

- por debajo de `REID_T_LOW`: autos distintos → alerta roja de fraude (`BURST_MISMATCH`);
- desde `REID_T_HIGH`: el mismo auto;
- entre los dos: no se sabe → alerta amarilla (`BURST_UNCERTAIN`).

Este script mide las similitudes reales con **tus fotos** y propone los dos umbrales. Los valores de
fábrica (0,50 y 0,70) son una estimación sin datos.

> Estado: el script está escrito y probado con datos sintéticos. Todavía no se corrió con fotos reales.

### 1. Armar el dataset

Fotos de autos con rostros y patentes de terceros: **fuera del repo**. En `EVAL_DATA_DIR` (por ejemplo
`D:\vv-eval`):

```text
D:\vv-eval\reid\
  fotos.csv
  fotos\          <- las fotos
```

`fotos.csv` tiene dos columnas. `auto` es un nombre cualquiera, el mismo para todas las fotos del mismo
auto físico:

```csv
archivo,auto
a1.jpg,auto-1
a2.jpg,auto-1
b1.jpg,auto-2
b2.jpg,auto-2
```

Conviene que cada auto tenga **al menos 2 fotos** (de ángulos y luces distintos, como llegarán en las
alertas) y que haya **varios autos del mismo modelo y color**: son las comparaciones difíciles. Cuantos
más autos, mejor se estima la cola de la distribución de los autos distintos.

### 2. Correrlo

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval"
$env:EVAL_OUT = "D:\vv-eval\out\reid"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/reid.py
```

Hace falta tener exportados `rfdetr_seg_coco` y `dinov2_vits14` (ver
[tools/export/README.md](../export/README.md)). Para cada foto corre lo mismo que el servicio: segmenta los
autos, se queda con el más grande y calcula su embedding. Las fotos sin auto o ilegibles se descartan y se
cuentan.

### 3. Cómo leer el resultado

Muestra, para los pares del **mismo auto** y para los de **autos distintos**, la media, el desvío y los
percentiles de la similitud, un histograma de texto y el **AUC**: la probabilidad de que un par del mismo
auto se parezca más que uno de autos distintos. 1,0 es separación perfecta; 0,5, azar.

Después propone los umbrales con dos límites de error, que se pueden cambiar:

| Opción | Qué limita | Por defecto |
|---|---|---|
| `--max-false-fraud` | Pares del **mismo auto** que quedarían por debajo de `REID_T_LOW`: una alerta roja de fraude sobre un reporte honesto. | 1 % |
| `--max-false-accept` | Pares de **autos distintos** que llegarían a `REID_T_HIGH`: un fraude que pasa por el mismo auto. | 1 % |

Los límites se cumplen siempre. Lo que cambia con la calidad de los datos es el ancho de la banda
`REID_T_LOW`–`REID_T_HIGH`: si las distribuciones se separan, la banda cae en el hueco y casi ningún par
queda en amarillo; si se solapan, la banda se ensancha y más pares van a revisión. El reporte dice si se
separan, cuánto queda en cada categoría con los umbrales propuestos y con los actuales.

`reid_report.json` tiene las mismas cifras y el histograma; no tiene nombres de fotos.

### 4. Aplicar la propuesta

Poné `REID_T_LOW` y `REID_T_HIGH` en el `.env` del servicio. Con pocos autos, los percentiles extremos se
apoyan en muy pocos pares: tomá la propuesta como orientativa y mirá el histograma.

### Problemas frecuentes (re-ID)

- **"necesita las columnas archivo, auto"** o **"fila N"**: el CSV tiene un error; el mensaje dice la fila.
- **"Sin propuesta: hacen falta pares iguales y pares distintos"**: hay que tener algún auto con 2 o más
  fotos y al menos 2 autos.
- **Muchas fotos descartadas por "sin auto"**: el segmentador no encontró un auto; probá bajar
  `--segmenter-min-conf`.

## Detector de autos, ruedas, patentes y rostros (`detector.py`)

Mide qué tan bien encuentra el detector (`rfdetr_openimages`) cada clase en fotos que **nunca vio al
entrenar**: el split `test` del subconjunto de Open Images que baja
[tools/training/detector/download.py](../training/detector/README.md). Corre el adaptador real, el mismo
código y el mismo ONNX que el servicio.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\detector"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/detector.py --split test
```

Por cada clase muestra:

| Columna | Qué significa |
|---|---|
| `reales` | Cajas anotadas en las fotos. |
| `umbral` | Confianza mínima con la que el servicio acepta esa clase (`DETECTOR_MIN_CONF` para autos y ruedas, 0,20 para patentes y 0,30 para rostros: son los de `PLATE_REDACT_MIN_CONF` y `FACE_MIN_CONF`). |
| `recall` | De las cajas reales, qué fracción encontró (IoU 0,5 o más) con ese umbral. |
| `prec.` | De las cajas que dibujó, qué fracción acertó. |
| `FP` | Falsos positivos: cajas que no corresponden a nada. |
| `chicas` y `rec.ch.` | Cajas con un lado menor a 32 px y el recall sobre ellas. |
| `AP50` | Calidad promedio sin depender del umbral: 1,0 es perfecto. |

Para tapar rostros y patentes manda el **recall**: una caja de más tapa de más, una de menos deja un
rostro a la vista. La precisión importa menos. La salida completa queda en `detector_report.json`, solo
con cifras. `--detector-min-conf`, `--plate-min-conf` y `--face-min-conf` cambian los umbrales.

### Mosaico para patentes y rostros chicos

`--plate-tiling` y `--face-tiling` (los mismos `PLATE_TILING` y `FACE_TILING` del servicio) recorren la foto
también en tiles de `--tile-px` (640 por defecto) con solape `--overlap` (0,25), y fusionan las cajas con NMS.
Las fotos de más de 1600 px se achican a ese lado antes de cortar los tiles, para acotar el costo. Solo mejora
los objetos chicos y cuesta **una inferencia más por tile**. Con el split de prueba de Open Images (204 fotos de
unos 1000 px, CPU de 6 núcleos):

| Patentes | Recall | Recall en chicas (< 32 px) | AP50 | Falsos positivos | ms por foto |
|---|---|---|---|---|---|
| sin mosaico | 0,83 | 0,61 | 0,787 | 40 | 288 |
| `--plate-tiling` (640 px) | 0,90 | 0,76 | 0,831 | 71 | 1437 |
| `--plate-tiling --tile-px 480` | 0,90 | 0,76 | 0,825 | 79 | 2082 |

Con tiles de 640 px el recall de patentes sube 7 puntos y el de las chicas 15, a cambio de unos 1,1 s más por
foto y más cajas de más (que, al tapar, cuestan poco). Los tiles de 480 px no mejoran y son más lentos. Autos,
ruedas y rostros no cambian: el mosaico solo toca las clases pedidas.

## Detectores de rostros (`face.py`)

El servicio tapa los rostros de las fotos con un relleno opaco antes de guardarlas. Si un rostro no se
detecta, **queda a la vista**: por eso lo que importa es el **recall** (qué fracción de las caras se encuentra),
más que la precisión. Este script compara detectores sobre fotos con las caras anotadas.

| Detector | Licencia de los pesos (`commercial`) | ¿Puede entrar al servicio? |
|---|---|---|
| RF-DETR (clase *Human face* de Open Images) | `yes` (Apache 2.0, datos CC BY 4.0) | Sí. |
| SCRFD-10G (InsightFace) | `escalable` (InsightFace vende licencias comerciales) | Sí, `scrfd_10g`. |
| RetinaFace-R50 (biubug6) | `no` (WIDER FACE, CC BY-NC-ND) | Solo por la excepción del puerto `face`. |
| YuNet | `no` (WIDER FACE, CC BY-NC-ND) | Solo por la excepción del puerto `face`. |

El puerto `face` es el único que puede seleccionar un modelo `commercial = no`, si es el que mejor protege la
privacidad (el registro lo deja pasar y avisa en el log). Los que no son RF-DETR ni SCRFD **no se incluyen en
el repo**: los cargás vos desde un ONNX tuyo.

### 1. Armar las fotos

Hay dos conjuntos de referencia, fuera del repo (tienen rostros de terceros):

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"      # ahí está el subset de Open Images del detector
$env:EVAL_DATA_DIR = "D:\vv-eval"
uv run --frozen --extra dev --extra tools python tools/eval/face_sets.py wider        # 1000 fotos al azar de WIDER FACE (validación)
uv run --frozen --extra dev --extra tools python tools/eval/face_sets.py openimages   # el split de prueba de Open Images, solo rostros
```

- **WIDER FACE (validación)**: referencia académica. Se baja de Hugging Face y se usa una muestra al azar
  (semilla 0). Se descartan las caras que WIDER marca como inválidas y las de menos de 8 px. **Nunca para
  entrenar.** Es un benchmark muy duro (de las 11 896 caras, 8 611 miden menos de 32 px) y muchas caras chicas
  no están anotadas, así que sus falsos positivos por foto están inflados.
- **Open Images (prueba)**: fotos que RF-DETR nunca vio. Se parece más a las fotos del servicio, pero tiene
  solo 135 caras (13 chicas), así que sus recalls por tamaño son ruidosos.

También sirven fotos propias con las caras anotadas en CVAT (una sola etiqueta de tipo *Rectangle*, exportada en
formato "COCO 1.0") en `EVAL_DATA_DIR\face\`: es lo que mejor predice cómo va a andar el servicio.

### 2. Conseguir los modelos

- **YuNet**: `face_detection_yunet_2023mar.onnx` del repositorio `opencv_zoo` (con Git LFS).
- **SCRFD**: `det_10g.onnx` del paquete `buffalo_l` de InsightFace (el mismo que registra el manifest como
  `scrfd_10g`; también sirve `scrfd_10g_bnkps.onnx`).
- **RetinaFace-R50**: un ONNX de entrada cuadrada fija 640 exportado con `convert_to_onnx.py` del repositorio de
  biubug6 (`--network resnet50 --long_side 640`).
- **RF-DETR**: `rfdetr_openimages` ya exportado en `MODELS_DIR` (ver
  [tools/training/detector/README.md](../training/detector/README.md)).

### 3. Correrlo

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\face"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/face.py --dataset-dir D:\vv-eval\face-wider --rfdetr-model rfdetr_openimages --scrfd-model D:\modelos\det_10g.onnx --retinaface-model D:\modelos\retinaface_r50.onnx --yunet-model D:\modelos\face_detection_yunet_2023mar.onnx --tiling rfdetr scrfd retinaface_r50
```

Pasá solo los que tengas. `--size` es el lado de la entrada de SCRFD y RetinaFace (640 por defecto).
`--tiling NOMBRE...` suma una versión `NOMBRE+mosaico` de esos detectores (`--tile-px`, `--overlap`).
Esa versión no aplica el tope de 1600 px del servicio (corta la foto tal cual está), así que es el peor caso de
latencia.
`--service-model scrfd_10g` corre un modelo del manifest a través del **adaptador del servicio** (y
`--service-tiling` suma el adaptador con `FACE_TILING=true`): da lo mismo que `--scrfd-model` y sirve para
comprobar el adaptador con los pesos reales.

### 4. Cómo leer el resultado

Por detector muestra el **recall con IoU 0,5 o más** a distintos topes de falsos positivos por foto (0,1, 0,5 y
1,0: qué tanto se encuentra si solo se tolera esa cantidad de cajas de más), el **AP50** (calidad sin depender
del umbral) y cuánto tarda en CPU. Todos los detectores se comparan en el mismo punto de falsos positivos,
porque cada uno tiene su propia escala de confianza. Después, por cada tope, el **recall por tamaño de cara**
(lado menor: chicas < 32 px, medianas de 32 a 96 px, grandes > 96 px), los falsos positivos reales y la latencia
(mediana y p95). `face_report.json` tiene además la curva completa y el umbral de cada punto.

### 5. Resultados de referencia

Recall con a lo sumo **0,5 falsos positivos por foto**; latencia por foto en una CPU de 6 núcleos
(AMD Ryzen 5 3600; la máquina no estaba del todo libre, tomalas como orden de magnitud).

**WIDER FACE, 998 fotos de validación y 11 896 caras** (8 611 chicas, 2 701 medianas, 584 grandes):

| Detector | AP50 | Recall | Chicas | Medianas | Grandes | ms (mediana / p95) |
|---|---|---|---|---|---|---|
| RF-DETR | 0,47 | 0,33 | 0,14 | 0,77 | 0,96 | 304 / 410 |
| **SCRFD-10G** | **0,80** | **0,66** | **0,55** | **0,92** | **0,97** | **118 / 200** |
| RetinaFace-R50 | 0,63 | 0,51 | 0,35 | 0,90 | 0,97 | 317 / 410 |
| YuNet | 0,75 | 0,57 | 0,46 | 0,83 | 0,90 | 64 / 127 |
| RF-DETR + mosaico | 0,58 | 0,39 | 0,21 | 0,83 | 0,96 | 1239 / 2266 |
| SCRFD-10G + mosaico | 0,87 | 0,69 | 0,61 | 0,91 | 0,96 | 762 / 7421 |
| RetinaFace-R50 + mosaico | 0,82 | 0,65 | 0,54 | 0,92 | 0,98 | 1561 / 2471 |

**Open Images (prueba), 204 fotos y 135 caras** (13 chicas, 57 medianas, 65 grandes):

| Detector | AP50 | Recall | Chicas | Medianas | Grandes | ms (mediana) |
|---|---|---|---|---|---|---|
| RF-DETR | 0,83 | 0,87 | 0,46 | 0,91 | 0,91 | 294 |
| SCRFD-10G | 0,73 | 0,86 | 0,62 | 0,89 | 0,88 | 105 |
| RetinaFace-R50 | 0,82 | 0,86 | 0,38 | 0,89 | 0,92 | 392 |
| YuNet | 0,72 | 0,82 | 0,69 | 0,81 | 0,86 | 72 |
| RF-DETR + mosaico | 0,85 | 0,90 | 0,38 | 0,95 | 0,95 | 1589 |
| SCRFD-10G + mosaico | 0,68 | 0,86 | 0,62 | 0,89 | 0,88 | 563 |

Cómo leerlo:

- **Las caras chicas son el punto débil de RF-DETR**: en WIDER encuentra 14 de cada 100 (SCRFD, 55). Se entrenó
  con Open Images, que anota poco las caras chicas, y achica la foto a 512 px.
- **SCRFD-10G** gana en WIDER por mucho y en Open Images queda empatado con RF-DETR a 0,5 falsos positivos por
  foto, siendo **unas tres veces más rápido** (105 contra 294 ms). Ojo: SCRFD y RetinaFace se entrenaron con WIDER
  FACE, así que ahí juegan de local; Open Images es el árbitro más neutral y no los contradice.
- Con el tope de **0,1** falsos positivos por foto SCRFD parece peor en Open Images (0,50 contra 0,79 de recall):
  es la forma de su confianza (pone alta confianza a caras que Open Images no anotó). En un servicio que tapa de
  más antes que de menos, importa más el punto de 0,5 a 1,0.
- **El mosaico ayuda a las caras chicas con SCRFD en WIDER** (de 0,55 a 0,61 con 0,5 falsos positivos y de 0,34 a
  0,45 con 0,1) pero **no en Open Images**, donde las caras no son tan chicas, y cuesta 5 a 6 veces la latencia
  (con picos de varios segundos en fotos grandes). Con RF-DETR ayuda poco: en WIDER las chicas pasan de 0,14 a 0,21.
- Las cifras de Open Images se apoyan en pocas caras chicas: confirmalas con fotos propias antes de decidir.

El criterio de [docs/modelos.md](../../docs/modelos.md) (que RF-DETR quede cerca de RetinaFace-R50) dio
`rfdetr_queda: false` en WIDER (RF-DETR pierde 14 puntos a 0,1 falsos positivos) y `true` en Open Images.

### 5b. Validación de `scrfd_10g` con el adaptador del servicio

WIDER FACE val **completo** (`face_sets.py wider --sample 4000`: 3220 fotos con caras válidas y 32 596 caras de al
menos 8 px; 23 000 chicas, 7 753 medianas y 1 843 grandes) y Open Images (prueba), con `--service-model scrfd_10g
--service-tiling` y el barrido de `FACE_MIN_CONF` (`--sweep-from 0.20 --sweep-to 0.50`). Recall con IoU 0,5 o más:

| `FACE_MIN_CONF` | WIDER: recall | chicas | medianas | grandes | falsos positivos por foto | Open Images: recall | falsos positivos por foto |
|---|---|---|---|---|---|---|---|
| 0,20 | 0,809 | 0,746 | 0,956 | 0,983 | 4,48 | 0,867 | 1,18 |
| 0,25 | 0,787 | 0,716 | 0,951 | 0,983 | 2,67 | 0,867 | 0,81 |
| **0,30** | 0,760 | 0,680 | 0,947 | 0,981 | 1,66 | 0,859 | 0,61 |
| 0,35 | 0,728 | 0,637 | 0,941 | 0,978 | 1,03 | 0,859 | 0,49 |
| 0,40 | 0,693 | 0,590 | 0,930 | 0,974 | 0,64 | 0,859 | 0,37 |
| 0,50 | 0,615 | 0,486 | 0,914 | 0,967 | 0,25 | 0,852 | 0,29 |

Con `FACE_TILING=true` (tiles de 640 px):

| `FACE_MIN_CONF` | WIDER: recall | chicas | medianas | grandes | falsos positivos por foto | Open Images: recall | chicas (13) |
|---|---|---|---|---|---|---|---|
| 0,25 | 0,885 | 0,849 | 0,966 | 0,986 | 5,93 | 0,882 | 0,769 |
| **0,30** | 0,869 | 0,829 | 0,961 | 0,984 | 3,88 | 0,874 | 0,769 |
| 0,40 | 0,830 | 0,777 | 0,949 | 0,979 | 1,78 | 0,859 | 0,615 |

Latencia por foto en CPU (Ryzen 5 3600): 105 ms en Open Images y 164 ms (p95 262) en WIDER sin mosaico;
575 ms (p95 1,2 s) y 1 084 ms (p95 7,0 s) con mosaico. Las caras chicas son el punto débil (0,68 en WIDER y 0,62 en
Open Images con 0,30), y el mosaico las sube unos 15 puntos a cambio de 5 a 6 veces la latencia y del doble de
falsos positivos (que al tapar cuestan poco). Los falsos positivos de WIDER están inflados por caras sin anotar.

### 6. Configuración de redacción (provisoria)

Con estos datos quedó como configuración de `.env.example` (y `PLATE_TILING=true` como default del código),
**provisoria hasta validar con fotos propias**. `REDACTION_REQUIRED` no se tocó (sigue en `false` solo en local):

```dotenv
MODEL_DETECTOR=rfdetr_openimages   # autos, ruedas y patentes
MODEL_FACE=scrfd_10g               # rostros: mejor recall, sobre todo en caras chicas, y 3 veces más rápido
FACE_MIN_CONF=0.30                 # con SCRFD son unos 0,8 falsos positivos por foto en Open Images
PLATE_REDACT_MIN_CONF=0.20
FACE_TILING=false                  # sin evidencia de mejora en fotos parecidas a las del servicio; cuesta 5 veces la latencia
PLATE_TILING=true                  # +7 puntos de recall de patentes; cuesta ~3,5 s más por alerta de 3 fotos (ver la tabla de la cascada)
```

Con `MODEL_FACE=scrfd_10g` el detector de rostros ya no comparte la inferencia con el de patentes (RF-DETR
resuelve los dos con una sola pasada): son unos 105 ms más por foto. La confianza de SCRFD no es la de
RF-DETR: `FACE_MIN_CONF` se eligió mirando la curva de SCRFD, no copiando el valor viejo.

## Daños de la carrocería (`damage.py`)

Mide qué tan bien el detector de daños (`rfdetr_cardd`) encuentra abolladuras, rayones, grietas, vidrios
rotos, ópticas rotas y neumáticos pinchados en las fotos de **prueba** de CarDD (374 fotos que el modelo nunca
vio al entrenar). Corre el adaptador real, el mismo código y el mismo ONNX que el servicio.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\damage"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/damage.py --split test
```

Se lee igual que la evaluación del detector de vehículos (`reales`, `recall`, `prec.`, `FP`, `chicas`,
`rec.ch.` y `AP50`; ver arriba), con el umbral `DAMAGE_T_LOW` (0,969 por defecto, provisorio; o `--damage-min-conf`).
El AP50 se calcula con el umbral casi en cero, así que no depende de él.

Para el servicio lo que importa es **si hay algún daño**, no cuál: la foto de detalle pasa si el detector
encuentra al menos una caja con confianza suficiente. Por eso, además de la tabla por clase, conviene mirar el
recall de las clases más frecuentes (abolladuras y rayones) y cuántos falsos positivos por foto hay.

CarDD no se redistribuye: el dataset, la salida y los pesos entrenados quedan en tu máquina, fuera del repo.

### Falsos positivos de daño (`damage_fp.py`)

Las fotos de CarDD tienen todas algún daño, así que `damage.py` no dice qué pasa con un auto **sin** daño.
`damage_fp.py` lo mide con fotos de autos del subset de Open Images (el de `detector`) y lo cruza con el
recall por foto de CarDD, para elegir `DAMAGE_MIN_CONF`:

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\damage"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/damage_fp.py select   # copia las fotos con un auto grande
uv run --frozen --extra dev --extra tools python tools/eval/damage_fp.py sheet    # hoja al azar para revisar a ojo
uv run --frozen --extra dev --extra tools python tools/eval/damage_fp.py run      # barrido de umbrales
```

1. `select` elige las fotos donde un auto ocupa al menos el 30 % del cuadro (`--min-car-fraction`) y las copia a
   `<TRAIN_DATA_DIR>\damage-fp\images`.
2. `sheet` arma `damage_fp_sheet.jpg` con fotos al azar. **Miralas**: si alguna tiene daño evidente (o no es un
   auto), agregá su nombre, uno por línea, a `<TRAIN_DATA_DIR>\damage-fp\excluidas.txt`.
3. `run` barre el umbral de 0,1 a 0,9. Por cada uno muestra qué fracción de las fotos sin daño tiene alguna caja
   (**falso positivo por foto**) y qué fracción de las fotos con daño de CarDD la tiene (**recall por foto**), y
   propone el umbral que más separa las dos (recall menos falsos positivos). También desglosa por clase y
   prueba un umbral distinto por clase. `damage_fp_top.jpg` tiene las fotos sin daño con más confianza y
   `damage_fp_flagged.jpg` un sorteo de las que superan `--review-conf`: miralas para ver si son falsos
   positivos de verdad o daño que se pasó por alto al revisar.

Es una propuesta: el default no cambia hasta que se apruebe.

#### Calibración de `DAMAGE_T_LOW` y `DAMAGE_T_HIGH`

El mismo comando calcula los dos umbrales con las reglas de decisión: `T_LOW` queda **justo sobre la mayor
confianza de una foto de auto sin daño** (que ninguna negativa llegue a amarillo ni a verde, regla A), y se cuenta
cuántas fotos con daño de CarDD quedan bajo `T_LOW` (válidos que terminarían en rojo, regla C). `T_HIGH` no se
puede calibrar sin un set de casos dudosos y queda igual a `T_LOW`. Con 414 negativas (tras revisar a mano las 24
con más confianza y excluir 2 con daño) `T_LOW` = 0,969 y 300 de las 374 positivas (80 %) quedan en rojo: **no
cumple C**. Las fotos de Open Images son de autos enteros, no de
detalles; con fotos propias de detalle sin daño la tasa puede ser distinta.

## Latencia de la cascada completa (`cascade.py`)

Mide cuánto tarda **una alerta entera** en CPU con los modelos reales: el mismo caso de uso, los mismos
adaptadores y el mismo ONNX que el servicio, de a una alerta por vez. Se arma con autos de Open Images (la
foto GENERAL y la PLATE son la misma, con la patente que lee el OCR como patente reportada, para que la
cascada llegue a todas las etapas en lugar de cortarse en un RED) y daños de CarDD.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\cascade"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/damage_fp.py select   # una vez: junta las fotos de autos
uv run --frozen --extra dev --extra tools python tools/eval/cascade.py --curb-proxy
```

| Opción | Para qué |
|---|---|
| `--alerts N` | Alertas por escenario (20 por defecto), más una de calentamiento que no cuenta. |
| `--curb-proxy` | Mientras no haya modelo de rebajes, simula su costo corriendo el segmentador de autos (misma arquitectura). Sin esto `curb` es `null` y los escenarios de rebaje cortan antes. |
| `--no-redaction` | `REDACTION_REQUIRED=false`, como en local. |
| `--plate-tiling on\|off` | `PLATE_TILING` (por defecto `on`, como el servicio). |
| `--long-side N` | Reescala las fotos a ese lado mayor en px (4000 simula una foto de celular de 4000 × 3000). Las copias quedan en `EVAL_OUT\cascade-photos`. |
| `--threads N` | Hilos de ONNX Runtime (0: los que elija). |

Mide tres escenarios (CRASHED de 3 fotos, CRASHED de 5 y PARKED_ON_GARAGE de 3) y muestra la mediana, el p95 y el
máximo de cada alerta, y la mediana de cada etapa (`load`, `vehicle`, `burst`, `plate`, `context`, `redaction`).

### Resultados de referencia

Medianas por alerta en un AMD Ryzen 5 3600 (6 núcleos), con los modelos de la configuración de redacción
(`rfdetr_openimages` para autos y patentes, `scrfd_10g` para rostros, redacción activa, rebajes simulados),
una alerta por vez. Con `PLATE_TILING=true` el servicio recorre cada foto también en tiles de 640 px, y
achica a 1600 px las fotos más grandes antes de cortarlos.

| Fotos | Mosaico de patentes | CRASHED de 3 fotos | CRASHED de 5 fotos | PARKED_ON_GARAGE de 3 fotos |
|---|---|---|---|---|
| ~1000 px (Open Images, CarDD) | no | 2,8 s (p95 3,0) | 4,2 s (p95 4,5) | 3,4 s (p95 3,9) |
| ~1000 px | **sí** | 6,3 s (p95 6,5) | 10,3 s (p95 10,9) | 6,8 s (p95 7,2) |
| 4000 × 3000 px | no | 3,2 s (p95 3,9) | 4,9 s (p95 5,5) | 4,2 s (p95 5,0) |
| 4000 × 3000 px | **sí** | **10,8 s** (p95 11,6) | **16,8 s** (p95 17,8) | **11,3 s** (p95 14,0) |

Con el mosaico de patentes la etapa `vehicle` (una detección por foto) pasa de unos 300 ms a 1,5 s con fotos de
1000 px y a 3,1 s con fotos de 4000 × 3000 (10 inferencias por foto: la completa y 9 tiles sobre la foto achicada
a 1600 px); `plate` hace lo mismo. Las fotos de 4000 × 3000 son reescalados de las de Open Images y CarDD
(~1000 px), así que sirven para el costo y no para la calidad de la detección. Con 10 alertas por escenario
(20 con las de ~1000 px) los p95 son orientativos. Carga y decodificación de una foto de 12 megapíxeles: unos
50 ms cada una.

#### Con presencia de auto y re-ID en todas las fotos

Misma máquina, `PLATE_TILING=true`, `scrfd_10g` para rostros, redacción activa, rebajes simulados y la
cascada completa: presencia de auto en cada foto (`NO_VEHICLE`), re-ID de detalles y contexto contra la general,
redacción de todas las fotos. El banco afloja los umbrales (`--real-thresholds` los deja como están) para que las
alertas recorran todas las etapas: las 90 alertas medidas terminaron verdes salvo 1 de 10 en la de 4000 px del
garage. Medianas por alerta, una por vez, sin otros programas abiertos:

| Fotos | CRASHED de 3 fotos | CRASHED de 5 fotos | PARKED_ON_GARAGE de 3 fotos |
|---|---|---|---|
| ~1000 px (20 alertas) | **11,8 s** (p95 12,7) | **19,4 s** (p95 23,6) | **10,5 s** (p95 12,3) |
| 4000 × 3000 px (10 alertas) | **15,7 s** (p95 18,0) | **24,0 s** (p95 28,1) | **16,7 s** (p95 22,2) |

Por etapa, con fotos de ~1000 px y 3 fotos (CRASHED): carga 32 ms, **vehículo (presencia) 7,6 s**, ráfaga y re-ID 2,3 s,
patente 0,1 s, daño 0,6 s y redacción 0,7 s. La presencia es la etapa cara: corre el detector con mosaico en
**cada** foto (unos 2,5 s por foto) y, cuando el detector no ve el auto, el segmentador. Sin el mosaico de patentes
esa etapa baja a unos 0,3 s por foto. La re-ID suma una segmentación y un embedding por cada foto de detalle
(unos 0,4 s). Con otro programa pesado abierto (un juego) las mismas corridas dieron tiempos mucho menos estables (hasta un 30 % de diferencia entre corridas).

**Lo que no mide:** la red (descarga y subida a S3) ni el descifrado HPKE, que acá son en memoria. Las fotos de
Open Images miden unos 1000 px; las del celular pueden ser más grandes y sumar tiempo de decodificación.
También es una alerta por vez: con varios workers en la misma CPU cada una va a tardar más.

## Presencia de auto en todas las fotos (`presence.py`)

Mide qué regla decide bien que en una foto hay un auto (o, en PLATE y DETAIL, una parte). Corre el detector, el
segmentador y el detector de daños reales sobre cuatro conjuntos, fuera del repo:

- **general**: 417 fotos de autos enteros de Open Images (`damage_fp.py select`);
- **plate**: 137 recortes de patentes del dataset de Roboflow, con un 50 % de margen;
- **detail**: 374 fotos de prueba de CarDD (primeros planos de la carrocería);
- **negativas**: 489 fotos de Open Images sin `Car`, `Vehicle registration plate` ni `Wheel` anotados.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"; $env:EVAL_DATA_DIR = "D:\vv-eval"; $env:EVAL_OUT = "D:\vv-eval\out\presence"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/presence.py
```

Muestra, por conjunto, cuántas fotos pasan como auto con cada regla y, por cada **área mínima** (fracción de la
foto que ocupa el hallazgo), las negativas que pasan (regla A: tienen que ser 0) y las válidas rechazadas
(regla C). Las reglas: `solo_car` (caja `car`), `R1` (car, wheel o plate), `R2` (R1 o un auto segmentado) y `R3`
(R2 o una caja de daño).

Resultado (umbrales de confianza 0,40 para el detector y 0,50 para el segmentador):

| Regla | Área mínima | Negativas que pasan (de 489) | general rechazadas (de 417) | plate rechazadas (de 137) | detail rechazadas (de 374) |
|---|---|---|---|---|---|
| `car` + segmentador | 0 | 8 | 2 | n/a | n/a |
| `car` + segmentador | **5 %** | **0** | **2** | n/a | n/a |
| R2 | 0 | 8 | n/a | 1 | 20 |
| R2 | **5 %** | **0** | n/a | **1** | **21** |
| R3 (con daño) | 5 % | 179 | n/a | 1 | 3 |

Las 8 negativas que pasaban sin área mínima son retratos con un auto chico y sin anotar al fondo: con un 5 % de la
foto dejan de pasar. **El detector de daños no sirve para esto** (con 0,70 dispara en 179 de las 489 fotos sin
auto). Los primeros planos de la carrocería que no muestran ninguna parte reconocible (21 de 374) se rechazan:
la regla A gana sobre la C.

## Veredicto de la IA: solo GREEN o RED

La IA emite **GREEN o RED**; ya no hay amarillo. RED solo con evidencia firme:

| Regla | Código | Cuándo |
|---|---|---|
| R1 | `NO_VEHICLE` (fraude) | En alguna foto no hay evidencia de un auto: GENERAL y CONTEXT, un auto; PLATE, un auto, una parte o una patente; DETAIL, lo mismo o una caja de daño de confianza `DAMAGE_PRESENCE_T`. |
| R2 | `BURST_MISMATCH` (fraude) | La re-ID confirma que es otro auto (coseno < `REID_T`). Dudoso o no comparable: GREEN. |
| R3 | `PLATE_MISMATCH` (fraude) | Lectura segura (confianza >= `OCR_MIN_CONF`) que no coincide. Casi igual o ilegible: GREEN. |
| R4 | `EVIDENCE_UNREADABLE`, `REDACTION_UNAVAILABLE` | Sin fraude: foto ilegible o no se pudo redactar todas las fotos. |
| R5 | (sin códigos) | Todo lo demás: GREEN. |

Salieron de la cascada el chequeo de **rebajes** (experimental y en construcción: `MODEL_CURB` vacío, sin pesos) y el
chequeo de **daño** (solo queda como prueba de presencia en DETAIL). `presence.py` calibra `DAMAGE_PRESENCE_T`: el primer
valor sobre la mayor confianza de daño de una de las 489 fotos sin auto de Open Images (0,97: pasan 0 de 489, cota del
95 % = 0,6 %), que recupera 5 de los 21 primeros planos de CarDD que las otras reglas rechazaban.

## Rebajes: ¿el auto bloquea el rebaje? (`curb.py`)

Mide cómo decide el servicio si un auto bloquea el tramo rebajado del cordón, sobre las fotos de **prueba**
del dataset propio (`EVAL_DATA_DIR\curb\dataset\test`, que arma `tools/training/curb/prepare.py` y no cambia).
Las máscaras de auto salen de `rfdetr_seg_coco` y la decisión de la misma función del servicio
(`driveway_overlap`), con `RAMP_T_LOW`, `RAMP_T_HIGH` y `RAMP_SCALE_FALLBACK`.

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval"
$env:EVAL_OUT = "D:\vv-eval\out\curb"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/curb.py
```

Hace falta `rfdetr_seg_coco` y el modelo de rebajes entrenado (`rfdetr_seg_curb`, ver
[tools/training/curb/README.md](../training/curb/README.md)). Para correrlo sin SAM 3, que necesita el acceso de
Hugging Face y la GPU: `--no-sam3`. Con `--detector rfdetr_openimages` mide la rueda del auto; sin eso se usa
la escala de respaldo (`RAMP_SCALE_FALLBACK`).

### Los tres sistemas

| Sistema | Qué es |
|---|---|
| `sam3` | SAM 3 sin entrenar, con prompts de texto. Es la referencia: "¿hace falta entrenar?" |
| `entrenado` | El RF-DETR-Seg entrenado con tus fotos. |
| `oraculo` | Las máscaras que anotaste vos a mano. **No es un modelo**: sirve para separar errores (ver abajo). |

La comparación sin entrenar contra entrenado es un resultado para la tesis.

### Qué decisión se espera de cada tipo de foto

| Tipo | Decisión esperada |
|---|---|
| `positivo_claro` | `BLOCKING` |
| `positivo_marginal` | `BLOCKING` (la decisión es binaria: ya no hay `BLOCKING_MARGINAL`) |
| `negativo_auto_junto_rebaje` | `NOT_BLOCKING` |
| `negativo_auto_sin_rebaje` | `NO_DRIVEWAY` |
| `negativo_parecido` | `NO_DRIVEWAY` (o `NO_CAR` si no hay auto) |
| `negativo_rebaje_sin_auto` | `NO_CAR`, y además el modelo tiene que **detectar la máscara de `curb_cut`** |

Donde hay varios autos se evalúa el más grande (el servicio elige al del reporte por su apariencia, pero acá hay
una sola foto).

### Cómo leer el resultado

Por cada sistema muestra los aciertos, la **precisión** y el **recall de `BLOCKING`** y los resultados por tipo (`curb_report.json`). Las fallas se dibujan en
`EVAL_OUT\failures\<sistema>\`, con lo anotado en verde y lo predicho en naranja.

Cada falla se clasifica:

- **`mascara_mala`**: la máscara del sistema no se parece a la anotada (IoU menor a 0,5) en el cordón o en el
  rebaje. El problema es el modelo: hace falta mejor modelo, más fotos o mejores prompts.
- **`decision_mala_con_mascara_buena`**: las dos máscaras están bien y aun así la decisión es otra. El problema
  es el criterio, no el modelo: **se ajusta el umbral (`RAMP_T_LOW`), no se reentrena**.

Para eso el reporte incluye un **barrido de `RAMP_MARGIN` con las máscaras anotadas** (las del oráculo) y el
valor que más decisiones acierta; si hay empate, el más cercano al actual. Es la forma de saber cuánto ayuda el
margen sin que cuente el error del modelo.

Con pocas fotos de prueba, una sola foto mueve los porcentajes: mirá siempre las cantidades.

### Referencia sin entrenar: `--baseline`

Cuando todavía no hay máscaras corregidas ni modelo entrenado, `--baseline` corre **solo SAM 3** sobre todas las
fotos de `raw\` que no están descartadas y compara la decisión con el `tipo` del `fotos.csv`:

```powershell
uv run --frozen --extra dev --extra tools python tools/eval/curb.py --baseline --detector rfdetr_openimages --curb-model none
```

Muestra, por tipo, la decisión binaria (`BLOCKING` o `NOT_BLOCKING`, más `NO_DRIVEWAY` y `NO_CAR`) y el veredicto
que daría el servicio con `RAMP_T_LOW` y `RAMP_T_HIGH` (verde si bloquea con claridad, amarillo si es dudoso, rojo
si no hay problema). Además calibra los umbrales con las reglas de decisión: `T_LOW` queda justo sobre el mayor
solape de un negativo (que ningún caso RED llegue a verde ni amarillo), se cuentan los positivos que quedan
debajo (válidos que terminarían en rojo) y se informa la cota del 95 % (3/N). `T_HIGH` **no se puede calibrar**
mientras no haya un set de casos dudosos (amarillos): se deja igual a `T_LOW`. La salida es `curb_baseline.json`.
