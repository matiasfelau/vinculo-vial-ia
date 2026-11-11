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

### R3 con pares cruzados

El script también mide qué hace R3 (`PLATE_MISMATCH` solo con una lectura segura que no coincide) cuando la patente de
una foto se compara con la etiqueta de **otra** (`pares_cruzados` en el reporte). Con `OCR_MIN_CONF=0,78` y solo los
formatos en alcance: de 14 835 pares, el **75,2 %** sale `MISMATCH` (RED); el resto se escapa como GREEN, casi todo
(24,8 %) porque la lectura de esa foto no llega a 0,78 y 4 pares (0,03 %) por una etiqueta casi igual (`FUZZY`). Ningún
par da `MATCH`. Cuando la lectura es segura, el 99,96 % sale `MISMATCH`. Del otro lado, ninguna patente en alcance que
coincide con su etiqueta sale `MISMATCH` con 0,78.

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
FACE_TILING=true                   # +15 puntos de recall en caras chicas; entra en los 12 s solo porque la redacción va en paralelo (ver la cascada)
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
| `--face-tiling on\|off` | `FACE_TILING` (por defecto `on`, como el servicio). |
| `--redact-workers N` | `REDACT_WORKERS`: fotos que se planifican a la vez al redactar (por defecto el del servicio, 3). |
| `--model-max-side N` | `MODEL_MAX_SIDE_PX`: lado mayor al que se reduce la foto antes de los modelos (1024). |
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

#### Un solo pase del detector, fotos reducidas y redacción en paralelo (meta: 5 fotos en 12 s)

Cambios que bajan la latencia sin tocar el veredicto: el detector corre **un pase por foto** (sobre la foto
reducida a `MODEL_MAX_SIDE_PX`=1024) y lo reutiliza la presencia, la lectura de la patente y la redacción; el
mosaico (`FACE_TILING`, `PLATE_TILING`) solo corre al redactar, es decir, nunca si el veredicto ya es RED; y la
redacción planifica hasta `REDACT_WORKERS`=3 fotos a la vez (con la memoria del detector compartida entre hilos).
Las cajas vuelven a las coordenadas de la foto original antes de tapar.

Mismas condiciones que arriba (Ryzen 5 3600, una alerta por vez, sin otros programas abiertos, fotos de
4000 × 3000 px, 8 alertas por escenario, umbrales aflojados para recorrer todas las etapas). Medianas en ms:

| Configuración | Escenario | Total | p95 | load | vehicle | burst | plate | redaction |
|---|---|---|---|---|---|---|---|---|
| Una foto a la vez, mosaico de patentes | CRASHED de 3 fotos | 7 808 | 8 857 | 180 | 910 | 1 355 | 54 | 5 258 |
| | **CRASHED de 5 fotos** | **13 017** | 15 156 | 306 | 1 587 | 2 344 | 61 | 8 710 |
| | PARKED_ON_GARAGE de 3 | 8 099 | 8 896 | 189 | 928 | 1 475 | 61 | 5 484 |
| Una foto a la vez, mosaicos de patentes y rostros | CRASHED de 3 fotos | 9 086 | 9 745 | 183 | 934 | 1 409 | 58 | 6 474 |
| | **CRASHED de 5 fotos** | **14 674** | 15 864 | 298 | 1 514 | 2 445 | 64 | 10 634 |
| | PARKED_ON_GARAGE de 3 | 9 183 | 9 800 | 180 | 907 | 1 429 | 52 | 6 501 |
| 3 fotos a la vez, mosaico de patentes | CRASHED de 3 fotos | 5 383 | 5 773 | 179 | 934 | 1 445 | 60 | 2 809 |
| | **CRASHED de 5 fotos** | **8 894** | 9 367 | 304 | 1 582 | 2 250 | 62 | 4 730 |
| | PARKED_ON_GARAGE de 3 | 5 353 | 5 951 | 181 | 899 | 1 460 | 51 | 2 768 |
| **3 fotos a la vez, mosaicos de patentes y rostros (default)** | CRASHED de 3 fotos | 5 926 | 6 244 | 189 | 883 | 1 367 | 52 | 3 433 |
| | **CRASHED de 5 fotos** | **9 880** | **10 609** | 292 | 1 465 | 2 136 | 51 | 5 942 |
| | PARKED_ON_GARAGE de 3 | 5 901 | 6 521 | 183 | 871 | 1 422 | 54 | 3 400 |

Con el default, la alerta de 5 fotos (la peor) tarda 9,9 s de mediana y 10,6 s de p95, bajo los 12 s. Una sola foto cuesta
unos 320 ms el pase completo, 1,3 s el mosaico de patentes (4 tiles de 640 px sobre la foto reducida) y 0,1 s
(0,6 s con mosaico) los rostros; el render es de unos 35 ms. El paralelismo es el que cierra la cuenta: con 3 hilos el
tiempo de planificar 5 fotos pasa de unos 10 s a unos 6,5 s. No es lineal porque ONNX Runtime ya usa varios núcleos por
inferencia.
Medido en una sola máquina de escritorio, sin red ni HPKE: en otra CPU hay que volver a medir. Si se corren varias
alertas a la vez (`WORKERS` > 1) cada una usa hasta `REDACT_WORKERS` hilos más, y conviene bajar este último.

**Lo que no mide:** la red (descarga y subida a S3) ni el descifrado HPKE, que acá son en memoria. Las fotos de
Open Images miden unos 1000 px; las del celular pueden ser más grandes y sumar tiempo de decodificación.
También es una alerta por vez: con varios workers en la misma CPU cada una va a tardar más.

### Latencia por alerta sin segmentador (3 y 5 fotos)

Mismas condiciones de arriba (**AMD Ryzen 5 3600**, una alerta por vez, fotos de 4000 × 3000 px, `REDACT_WORKERS` 3,
mosaicos de patentes y rostros, umbrales aflojados para recorrer todas las etapas), pero **20 alertas por escenario**
para que el p95 sea una medida y no el máximo de ocho. Antes de medir se terminaron los procesos sobrantes de
corridas anteriores y se corrió una sola medición a la vez. En ms:

| Configuración | Escenario | Mediana | **p95** | Máximo |
|---|---|---|---|---|
| **Oficial: sin segmentador ni daño** | **3 fotos** (general, patente, detalle) | 5 680 | **6 522** | 6 705 |
| | **5 fotos** (general, patente, 3 detalles) | 9 103 | **10 339** | 10 427 |
| | 3 fotos, PARKED_ON_GARAGE (general, patente, contexto) | 5 613 | 6 274 | 6 327 |
| Comparación: sin segmentador, con daño | 3 fotos | 5 517 | 6 052 | 6 065 |
| | 5 fotos | 9 247 | 9 631 | 9 921 |
| | 3 fotos, PARKED_ON_GARAGE | 6 540 | 7 439 | 7 894 |
| Comparación: con segmentador y daño (antes) | 3 fotos | 7 888 | 8 924 | 9 196 |
| | 5 fotos | 12 872 | 13 551 | 14 462 |
| | 3 fotos, PARKED_ON_GARAGE | 7 974 | 9 009 | 9 947 |

Etapas medianas de la configuración oficial (ms): 3 fotos, carga 194, presencia 1 070, re-ID 310, patente 70,
redacción 4 040; 5 fotos, carga 302, presencia 1 664, re-ID 489, patente 64, redacción 6 576. Sacar el segmentador
baja la re-ID de unos 2 000-3 000 ms a 300-500 ms. Los dos escenarios quedan bajo los 12 s del objetivo (p95 de 6,5 s con
3 fotos y de 10,3 s con 5). La diferencia entre "con daño" y "sin daño" está dentro del ruido de la medición.

### Producción: hilos por alerta y concurrencia del consumer

Tres perillas multiplican el uso de CPU de **una** instancia del servicio:

| Variable | Qué controla | Por defecto |
|---|---|---|
| `WORKERS` | Alertas que se procesan a la vez. Es el tamaño del pool de inferencia **y** el `prefetch` del consumer de RabbitMQ: el broker no entrega más de `WORKERS` mensajes sin confirmar a la instancia. | 2 |
| `REDACT_WORKERS` | Fotos que **cada alerta** planifica a la vez al redactar (rostros, patentes y OCR). Se acota a la cantidad de fotos de la alerta (3 a 5). | 3 |
| `ONNX_INTRA_OP_THREADS` | Hilos de ONNX Runtime **por inferencia**. Con 0 cada inferencia usa todos los núcleos. | 0 |

`REDACT_WORKERS` **multiplica** los hilos de cada alerta, y `WORKERS` multiplica ese resultado:

```
inferencias simultáneas  ≈ WORKERS × REDACT_WORKERS         (solo durante la redacción)
hilos de cálculo         ≈ WORKERS × REDACT_WORKERS × ONNX_INTRA_OP_THREADS
```

Con los valores por defecto hay hasta 2 × 3 = 6 inferencias a la vez y, con `ONNX_INTRA_OP_THREADS=0`, cada una
intenta usar todos los núcleos: el sistema se sobrecarga y cada alerta tarda **más** que con menos paralelismo
(los 9,9 s de la alerta de 5 fotos se midieron con `WORKERS` = 1, una alerta por vez, en 6 núcleos físicos).
La regla para no saturar la CPU es:

```
WORKERS × REDACT_WORKERS × ONNX_INTRA_OP_THREADS  ≤  núcleos físicos del contenedor
```

La regla es conservadora: con `WORKERS=1` lo medido (abajo) la supera y rinde, porque una inferencia sola no ocupa
todos los núcleos (por eso 3 fotos a la vez son más rápidas que una por una). Úsala como tope cuando hay varias alertas
a la vez. Si hay varias réplicas del servicio en el mismo nodo, el tope es el de **todas** juntas. Puntos de partida (hay que
validarlos con `tools/eval/cascade.py --redact-workers N --threads M` en el hardware real):

| Núcleos físicos | `WORKERS` | `REDACT_WORKERS` | `ONNX_INTRA_OP_THREADS` | Para qué |
|---|---|---|---|---|
| 4 | 1 | 2 | 2 | Una alerta por vez, lo más rápida posible. |
| 4 | 2 | 1 | 2 | Dos alertas a la vez, cada una más lenta. |
| 8 | 2 | 2 | 2 | Término medio. |
| 6 (Ryzen 5 3600) | 1 | 3 | 0 | Lo medido: 9,9 s con 5 fotos de 4000 × 3000 px. |

- **Latencia de una alerta** (la que tiene meta de 12 s): conviene `WORKERS` bajo y `REDACT_WORKERS` alto.
- **Cantidad de alertas por minuto**: conviene `WORKERS` más alto y `REDACT_WORKERS` bajo. El paralelismo entre fotos
  ya se aprovecha dentro de cada alerta, así que subir ambos a la vez no suma: compite por los mismos núcleos.
- Más `WORKERS` también significa más fotos descifradas y decodificadas en RAM a la vez (cada foto de 12 megapíxeles
  ocupa unos 36 MB sin comprimir); el límite de memoria del contenedor tiene que cubrir `WORKERS` alertas de 5 fotos.
- Si la cola crece, es preferible agregar réplicas (cada una con su propio `prefetch`) y repartir los núcleos entre
  ellas, no subir `WORKERS` por encima de los núcleos disponibles.

### Autos y patentes: precisión, recall y mAP50 (micro-promedio)

Lo que se presenta del detector son **autos y patentes**: la clase rueda no se presenta (aunque la regla de presencia
R1 la usa; ver abajo) y los rostros los cubre otro modelo. Mismos datos y umbrales del servicio (autos 0,40, patentes
0,20, el de redacción), sobre las 204 fotos del split `test` de Open Images que el detector nunca vio, con cada foto
reducida a `MODEL_MAX_SIDE_PX` = 1024 como hace el servicio. El micro-promedio junta las cajas de las dos clases
(el acierto sigue siendo por clase, con IoU 0,5); el AP50 combinado ordena todas por confianza en una sola curva.

| Clase | Cajas reales | Precisión | Recall | AP50 | Falsos positivos |
|---|---|---|---|---|---|
| Auto | 244 | 0,747 | 0,898 | 0,897 | 74 |
| Patente, sin mosaico | 135 | 0,737 | 0,830 | 0,787 | 40 |
| Patente, con mosaico (`PLATE_TILING`) | 135 | 0,632 | 0,904 | 0,831 | 71 |
| **Auto + patente (micro), sin mosaico** | 379 | **0,744** | **0,873** | **0,851** | 114 |
| **Auto + patente (micro), con mosaico** | 379 | **0,702** | **0,900** | **0,866** | 145 |

El mosaico sube el recall de las patentes (y de las chicas, de 0,61 a 0,76) a costa de más cajas de más, que al tapar
cuestan poco. Las cifras anteriores (precisión 0,81, recall 0,79, mAP50 0,83) eran promedios de las cuatro clases:
con la rueda (recall 0,54 con el umbral de 0,40) y los rostros (0,76) el promedio no describe a autos y patentes.

**Latencia del detector por foto** (CPU, ONNX Runtime; **AMD Ryzen 5 3600**, 6 núcleos, la PC sin otras cargas
significativas, sin contar la decodificación ni la reducción de la foto):

| Configuración | Media | p95 |
|---|---|---|
| Un pase (autos y patentes; lo que decide la presencia) | 381 ms | 468 ms |
| Un pase y el mosaico de patentes (lo que corre al redactar) | 1 808 ms | 2 174 ms |

Se descartaron dos mediciones anteriores (695 y 2 137 ms de media): se hicieron con otros procesos consumiendo la CPU y
la memoria de la PC y salían casi al doble de lo que cuesta el modelo.

**R1 usa la clase rueda.** En PLATE y DETAIL cuenta como presencia de un auto una caja `car`, `wheel` o `plate` del
detector; en GENERAL y CONTEXT hace falta una caja `car`. Por eso el recall de la rueda importa para esos dos roles
(ver "R1 sin segmentador").

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

### R1 tal como la aplica el servicio, por rol

`presence.py` también mide la regla que corre en `_has_car`, con el área mínima de 5 % y `DAMAGE_PRESENCE_T` = 0,97
(el daño solo cuenta en DETAIL y no pide área): GENERAL y CONTEXT piden un auto (detector o segmentador); PLATE y DETAIL
también aceptan una rueda o una patente. Fotos válidas rechazadas (regla C) y negativas que pasan (regla A):

| Rol | Conjunto | Válidas rechazadas | Antes (R2 sin daño) | Negativas que pasan (de 489) |
|---|---|---|---|---|
| GENERAL (y CONTEXT) | 417 fotos de Open Images | **2** (0,5 %) | 2 (0,5 %) | 0 |
| PLATE | 137 recortes de Roboflow | **1** (0,7 %) | 1 (0,7 %) | 0 |
| DETAIL | 374 fotos de prueba de CarDD | **16** (4,3 %) | 21 (5,6 %) | 0 |

Con 0 negativas que pasan, la cota del 95 % es 3/489 = 0,6 %. El daño recupera 5 de los 21 primeros planos de
CarDD que se rechazaban sin él. Las rechazadas siguen siendo fotos sin ninguna parte reconocible.

### R1 sin segmentador (configuración oficial)

El segmentador (`rfdetr_seg_coco`) salió de la cascada, como el detector de daños y el chequeo de rebajes: con
`MODEL_SEGMENTER=null` (y `MODEL_DAMAGE=null`) la presencia la decide **solo el detector** (`car`; en PLATE y DETAIL
también `wheel` y `plate`, con el área mínima de 5 % de la foto) y la re-ID compara el recorte de la caja `car`. El
código del segmentador queda como experimental. Se midió todo de nuevo, con la foto reducida a 1024 px como hace el
servicio y comprobando que ningún modelo dejara de responder (0 fotos con señales faltantes en los cinco conjuntos).

**Fotos válidas rechazadas** (regla C; los mismos 417, 137 y 374 conjuntos de arriba):

| Rol | Con segmentador y daño (antes) | Sin segmentador, con daño | **Sin segmentador ni daño (oficial)** |
|---|---|---|---|
| GENERAL (y CONTEXT), de 417 | 2 (0,5 %) | 5 (1,2 %) | **5 (1,2 %)** |
| PLATE, de 137 | 1 (0,7 %) | 1 (0,7 %) | **1 (0,7 %)** |
| DETAIL, de 374 | 16 (4,3 %) | 21 (5,6 %) | **28 (7,5 %)** |

Sacar el segmentador cuesta 3 fotos de GENERAL (autos que el detector no marca con 0,40) y 5 de DETAIL, y sacar además
el daño cuesta 7 primeros planos de DETAIL más: el daño, con `DAMAGE_PRESENCE_T` = 0,97, recuperaba 7 de los primeros planos de carrocería
que el detector no reconocía. La medición con segmentador y con daño queda solo como comparación.

**Fotos sin auto aceptadas** (regla A), con dos tandas de negativas de Open Images (sin `Car`, `Vehicle registration
plate` ni `Wheel` anotados):

- **Vistas:** 489 fotos de las tres particiones del subconjunto del detector, **incluida la de entrenamiento**.
- **Nuevas:** 1 200 fotos que el detector nunca vio (`negatives.py`: partición `test` de Open Images V7, otras clases
  —rostros, personas, edificios, perros, árboles, flores— y ninguna foto del subconjunto).

| Rol | Vistas (de 489) | Cota 95 % | **Nuevas (de 1 200)** | Cota 95 % |
|---|---|---|---|---|
| GENERAL (y CONTEXT), sin segmentador | 0 | 0,6 % (3/N) | **15 (1,25 %)** | **1,9 %** |
| PLATE y DETAIL, sin segmentador | 0 | 0,6 % (3/N) | **21 (1,75 %)** | **2,5 %** |

Las dos cifras son las mismas con y sin segmentador y con y sin daño: el segmentador no filtraba estas fotos. **La
cifra de 0 sobre las vistas era optimista**: el detector se entrenó con esas fotos. Sobre fotos nuevas pasan 1,3 a
1,8 %. Se miraron las 21 que pasan: casi todas son otros objetos con forma de vehículo (aviones y planeadores, lanchas,
un camión, una topadora, una quitanieves, sillas de ruedas, bicicletas) o un objeto con una patente o una rueda parecidas
(un reloj, una historieta), y 3 tienen un auto real al fondo sin anotar. Son fallos reales de R1 sobre cosas parecidas a
un auto, no errores de etiquetas. Con 0 aceptadas la cota sería 3/1 200 = 0,25 %; con 15 a 21, es la de Clopper-Pearson.

## Calibración de `REID_T` con VeRi-776 (`reid_veri.py`)

R2 (`BURST_MISMATCH`) pide **0 pares del mismo auto por debajo de `REID_T`** (cota del 95 % = 3/N). Se calibra con
VeRi-776, solo para **validar**: no se entrena nada y no se copia nada al repo (su licencia es solo de investigación,
sin uso comercial). Para cada recorte de `image_test` (los 1 678 de `image_query` están dentro) corre lo mismo que la
cascada: segmenta los autos, toma el más grande y embebe su máscara (`rfdetr_seg_coco` + `dinov2_vits14`).

```powershell
$env:VERI_DIR = "D:\vv-eval\veri\VeRi"; $env:EVAL_OUT = "D:\vv-eval\out\reid-veri"; $env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/reid_veri.py
```

Opciones: `--per-track N` (recortes por auto y cámara, 3), `--workers N` (recortes embebidos a la vez, 4), `--limit N`
(para probar) y `--max-false-red` (tasa de pares del mismo auto debajo del umbral propuesto, 0,1 %).

- **Mismo auto**: mismo ID y cámaras distintas. **Autos distintos**: IDs distintos; aparte los **difíciles**, con el
  mismo color y el mismo tipo según `test_label.xml`.
- Un recorte donde el segmentador no encuentra ningún auto **no es comparable** (en la cascada R2 no decide: GREEN).

Resultado (5 011 recortes elegidos, 198 autos y 19 cámaras):

| Dato | Valor |
|---|---|
| Recortes comparables | 4 010 de 5 011; 1 001 (20 %) sin auto segmentado |
| Pares del mismo auto (N) | 48 437 |
| Pares de autos distintos / difíciles | 7 985 843 / 902 906 |
| Coseno medio: mismo auto / distintos / difíciles | 0,654 / 0,557 / 0,632 |
| AUC (1,0 = se separan del todo): todos / difíciles | 0,664 / 0,548 |
| **Umbral exacto** (0 pares del mismo auto debajo) | **−0,037**, cota del 95 % = 3/48 437 = 0,006 % |
| Autos distintos que detecta con el exacto (todos / difíciles) | 0,004 % / 0 % (**inútil**) |
| **`REID_T` propuesto** | **0,075**: 48 pares del mismo auto debajo (0,10 %), repartidos en 12 de 198 autos |
| Autos distintos que salen `BURST_MISMATCH` con 0,075 (todos / difíciles) | 0,55 % / 0,03 % |

Curva (falso rojo = par del mismo auto por debajo del umbral; detectados = autos distintos que salen `BURST_MISMATCH`):

| `REID_T` | Pares del mismo auto debajo (falsos rojos) | Autos distintos detectados | Difíciles detectados |
|---|---|---|---|
| 0,05 | 22 (0,05 %) | 0,26 % | 0,01 % |
| **0,075** | **48 (0,10 %)** | **0,55 %** | **0,03 %** |
| 0,10 | 86 (0,18 %) | 1,01 % | 0,05 % |
| 0,20 | 303 (0,63 %) | 4,21 % | 0,35 % |
| 0,30 | 902 (1,86 %) | 8,33 % | 1,51 % |
| 0,40 | 2 413 (4,98 %) | 15,8 % | 5,02 % |
| 0,50 (valor anterior) | 6 770 (13,98 %) | 31,3 % | 15,6 % |

Los histogramas de similitud se solapan casi por completo: los pares del mismo auto se concentran entre 0,50 y 0,80
y los de autos distintos entre 0,45 y 0,75. **En VeRi la re-ID no separa lo suficiente** para acusar fraude sin falsos
rojos: con 0,50 (el valor anterior) el 14 % de los pares del mismo auto se marcaría como otro auto. Con 0,075 el falso
rojo cae al 0,1 %, pero R2 casi no se dispara (detecta 0,55 % de los autos distintos): solo salta ante recortes
groseramente distintos. Con la regla "solo con evidencia firme", es lo correcto con estos datos.

**Limitaciones**

- **Dominio distinto.** VeRi son recortes de cámaras de vigilancia de baja resolución, con cámaras y luz distintas
  entre fotos del mismo auto. Las fotos de la app son de celular, de la misma escena y en segundos: lo esperable es
  que el mismo auto se parezca **más** y que 0,075 sea demasiado bajo. Hay que medirlo con fotos propias (pares
  general–patente del mismo auto y de autos distintos) antes de subirlo.
- **`REID_T` queda provisorio** hasta esa validación (`.env.example`, `config.py`).
- Los pares no son independientes (200 autos, unos 24 recortes por auto): la cota 3/N con N = 48 437 es optimista; el
  tamaño efectivo está más cerca de la cantidad de autos y de pares de cámaras.
- El 20 % de los recortes sin auto segmentado es por la baja resolución; en fotos de celular no se espera.
- Se usan hasta 3 recortes por auto y cámara (5 011 de los 11 579) para acotar el costo (unos 40 minutos en CPU).
- **Esta calibración usó máscaras del segmentador.** El segmentador salió de la cascada y ahora el embedder recibe la caja
  `car` del detector: con ese recorte, `REID_T` = 0,075 sigue dejando U1 en 0,1 % en VeRi y en 0,3 % en 3DRealCar (ver
  "Re-ID con 3DRealCar", abajo), pero detecta muy pocos autos distintos (S4 de 0,1 % y 5 %).

## Re-ID con 3DRealCar (`reid_3drealcar.py`)

Compara DINOv2 ViT-S/14 **sin ajustar** (`dinov2_vits14`) con el **ajustado** con 3DRealCar (`dinov2_vits14_reid`, ver
[tools/training/reid](../training/reid/README.md)), los dos sobre el **recorte de la caja `car` del detector** (la cascada
ya no tiene segmentador), con la regla del servicio: otro auto si el coseno es menor que `REID_T`. 3DRealCar (Du et al.,
ICCV 2025) es Apache-2.0 con uso comercial; licencia y cita en [THIRD_PARTY.md](../../THIRD_PARTY.md).

```powershell
$env:REID_DATA_DIR = "D:\vv-eval\3drealcar"; $env:MODELS_DIR = "models\weights"; $env:EVAL_OUT = "D:\vv-eval\out\reid-eval"
uv run --frozen --extra dev --extra tools python tools/eval/reid_3drealcar.py --models dinov2_vits14 dinov2_vits14_reid --veri-dir D:\vv-eval\veri\VeRi
```

**Protocolo.** Partición agrupada por auto (ninguno está en dos particiones), estratificada por luz, tipo y color, con
semilla fija. Pares del mismo auto: vistas separadas por **30° o más** según la pose de la cámara (se reportan por rango
de ángulo). Pares de autos distintos: una muestra de 100 000 y los **difíciles** (misma marca y mismo color, 30 000). El
umbral se **calibra sobre validación**: el mayor con la cota del 95 % de U1 en 1 % o menos; con eso fijo se reporta sobre
prueba. **U1** = pares del mismo auto marcados como distintos; **S4** = pares de autos distintos detectados.

**Datos que se usaron (límite importante).** El zip de 3DRealCar pesa 55 a 110 GB por tramo y Google Drive agotó la
cuota de descarga cuando se habían bajado 406 autos completos: **175 de luz estándar y 231 de luz oscura, ninguno de luz
reflectiva** (los 425 reflectivos seleccionados quedaron sin bajar). Con 411 autos (4 864 recortes; el detector no
encontró auto en 49 de 4 913 fotos) la partición queda en 293 autos de entrenamiento, 59 de validación y 59 de prueba
(26 estándar y 33 oscuros). **La condición reflectiva no se midió, ni se entrenó con ella.** Para completarla, repetir
`fetch.py pull` cuando se libere la cuota (baja lo que falta), luego `prepare.py`, `train.py`, `export.py` y esta
evaluación: la partición de los autos que ya están no cambia.

### Tabla 3: línea base contra ajustado (prueba: 59 autos, 3 287 pares del mismo auto, 100 000 distintos, 30 000 difíciles)

Con el umbral calibrado en validación (0,0736 sin ajustar; 0,0862 ajustado). Entre paréntesis, la cota del 95 % de U1.

| Conjunto | Modelo | AUC | U1 | S4 |
|---|---|---|---|---|
| Todos | sin ajustar | 0,684 | 0,3 % (0,6 %) | 5,1 % |
| | **ajustado** | **0,988** | **0,8 % (1,1 %)** | **76,5 %** |
| Difíciles (misma marca y color) | sin ajustar | 0,590 | 0,3 % (0,6 %) | 0,9 % |
| | **ajustado** | **0,951** | 0,8 % (1,1 %) | **47,4 %** |
| Luz estándar | sin ajustar | 0,681 | 0,8 % (1,3 %) | 9,2 % |
| | **ajustado** | **0,991** | 0,5 % (0,9 %) | **69,4 %** |
| Luz oscura | sin ajustar | 0,622 | 0,0 % (0,2 %) | 0,6 % |
| | **ajustado** | **0,974** | 1,1 % (1,6 %) | **70,4 %** |
| Luz reflectiva | ambos | no medida (sin datos, ver arriba) | | |
| Difíciles, luz estándar | ajustado | 0,997 | | 76,3 % |
| Difíciles, luz oscura | ajustado | 0,881 | | 30,2 % |

Con el `REID_T` que hoy usa el servicio (0,075): sin ajustar, U1 0,3 % y S4 5,2 %; ajustado, U1 0,6 % (cota 0,9 %) y S4 74,0 %.
U1 por rango de ángulo del ajustado (umbral 0,0862): 30-60° 0,7 % (de 550 pares), 60-90° 0,3 % (636), 90-135° 1,4 % (1 029)
y 135-180° 0,7 % (1 072): las vistas más separadas no fallan más que las cercanas.

**Lo que se separa.** El efecto del *ajuste* es enorme: el AUC sube de 0,68 a 0,99 y el S4 de 5 % a 76 %, con el mismo
recorte. El efecto del *recorte* (caja del detector en vez de máscara del segmentador) es chico: en VeRi, DINOv2 sin
ajustar da AUC 0,664 con máscaras (medición anterior, 4 010 recortes) y 0,641 con la caja del detector (4 506 recortes).
Sin ajustar, la re-ID casi no detecta autos distintos con ninguno de los dos recortes.

**Propuesta de `REID_T` (no aplicada).** Para `dinov2_vits14_reid`: **0,086**, con **U1 0,8 % (cota 1,1 %) y S4 76,5 %**
sobre prueba (objetivo S4 >= 70 % cumplido; la cota de U1 queda apenas sobre el 1 % pedido: 27 pares de 3 287). Sin ajustar,
con el recorte de la caja, el umbral calibrado (0,074) es casi el actual (0,075): U1 0,3 % y S4 5 %. **El servicio sigue con
`dinov2_vits14` y `REID_T` = 0,075: el modelo nuevo no está activado** (`MODEL_EMBEDDER`) y el umbral no se cambió.

### Referencia en VeRi-776 (¿generaliza fuera de 3DRealCar?)

Los mismos modelos y el mismo recorte del detector sobre los recortes de VeRi de `reid_veri.py` (4 506 de 5 011; en 505 el
detector no encuentra un auto): 55 206 pares del mismo auto, 10 090 292 de autos distintos y 915 116 difíciles.

| Modelo | AUC (todos / difíciles) | Con `REID_T` 0,075: U1 / S4 | Con el umbral de 3DRealCar: U1 / S4 | Con el umbral propio de VeRi (U1 0,9 %): S4 |
|---|---|---|---|---|
| Sin ajustar | 0,641 / 0,526 | 0,1 % / 0,1 % | 0,1 % / 0,1 % | 1,7 % |
| **Ajustado** | **0,810 / 0,621** | 0,1 % / 0,8 % | 0,2 % / 1,1 % | 4,6 % |

**Generaliza solo en parte:** el AUC sube de 0,64 a 0,81, pero el S4 útil sigue siendo bajo (1 a 5 %). VeRi son cámaras de
vigilancia de baja resolución y 3DRealCar son fotos de celular alrededor del auto: el parecido con las fotos de la app es
mayor con 3DRealCar, pero no está validado.

### Limitaciones

- **U1 es optimista:** los pares del mismo auto salen de una misma sesión de captura (el mismo lugar, el mismo día y la misma
  luz), así que se parecen más que dos fotos de personas distintas en momentos distintos. El S4 de los pares difíciles, no.
- **Puede haber diferencias con el parque automotor argentino:** las marcas y modelos están anotados en chino (BYD, Changan...).
- **No hay primeros planos de partes del auto:** el recorte de DETAIL (un primer plano) no es comparable; el servicio ya
  lo trata como "sin auto que comparar".
- **Pocos autos de prueba** (59, sin reflectivos): las cotas de los grupos chicos (luz estándar, 26 autos) son anchas.
- Ajustado con la partición de entrenamiento (293 autos de luz estándar y oscura): hay que repetirlo con los reflectivos.

## Veredicto de la IA: solo GREEN o RED

La IA emite **GREEN o RED**; ya no hay amarillo. RED solo con evidencia firme:

| Regla | Código | Cuándo |
|---|---|---|
| R1 | `NO_VEHICLE` (fraude) | En alguna foto no hay evidencia de un auto: GENERAL y CONTEXT, un auto; PLATE, un auto, una parte o una patente; DETAIL, lo mismo o una caja de daño de confianza `DAMAGE_PRESENCE_T`. |
| R2 | `BURST_MISMATCH` (fraude) | La re-ID confirma que es otro auto (coseno < `REID_T`, provisorio: 0,075 con VeRi). Dudoso o no comparable: GREEN. |
| R3 | `PLATE_MISMATCH` (fraude) | Lectura segura (confianza >= `OCR_MIN_CONF`) que no coincide. Casi igual o ilegible: GREEN. |
| R4 | `EVIDENCE_UNREADABLE`, `REDACTION_UNAVAILABLE`, `VALIDATION_FAILED` | Sin fraude: foto ilegible, no se pudo redactar todas las fotos, o el mensaje terminó en la DLQ sin veredicto (`VALIDATION_FAILED`, ver abajo). |
| R5 | (sin códigos) | Todo lo demás: GREEN. |

`VALIDATION_FAILED` no sale de la cascada: lo publica el consumidor. Cuando un mensaje termina en la DLQ y su `alert_id`
es legible, antes de derivarlo publica `ia.evidence.validated.red` con `reason_codes: ["VALIDATION_FAILED"]`, `scores` y
`model_versions` vacíos y `redacted: []`, con publisher confirm, para que Alert no espere el timeout de
`PENDING_VALIDATION`. Pasa con payload inválido (`invalid_payload`), error del sistema (`system_error`), reintentos
agotados (`max_retries_exceeded`) y error inesperado (`unexpected_error`). El `eventId` se deriva del evento de entrada,
como en cualquier veredicto: una redelivery repite el mismo aviso y Alert lo deduplica.

- Un sobre inválido, o sin `alert_id` legible (un UUID), va solo a la DLQ, como antes.
- Si el aviso no se confirma, el mensaje todavía no se deriva: se reintenta por la vía normal (`.retry`). Agotados los
  `MAX_RETRIES` va a la DLQ sin aviso, para no quedar en un ciclo, y Alert lo expira por su timeout.
- El aviso lleva solo el `alert_id`, el `eventId` de causa y el código; nunca se loguea el payload ni `pat_enc_ia`.

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
