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
| `--min-conf 0.7` | Umbral con el que se arma la distribución (por defecto `OCR_MIN_CONF` o 0,60). |
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
