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

### R3 con confusiones de caracteres (`ocr_confusions.py`)

**¿El servicio normaliza por posición según el formato?** Sí, y no hizo falta implementarlo: `plate_check`
(`domain/plate.py`), con la lectura del mismo largo que la patente reportada, lee cada carácter como dígito o como letra
según el de la reportada en esa posición (`TO_DIGIT`: O, Q, D → 0; I → 1; Z → 2; S → 5; G → 6; B → 8, y `TO_LETTER` al
revés). El formato lo da la reportada (`ABC123` o `AB123CD`), así que vale para los dos; el adaptador de PARSeq hace lo
mismo al buscar la patente dentro del texto. Los tests de `tests/domain/test_plate.py` cubren los seis pares en los dos
formatos. El límite es que necesita el mismo largo: con un carácter de más o de menos no hay posiciones que comparar y
la confusión cuenta como otra diferencia.

`ocr_confusions.py` compara tres criterios para declarar `PLATE_MISMATCH` (RED, fraude), con la misma carga y recorte que
`ocr.py`, y la confianza de cada carácter que ahora devuelve `ParseqPlateReader.read_chars` (solo para `tools/`; el
servicio sigue usando `read`):

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval"; $env:EVAL_OUT = "D:\vv-eval\out\ocr-r3"; $env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/ocr_confusions.py
```

- **V0, la actual**: lectura segura (confianza >= `OCR_MIN_CONF`) y diferencia en 2 o más caracteres; uno solo es duda.
- **V1, la actual + confusiones**: igual, pero una diferencia entre caracteres que el OCR confunde con frecuencia no
  cuenta: es duda, no una patente distinta.
- **V2, solo confusiones y baja confianza**: sin la regla del "un solo carácter" y sin umbral global sobre la lectura.
  Hay MISMATCH si algún carácter difiere, no es una confusión frecuente y PARSeq lo leyó con confianza >= `t_char`.

**S5** es cuántos de los pares cruzados (la lectura de una foto contra la etiqueta de otra patente: 14 835 pares de las 129
lecturas en alcance) salen MISMATCH. **U2** es cuántas de las lecturas contra su propia etiqueta salen MISMATCH: patentes
correctas marcadas como equivocadas. Son **133 lecturas** (las 137 cajas del dataset menos 3 marcadores de posición y 1
ilegible), con la cota de una cola al 95 %.

**La lista de confusiones sale de los errores reales de PARSeq en este dataset**: las diferencias entre la lectura y la
etiqueta, después de la normalización por posición, en las lecturas del mismo largo con 3 diferencias o menos (más que
eso es un recorte roto, no una confusión). Una pareja entra a la lista si aparece en al menos 2 lecturas. Como la lista y
la evaluación usan las mismas lecturas, **se evalúa dejando una lectura afuera**: para cada una, la lista se arma sin sus
propios errores, así que U2 no premia memorizar. Se evalúa además la lista a priori del pedido (C/G, O/D/Q, M/N, E/F, P/R,
U/V, 3/8, 1/7, 5/6), que no depende del dataset.

| Pareja | Lecturas en que aparece |
|---|---|
| 0/6 | 4 |
| M/W | 3 |
| O/Q | 2 |
| 0/1, 2/4, 2/7, 3/8, 3/J, 6/8, 7/9, A/Z, D/O, H/M, I/J, J/L, N/W, V/Y | 1 cada una |

Solo **tres parejas se repiten** (0/6, M/W y O/Q); de ellas, la lista a priori solo comparte O/Q. Con 129 lecturas en alcance y 25 sin coincidencia exacta, es una lista corta y poco estable.

| Variante | `OCR_MIN_CONF` o `t_char` | S5 (de 14 835) | U2 (de 133) | Cota 95 % de U2 |
|---|---|---|---|---|
| **V0, la actual** | 0,78 (la del servicio) | **75,2 %** (11 151) | **1** | 3,5 % |
| V0 | 0,60 | 89,9 % (13 336) | 3 | 5,7 % |
| V1, + confusiones aprendidas | 0,78 | 75,2 % (11 151) | 1 | 3,5 % |
| V1, + confusiones aprendidas | 0,60 | 89,9 % (13 336) | 3 | 5,7 % |
| V1, + confusiones a priori | 0,78 | 75,2 % (11 149) | 1 | 3,5 % |
| V2, aprendidas | `t_char` 0,95 | 99,93 % (14 824) | 6 | 8,7 % |
| V2, aprendidas | 0,90 | 100 % | 8 | 10,6 % |
| V2, aprendidas | 0,78 | 100 % | 8 | 10,6 % |
| V2, aprendidas | 0,70 | 100 % | 10 | 12,4 % |
| V2, aprendidas | 0,60 | 100 % | 12 | 14,2 % |
| V2, aprendidas | 0,50 | 100 % | 16 | 17,7 % |
| V2, a priori | 0,95 / 0,90 / 0,78 / 0,70 | 99,97 % / 100 % / 100 % / 100 % | 6 / 8 / 9 / 12 | 8,7 / 10,6 / 11,5 / 14,2 % |
| V2, sin lista (solo baja confianza) | 0,95 / 0,90 / 0,78 / 0,70 | 99,99 % / 100 % / 100 % / 100 % | 6 / 8 / 9 / 13 | 8,7 / 10,6 / 11,5 / 15,1 % |

**Lectura.**

- **Las confusiones no cambian nada en V1**: S5 y U2 son idénticos a V0 (con la lista a priori, 2 pares menos de 14 835). Las
  diferencias de 2 o más caracteres no son, en este dataset, parejas que se repitan, así que la lista no las absorbe
  (no revisé lectura por lectura de qué son).
- **V2 detecta casi todo (S5 ≈ 100 %) pero marca como fraude 6 a 16 de 133 patentes correctas** (4,5 % a 12 %), contra 1 de
  la actual. Una sola diferencia con confianza alta alcanza, y PARSeq se equivoca con seguridad en patentes que son
  correctas: el costo (una penalización injusta) es el error caro. Con `t_char` de 0,95 baja a 6, que sigue siendo 6 veces
  la actual.
- **La lista de confusiones ayuda poco en V2**: a `t_char` 0,78 pasa de 9 (sin lista) a 8 (aprendida, dejando una
  afuera) o 9 (a priori); a 0,50, de 21 a 16. Con 3 parejas repetidas no hay con qué.
- **No recomiendo cambiar de variante con estos datos**: V0 y V1 son iguales y V2 cambia una pérdida de 1 patente correcta por 8.
  La decisión es tuya; el servicio queda como estaba (V0).

**Decisión (14/11/2026): el servicio queda en V0** (`MISMATCH` solo con una lectura segura que difiere en 2 o más caracteres,
más la normalización por posición de `plate_check`). **La lista de confusiones no aportó**: solo 3 parejas se repiten en las
lecturas reales de PARSeq (0/6, M/W y O/Q), V1 da lo mismo que V0 y V2 cuesta 8 patentes correctas marcadas como fraude.

**Por qué U2 es "1 de 133" y antes se hablaba de "0 de 137" (no es una regresión).** Son dos denominadores y dos alcances, no dos
resultados. El dataset tiene 137 cajas; quedan 133 lecturas (se descartan 3 marcadores de posición y 1 ilegible), de las cuales
129 son de los formatos en alcance (Mercosur y anterior) y 4 de otros formatos (motos, de dos líneas). Con `OCR_MIN_CONF` = 0,78
y la regla V0, **ninguna de las 129 lecturas en alcance sale `MISMATCH` (0 de 129)**: es lo que dice la sección "R3 con pares
cruzados" y lo mismo que medía `ocr.py`. La única que sale `MISMATCH` es una **patente de moto de dos líneas fuera de alcance**
(etiqueta de 6 caracteres, PARSeq lee 3 con confianza 0,98 porque solo ve una línea), y `ocr_confusions.py` cuenta las 133
lecturas, no solo las 129. A `OCR_MIN_CONF` = 0,60 las marcadas son 3 (las dos motos y una lectura de 6 caracteres con confianza
0,76), que es el "3 de 133" de la tabla. Código y regla no cambiaron entre las dos mediciones. Que una patente de dos líneas
leída a medias salga RED con confianza alta es un riesgo real del criterio actual (la moto está fuera de alcance, pero el
servicio no la distingue al recibirla): queda como límite, sin cambios.

Límites: 133 lecturas, 25 con errores y una lista aprendida de 3 parejas repetidas; el dataset de Roboflow tiene fotos
con ángulo y desenfoque que no son las de la app; la confianza de PARSeq por carácter se calibró con este mismo dataset
solo en el sentido de que se barrió `t_char` sobre él (optimista). No se probó una variante que combine V0 con la baja
confianza por carácter.

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
cascada llegue a todas las etapas en lugar de cortarse en un RED) y zonas del vehículo como foto de detalle.
**CarDD no se usa** (decisión del 12/11/2026): las fotos salen de `capture_sets.py` (ver "R1 con el encuadre de 1-2 m").
El detector por defecto es v2 (`--detector` elige otro) y el daño no está en la cascada.

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\cascade"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/cascade.py --general-dir D:\vv-eval\capture-12m\plate --detail-dir D:\vv-eval\capture-12m\detail --alerts 40
```

| Opción | Para qué |
|---|---|
| `--alerts N` | Alertas por escenario (20 por defecto), más una de calentamiento que no cuenta. |
| `--detail-dir CARPETA` | **Obligatoria**: las fotos de detalle (los `detail` de `capture_sets.py`). |
| `--general-dir CARPETA` | Autos con patente (por defecto `<TRAIN_DATA_DIR>\damage-fp\images`; la serie `plate` de `capture_sets.py` sirve). |
| `--detector NOMBRE` | `MODEL_DETECTOR` (por defecto `rfdetr_openimages_v2`). |
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

## R1 con el encuadre de 1-2 m (`capture_sets.py` y `capture_r1.py`): la medición vigente

> **Esta es la medición de R1 que vale** (decisiones del 12/11 y del 14/11/2026). Las secciones siguientes ("Presencia de auto en todas
> las fotos" y todo lo que sigue hasta "Precisión corregida") son **históricas**: se hicieron con otras reglas de presencia
> (la patente o la rueda contaban, DETAIL tenía una regla propia con el daño), con una foto de detalle de CarDD (que ya no se
> usa para evaluar la cascada) y con **dos criterios de exclusión manual de negativas distintos**: "sin las 3" (v1: las 3 con un
> auto real al fondo) y "sin las excluidas" (v2: 11 fotos con un vehículo real). **Se unifican en uno solo: no hay exclusiones
> manuales.** Los números viejos no se comparan con los de esta sección. La medición del 12/11 de esta misma sección (con otra
> definición de negativa y con el colectivo como vehículo) fue **reemplazada** por la del 14/11; queda en el historial de git.

### La regla

R1 es **la misma para todas las fotos** (GENERAL, PLATE, DETAIL y CONTEXT): tiene que haber un vehículo, que es una caja
`car` del detector con confianza `DETECTOR_MIN_CONF` o una instancia del segmentador con confianza `SEGMENTER_MIN_CONF`
(**det ∨ seg**), y ocupa al menos `PRESENCE_MIN_AREA` (5 %) de la foto. **La patente no cuenta como presencia en ninguna foto**
(alcanza una patente arrancada o impresa para hacer pasar cualquier imagen), tampoco la rueda ni el daño. DETAIL no tiene
ninguna regla propia: no hay un área mínima ni una confianza distintas, y verificar el daño o la rampa es trabajo futuro. El
segmentador (`rfdetr_seg_coco`) está en la cascada y `.env.example` lo deja activo (`MODEL_SEGMENTER=rfdetr_seg_coco`).

**Qué clases cuentan como vehículo (decisión del 14/11/2026): auto (incluido el taxi), van y camión.** Es el mapeo del adaptador de
v2 (`ROLES`): `car`, `truck` y `van` se devuelven como una caja `car`; **`bus` (colectivo) es una salida del modelo que el adaptador
ya no devuelve**, así que un colectivo no cuenta (cambió solo el mapeo, no hubo reentrenamiento). El segmentador es COCO sin
entrenar y solo devuelve la clase `car` de COCO (no camión ni colectivo). **v2 es el detector por defecto** (`.env.example`);
v1 queda en el manifest como respaldo y ya no se mide. **Limitaciones:**

- **La ambulancia se entrenó como `van`** (`Ambulance` → `van` en `v2_classes.py`), no es una salida propia: sigue contando como
  vehículo. Que no cuente exige reentrenar (por ejemplo, mandarla a las fotos de fondo o a una confusora); no se hizo.
- **Open Images no distingue un flete de un camión pesado y anota muchas pickups como `Truck`**, así que `truck` se mantiene
  completo: cuenta una pickup válida y también un camión pesado que nadie fotografiaría como un auto.

### Supuesto de captura

El usuario fotografía **a 1-2 m del vehículo**: en la foto de patente se ve la trompa o la cola completa (corta entre el capot
y el parabrisas); en la general, el auto casi entero; en la de detalle, una zona del auto con el auto alrededor. Tres
fotos de referencia en `D:\vv-eval\ref\` se usaron **solo para medir el encuadre**: no calibran ningún umbral, no están en el
repo y no entran a ningún conjunto de evaluación. (La carpeta tiene 8 archivos, no 3: `1.jpg` y `3.jpg` son la misma foto, así que
son 7 distintas, y las clasifiqué a ojo: 2 de patente, 1 de patente más lejana, 3 generales y 1 de detalle.)

`capture_sets.py ref` mide, con el detector v2 (confianza >= 0,30; solo como regla de medir, no de decidir), el ancho de la
patente y el ancho y el área de la caja del vehículo sobre la foto de 1280 × 720:

| Foto de referencia | Tipo | Ancho del vehículo | Área del vehículo | Ancho de la patente |
|---|---|---|---|---|
| `5.jpg` | patente (trompa completa) | 0,654 | 0,503 | 0,132 |
| `7.jpg` | patente (trompa completa) | 0,625 | 0,411 | 0,115 |
| `4.jpg` | patente, más lejos | 0,655 | 0,380 | 0,062 |
| `1.jpg` (= `3.jpg`) | general (cola, tres cuartos) | 0,818 | 0,468 | no se detecta |
| `6.jpg` | general | 0,980 | 0,808 | — |
| `8.jpg` | general | 0,914 | 0,747 | — |
| `2.jpg` | detalle (esquina trasera con rueda) | 0,892 | 0,889 | — |

De ahí salen las proporciones de los conjuntos: vehículo de 0,82 a 0,98 del ancho en general; patente de 0,08 a 0,15 del ancho (lo
pedido; las de referencia dan 0,115 y 0,132) y vehículo de ~0,62 a 0,66 en patente; en detalle el vehículo cubre ~0,89 de la foto.

### Resolución

La app genera JPEG de **lado mayor <= 1600 px y <= 900 KB** (límite de 1 MiB por parte, `EVIDENCE_MAX_PART_BYTES`, y de 5 MB de
Kong). Cada recorte se lleva a **1600 px de lado mayor (1600 × 900) y a JPEG de <= 900 KB**, y después pasa por el mismo camino
que una foto real: el servicio la reduce a `MODEL_MAX_SIDE_PX` = 1024 antes de los modelos. Las fotos de origen de Open
Images tienen a lo sumo 1024 px, así que **el recorte se agranda**: tiene menos detalle real que una foto de celular de 1600 px.
Los modelos ven la foto a 512 px (el detector) y a la resolución del segmentador, así que el efecto en la presencia es chico,
pero es un límite (abajo).

### Cómo se armaron los conjuntos

Fotos de la **partición de prueba de Open Images V7** (anotaciones de `test/labels/detections.csv`), nunca usadas para entrenar
v1 ni v2, bajadas del bucket público. Una foto no entra a más de un conjunto. Semilla 0. `capture_sets.py build` arma los
válidos y `capture_sets.py extras` las negativas por principio, los colectivos, las ambulancias y las pickups.

| Conjunto | Cantidad | Cómo se recorta (ventana 16:9 sobre la foto de origen) |
|---|---|---|
| **general** | 400 | Foto con una caja `Car` (o `Limousine`, `Taxi`; sin cajas de grupo, de dibujo ni tomadas desde adentro). Ventana centrada en el vehículo, con su ancho al 0,80-0,98 de la foto, de al menos 480 px. |
| **plate** | 300 | Foto con una caja de `Vehicle registration plate` dentro de un vehículo, con la patente al 10-30 % del ancho del vehículo (referencias: 0,18 y 0,20). Ventana con la **patente al 8-15 % del ancho**, el centro de la patente al 0,40-0,70 del ancho y 0,50-0,70 del alto, el vehículo entero a lo ancho y la base del vehículo dentro de la foto (se corta entre el capot y el parabrisas). |
| **detail** | 400 | Foto con una caja `Car`. Ventana de ancho 0,30-0,55 del vehículo, de al menos 320 px, con **al menos el 85 % de la foto dentro de la caja del vehículo** (referencia: 0,89). |
| **heavy** (camiones y vans) | 150 + 85 | Fotos con `Truck` o `Van` anotado y **sin `Car` ni subclases** anotado (cualquier caja). Misma ventana que general sobre el vehículo pesado más grande. Además, la **foto entera** (`heavy_*_whole`). Salieron menos de 150 vans (85): son todas las de la partición que cumplen esas condiciones y admiten una ventana. |
| **pickup** | 126 | Ver abajo. |
| **colectivo** y **ambulancia** | 65 y 1 | Fotos con `Bus` o `Ambulance` anotado (caja que no sea de grupo, de dibujo ni desde adentro) y **sin ningún vehículo excluido** (abajo). Misma ventana que general sobre el vehículo, y la foto entera. Son **negativas**: se mide cuántas se rechazan. |
| **sinvehiculo** | 2 000 | Fotos al azar (semilla) de la partición de prueba con alguna caja anotada y **sin ningún vehículo válido** (abajo). Se usan como están (sin recortar; a lo sumo 1 024 px). **Sin exclusiones manuales.** |

**Una foto sin auto es una foto sin ningún vehículo válido (decisión del 14/11/2026).** Son vehículos válidos `Car` (y su subclase
`Limousine`), `Taxi`, `Van` y `Truck`: una foto con alguno de ellos anotado (cualquier caja, también de grupo o de dibujo) **no es
una negativa**. **Siguen siendo negativas** las fotos con colectivo, ambulancia, moto, bicicleta, tren, barco, avión o silla de
ruedas (y sus parientes: aeronave, helicóptero, tanque, moto de nieve, etc.; la lista completa está en `OTHER_VEHICLES`). Las que
tienen como **única** señal de vehículo una rueda, un neumático, una patente o un `Vehicle` / `Land vehicle` genérico son
**ambiguas**: están dentro de las negativas y se reportan aparte (S1 con ellas dentro y sin ellas).

Cuántas fotos se excluyen por etiqueta (partición de prueba: 112 194 fotos con alguna caja y no usadas en ningún entrenamiento;
`negativas_conteos` en `specs.json`; se imprimen antes de medir):

| Etiqueta | Fotos que la tienen | Excluidas atribuidas a su primera etiqueta |
|---|---|---|
| `Car` | 15 288 | 15 288 |
| `Limousine` | 68 | 38 |
| `Taxi` | 85 | 1 |
| `Van` | 526 | 297 |
| `Truck` | 909 | 465 |
| **Total excluidas** | | **16 089** |

Quedan **96 105 fotos**: 86 664 sin ningún vehículo anotado, 6 987 con otro vehículo (aviones 2 407, barcos 1 710, bicicletas 648,
motos 351, trenes 266, colectivos 173, ambulancias 5, sillas de ruedas 107, y otros) y **2 454 ambiguas** (rueda 1 004, neumático 647,
`Vehicle` 1 045, `Land vehicle` 1 258, patente 30; una foto puede tener más de una). Hay muy pocas ambulancias: **solo 5 fotos de
la partición de prueba** tienen una ambulancia anotada, y 1 entró al conjunto (las otras estaban en el entrenamiento o no admiten
una ventana): su cifra es una anécdota, no una medición.

**Pickups.** Open Images V7 **tiene** la etiqueta de imagen "Pickup truck" (`/m/0cvq3`, de las 20 931 clases del archivo de descripciones
verificada por personas; **no es una clase con cajas**, no está entre las 600 de detección). La partición de prueba tiene **532
fotos** con la etiqueta confirmada. Como no hay cajas de pickup, se toma la caja de vehículo de la foto (`Truck`, `Car`, `Van`)
solo si **domina**: la mayor tiene al menos 2 veces el área de la segunda (si no, no se sabe cuál es la pickup y no se elige a
mano). Candidatas no usadas en ningún entrenamiento ni en otro conjunto: 400; con una caja dominante y una ventana válida: 126.
Cómo las anota Open Images: `Truck` 107, `Car` 17 y `Van` 2: **la mayoría como `Truck`**, que es justo lo que decide la limitación de arriba.
Misma ventana que general sobre esa caja, y la foto entera.

**Verificación de que ninguna foto se usó para entrenar** (`exclusion_check.json`): las 5 498 fotos elegidas (todos los conjuntos,
incluidos los de la medición anterior) contra las **3 689 fotos de entrenamiento y validación de v1 y v2** (v1: 1 057 + 201; v2:
2 139 + 435), por **identificador** (nombre de archivo), por **SHA-256 del contenido** (los 3 689 archivos comparados) y por
**`OriginalMD5`** de Open Images: **0 coincidencias** en las tres. (La partición de prueba es disjunta de las de entrenamiento y
validación de Open Images por construcción; esto lo verifica sobre los archivos.)

### Protocolo

- **Mitades con semilla fija.** Cada conjunto se parte en A y B con la misma semilla (`halves`, semilla 0). Los umbrales se
  **eligen en A** y los números se **reportan en B**.
- **Barrido:** `t_car` y `t_seg` de 0,30 a 0,95 de a 0,05 (14 valores cada uno), con `PRESENCE_MIN_AREA` = 5 % fijo.
- **"La mejor combinación"**: det ∨ seg con la **menor U3 en las mitades A de las válidas** entre las que dejan la **cota
  del 95 % de S1 en la mitad A de las negativas en <= 1 %** (con 1 000 negativas, a lo sumo 4 aceptadas). Se elige con las
  ambiguas dentro de las negativas; aparte, la que se elegiría exigiendo ese S1 solo sobre las negativas sin las ambiguas. Se
  reporta también con el **segmentador fijo en 0,50** (el del servicio), barriendo solo el detector.
- **Configuraciones que se reportan:** `car` 0,35 y 0,40 (la actual), ambas con el segmentador en 0,50, y la mejor.
- **S1**: negativas aceptadas (cota de una cola al 95 %: 3/N sin fallas, Clopper-Pearson si hay), con las ambiguas dentro y sin ellas.
- **U3 por alerta de 3 fotos** (general + patente + un detalle) = 1 − (1 − pG)(1 − pP)(1 − pD), **suponiendo que los rechazos de
  las tres fotos son independientes**. Es una estimación: no hay alertas reales, y el mismo auto o la misma luz correlacionan los
  rechazos, lo que haría la unión real más chica; la cota que se reporta usa las cotas de cada tipo y es conservadora. Las
  alertas de `PARKED_ON_GARAGE` llevan una foto de contexto (de la vereda) que este conjunto no tiene.
- **Quién la dejó pasar**: de cada negativa aceptada en B se guarda qué señal alcanzó (detector o segmentador), con qué
  clase del detector, confianza y área.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"; $env:MODELS_DIR = "models\weights"
$env:EVAL_OUT = "D:\vv-eval\capture-12m"
python tools/eval/capture_sets.py build      # válidos y heavy (baja ~6 900 fotos)
python tools/eval/capture_sets.py extras     # negativas por principio, colectivos, ambulancias y pickups (~3 000 fotos)
python tools/eval/capture_sets.py verify     # exclusion_check.json
$env:EVAL_OUT = "D:\vv-eval\out\capture-v3"
python tools/eval/capture_r1.py measure --sets-dir D:\vv-eval\capture-12m --detector rfdetr_openimages_v2   # ~1 h; --resume sigue
python tools/eval/capture_r1.py report --sets-dir D:\vv-eval\capture-12m
```

(`build` también genera `neg` y `heavy_Bus`, de la medición del 12/11: ya no se miden; quedan para que la semilla reproduzca
los conjuntos válidos.)

### Resultados con v2 y el segmentador (det ∨ seg)

Detector `rfdetr_openimages_v2@6bcad50b33c5` y `rfdetr_seg_coco`, con el área mínima del servicio (5 %), con las clases que
cuentan (auto, van y camión). Cifras: fallas sobre el tamaño de la mitad y, entre paréntesis, la tasa y la cota superior del 95 % (una
cola). **Todo es la mitad B**: los umbrales de las filas "mejor" se eligieron en la mitad A. Señales y análisis en
`D:\vv-eval\out\capture-v3` (se midieron las 3 954 fotos sin una sola señal faltante). `car` / `seg` son las confianzas mínimas del
detector y del segmentador.

| Configuración | `car` / `seg` | S1 en B, con las ambiguas | S1 en B, sin las ambiguas | GENERAL rechazadas | PLATE rechazadas | DETAIL rechazadas | U3 por alerta | Camión y van rechazados | Pickups rechazadas | Colectivos rechazados | Ambulancias rechazadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`car` 0,40 (la actual)** | 0,40 / 0,50 | 12/1000 (1,2 %; ≤ 1,9 %) | 5/975 (0,5 %; ≤ 1,1 %) | 1/200 (0,5 %; ≤ 2,4 %) | 1/150 (0,7 %; ≤ 3,1 %) | 9/200 (4,5 %; ≤ 7,7 %) | 5,6 % (≤ 12,7 %) | 4/118 (3,4 %; ≤ 7,6 %) | 2/63 (3,2 %; ≤ 9,7 %) | 33/33 (100,0 %) | 0/1 (0,0 %) |
| **`car` 0,35** | 0,35 / 0,50 | 13/1000 (1,3 %; ≤ 2,1 %) | 6/975 (0,6 %; ≤ 1,2 %) | 0/200 (0,0 %; ≤ 1,5 %) | 0/150 (0,0 %; ≤ 2,0 %) | 9/200 (4,5 %; ≤ 7,7 %) | 4,5 % (≤ 10,9 %) | 3/118 (2,5 %; ≤ 6,4 %) | 0/63 (0,0 %; ≤ 4,8 %) | 31/33 (93,9 %) | 0/1 (0,0 %) |
| **Mejor en A**, segmentador barrido | 0,65 / 0,55 | 8/1000 (0,8 %; ≤ 1,4 %) | 3/975 (0,3 %; ≤ 0,8 %) | 9/200 (4,5 %; ≤ 7,7 %) | 1/150 (0,7 %; ≤ 3,1 %) | 23/200 (11,5 %; ≤ 15,9 %) | 16,1 % (≤ 24,8 %) | 23/118 (19,5 %; ≤ 26,5 %) | 32/63 (50,8 %; ≤ 61,7 %) | 33/33 (100,0 %) | 0/1 (0,0 %) |
| **Mejor en A**, segmentador fijo en 0,50 | 0,75 / 0,50 | 7/1000 (0,7 %; ≤ 1,3 %) | 3/975 (0,3 %; ≤ 0,8 %) | 11/200 (5,5 %; ≤ 8,9 %) | 1/150 (0,7 %; ≤ 3,1 %) | 34/200 (17,0 %; ≤ 22,0 %) | 22,1 % (≤ 31,2 %) | 32/118 (27,1 %; ≤ 34,7 %) | 46/63 (73,0 %; ≤ 82,0 %) | 33/33 (100,0 %) | 0/1 (0,0 %) |
| Mejor en A sin las ambiguas, segmentador barrido | 0,35 / 0,95 | 13/1000 (1,3 %; ≤ 2,1 %) | 6/975 (0,6 %; ≤ 1,2 %) | 0/200 (0,0 %; ≤ 1,5 %) | 0/150 (0,0 %; ≤ 2,0 %) | 9/200 (4,5 %; ≤ 7,7 %) | 4,5 % (≤ 10,9 %) | 3/118 (2,5 %; ≤ 6,4 %) | 0/63 (0,0 %; ≤ 4,8 %) | 31/33 (93,9 %) | 0/1 (0,0 %) |
| Mejor en A sin las ambiguas, segmentador en 0,50 | 0,50 / 0,50 | 10/1000 (1,0 %; ≤ 1,7 %) | 4/975 (0,4 %; ≤ 0,9 %) | 3/200 (1,5 %; ≤ 3,8 %) | 1/150 (0,7 %; ≤ 3,1 %) | 16/200 (8,0 %; ≤ 11,9 %) | 10,0 % (≤ 17,9 %) | 7/118 (5,9 %; ≤ 10,8 %) | 12/63 (19,1 %; ≤ 29,0 %) | 33/33 (100,0 %) | 0/1 (0,0 %) |

(S1 son las negativas aceptadas, de 1 000 en B o 975 sin las ambiguas. "Camión y van" son los 118 válidos pesados de B; las
pickups son 63; los colectivos 33 y la ambulancia 1: para ellos la tabla cuenta las **rechazadas**, que es lo que se espera, y entre
paréntesis va solo la tasa.)

**S1 con más detalle** (negativas de A, de B, de las 2 000, y sin las ambiguas):

| Configuración | `car` / `seg` | S1 en A | S1 en B | S1 en todas | A sin las ambiguas | B sin las ambiguas |
|---|---|---|---|---|---|---|
| **`car` 0,40 (la actual)** | 0,40 / 0,50 | 8/1000 (0,8 %; ≤ 1,4 %) | 12/1000 (1,2 %; ≤ 1,9 %) | 20/2000 (1,0 %; ≤ 1,5 %) | 5/983 (0,5 %; ≤ 1,1 %) | 5/975 (0,5 %; ≤ 1,1 %) |
| **`car` 0,35** | 0,35 / 0,50 | 8/1000 (0,8 %; ≤ 1,4 %) | 13/1000 (1,3 %; ≤ 2,1 %) | 21/2000 (1,1 %; ≤ 1,5 %) | 5/983 (0,5 %; ≤ 1,1 %) | 6/975 (0,6 %; ≤ 1,2 %) |
| **Mejor en A**, segmentador barrido | 0,65 / 0,55 | 4/1000 (0,4 %; ≤ 0,9 %) | 8/1000 (0,8 %; ≤ 1,4 %) | 12/2000 (0,6 %; ≤ 1,0 %) | 1/983 (0,1 %; ≤ 0,5 %) | 3/975 (0,3 %; ≤ 0,8 %) |
| **Mejor en A**, segmentador fijo en 0,50 | 0,75 / 0,50 | 3/1000 (0,3 %; ≤ 0,8 %) | 7/1000 (0,7 %; ≤ 1,3 %) | 10/2000 (0,5 %; ≤ 0,9 %) | 2/983 (0,2 %; ≤ 0,6 %) | 3/975 (0,3 %; ≤ 0,8 %) |
| Mejor en A sin las ambiguas, segmentador barrido | 0,35 / 0,95 | 7/1000 (0,7 %; ≤ 1,3 %) | 13/1000 (1,3 %; ≤ 2,1 %) | 20/2000 (1,0 %; ≤ 1,5 %) | 4/983 (0,4 %; ≤ 0,9 %) | 6/975 (0,6 %; ≤ 1,2 %) |
| Mejor en A sin las ambiguas, segmentador en 0,50 | 0,50 / 0,50 | 7/1000 (0,7 %; ≤ 1,3 %) | 10/1000 (1,0 %; ≤ 1,7 %) | 17/2000 (0,9 %; ≤ 1,3 %) | 4/983 (0,4 %; ≤ 0,9 %) | 4/975 (0,4 %; ≤ 0,9 %) |

**Vehículos pesados válidos y pickups** (rechazados; `Truck` o `Van` anotado y sin `Car`, recortados con el encuadre de general; las
pickups son las fotos con la etiqueta de imagen "Pickup truck", ver arriba):

| Configuración | `car` / `seg` | camiones en B | vans en B | camión y van en B | camión y van, todos | pickups en B | pickups, todas | pickups, foto entera, en B |
|---|---|---|---|---|---|---|---|---|
| **`car` 0,40 (la actual)** | 0,40 / 0,50 | 4/75 (5,3 %; ≤ 11,8 %) | 0/43 (0,0 %; ≤ 7,0 %) | 4/118 (3,4 %; ≤ 7,6 %) | 5/235 (2,1 %; ≤ 4,4 %) | 2/63 (3,2 %; ≤ 9,7 %) | 3/126 (2,4 %; ≤ 6,0 %) | 2/63 (3,2 %; ≤ 9,7 %) |
| **`car` 0,35** | 0,35 / 0,50 | 3/75 (4,0 %; ≤ 10,0 %) | 0/43 (0,0 %; ≤ 7,0 %) | 3/118 (2,5 %; ≤ 6,4 %) | 3/235 (1,3 %; ≤ 3,3 %) | 0/63 (0,0 %; ≤ 4,8 %) | 1/126 (0,8 %; ≤ 3,7 %) | 0/63 (0,0 %; ≤ 4,8 %) |
| **Mejor en A**, segmentador barrido | 0,65 / 0,55 | 21/75 (28,0 %; ≤ 37,8 %) | 2/43 (4,7 %; ≤ 13,9 %) | 23/118 (19,5 %; ≤ 26,5 %) | 49/235 (20,8 %; ≤ 25,7 %) | 32/63 (50,8 %; ≤ 61,7 %) | 54/126 (42,9 %; ≤ 50,6 %) | 29/63 (46,0 %; ≤ 57,1 %) |
| **Mejor en A**, segmentador fijo en 0,50 | 0,75 / 0,50 | 29/75 (38,7 %; ≤ 48,8 %) | 3/43 (7,0 %; ≤ 17,1 %) | 32/118 (27,1 %; ≤ 34,7 %) | 69/235 (29,4 %; ≤ 34,6 %) | 46/63 (73,0 %; ≤ 82,0 %) | 79/126 (62,7 %; ≤ 69,9 %) | 34/63 (54,0 %; ≤ 64,8 %) |
| Mejor en A sin las ambiguas, segmentador barrido | 0,35 / 0,95 | 3/75 (4,0 %; ≤ 10,0 %) | 0/43 (0,0 %; ≤ 7,0 %) | 3/118 (2,5 %; ≤ 6,4 %) | 3/235 (1,3 %; ≤ 3,3 %) | 0/63 (0,0 %; ≤ 4,8 %) | 1/126 (0,8 %; ≤ 3,7 %) | 0/63 (0,0 %; ≤ 4,8 %) |
| Mejor en A sin las ambiguas, segmentador en 0,50 | 0,50 / 0,50 | 7/75 (9,3 %; ≤ 16,8 %) | 0/43 (0,0 %; ≤ 7,0 %) | 7/118 (5,9 %; ≤ 10,8 %) | 14/235 (6,0 %; ≤ 9,2 %) | 12/63 (19,1 %; ≤ 29,0 %) | 19/126 (15,1 %; ≤ 21,3 %) | 9/63 (14,3 %; ≤ 23,6 %) |

**Colectivos y ambulancias** (negativas: se espera que se rechacen; la tabla cuenta las **rechazadas**):

| Configuración | `car` / `seg` | colectivos rechazados en B | todos | foto entera | ambulancias rechazadas en B | todas | foto entera |
|---|---|---|---|---|---|---|---|
| **`car` 0,40 (la actual)** | 0,40 / 0,50 | 33/33 (100,0 %) | 64/65 (98,5 %) | 65/65 (100,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |
| **`car` 0,35** | 0,35 / 0,50 | 31/33 (93,9 %) | 61/65 (93,8 %) | 64/65 (98,5 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |
| **Mejor en A**, segmentador barrido | 0,65 / 0,55 | 33/33 (100,0 %) | 65/65 (100,0 %) | 65/65 (100,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |
| **Mejor en A**, segmentador fijo en 0,50 | 0,75 / 0,50 | 33/33 (100,0 %) | 65/65 (100,0 %) | 65/65 (100,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |
| Mejor en A sin las ambiguas, segmentador barrido | 0,35 / 0,95 | 31/33 (93,9 %) | 61/65 (93,8 %) | 64/65 (98,5 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |
| Mejor en A sin las ambiguas, segmentador en 0,50 | 0,50 / 0,50 | 33/33 (100,0 %) | 65/65 (100,0 %) | 65/65 (100,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) | 0/1 (0,0 %) |

**Quién dejó pasar a las negativas aceptadas en B** (`negativas_aceptadas_en_B` en `r1_capture.json`; "clases del detector" cuenta
detecciones que pasan, no fotos):

| Configuración | aceptadas en B | solo detector | solo segmentador | ambos | clases del detector | ambiguas |
|---|---|---|---|---|---|---|
| **`car` 0,40 (la actual)** | 12 | 10 | 0 | 2 | car 8, truck 5 | 7 |
| **`car` 0,35** | 13 | 11 | 0 | 2 | car 9, truck 5 | 7 |
| **Mejor en A**, segmentador barrido | 8 | 6 | 2 | 0 | car 3, truck 3 | 5 |
| **Mejor en A**, segmentador fijo en 0,50 | 7 | 5 | 2 | 0 | car 2, truck 3 | 4 |
| Mejor en A sin las ambiguas, segmentador barrido | 13 | 13 | 0 | 0 | car 9, truck 5 | 7 |
| Mejor en A sin las ambiguas, segmentador en 0,50 | 10 | 8 | 0 | 2 | car 7, truck 4 | 6 |

Las negativas aceptadas en B con los umbrales actuales (0,40 / 0,50), una por fila. El número es la posición en el conjunto
`sinvehiculo`; área = fracción de la foto. Las etiquetas son las de Open Images, no una descripción mía:

| # | Tipo | Etiquetas de Open Images | Lo dejó pasar | Detector (clase, confianza, área) | Segmentador (confianza, área) |
|---|---|---|---|---|---|
| 207 | otro_vehiculo | Airplane | detector | truck 0,76 / 0,13 | — |
| 292 | otro_vehiculo | Airplane | detector | car 0,62 / 0,98 | — |
| 421 | ambigua | Tire, Vehicle, Wheel | detector | car 0,75 / 0,99 | — |
| 659 | ambigua | Tire, Vehicle, Wheel | detector | truck 0,77 / 0,05; truck 0,64 / 0,43 | — |
| 883 | sin vehículo | — | ambos | car 0,53 / 0,33 | 0,85 / 0,11 |
| 946 | ambigua | Land vehicle, Tire, Wheel | detector | car 0,85 / 0,42 | — |
| 1116 | ambigua | Land vehicle | detector | car 0,66 / 0,99 | — |
| 1300 | ambigua | Land vehicle, Tire, Wheel | detector | car 0,54 / 0,44 | — |
| 1600 | ambigua | Land vehicle, Tire, Vehicle, Wheel | detector | truck 0,41 / 0,51 | — |
| 1705 | sin vehículo | — | ambos | car 0,56 / 0,52 | 0,73 / 0,17 |
| 1878 | ambigua | Vehicle, Wheel | detector | truck 0,90 / 0,35 | — |
| 1969 | sin vehículo | — | detector | car 0,42 / 0,85 | — |

Con `car` 0,35 se suma una más (la 594: `car` 0,36 con un 6 % de la foto, sin etiquetas de vehículo) y las demás son las mismas. Con
la mejor combinación en A (0,65 / 0,55):

| # | Tipo | Etiquetas de Open Images | Lo dejó pasar | Detector (clase, confianza, área) | Segmentador (confianza, área) |
|---|---|---|---|---|---|
| 207 | otro_vehiculo | Airplane | detector | truck 0,76 / 0,13 | — |
| 421 | ambigua | Tire, Vehicle, Wheel | detector | car 0,75 / 0,99 | — |
| 659 | ambigua | Tire, Vehicle, Wheel | detector | truck 0,77 / 0,05 | — |
| 883 | sin vehículo | — | segmentador | — | 0,85 / 0,11 |
| 946 | ambigua | Land vehicle, Tire, Wheel | detector | car 0,85 / 0,42 | — |
| 1116 | ambigua | Land vehicle | detector | car 0,66 / 0,99 | — |
| 1705 | sin vehículo | — | segmentador | — | 0,73 / 0,17 |
| 1878 | ambigua | Vehicle, Wheel | detector | truck 0,90 / 0,35 | — |

#### Lectura

- **Con los umbrales actuales (0,40 / 0,50) S1 en B es 12 de 1 000 (1,2 %; cota 1,9 %) y no cumple el 1 %.** Sin las ambiguas es
  5 de 975 (0,5 %; cota 1,1 %), apenas por encima del 1 %. Con 0,35 son 13 (cota 2,1 %) y 6 de 975 (cota 1,2 %): bajar `car` de 0,40 a
  0,35 suma una negativa aceptada y baja el rechazo de válidas (GENERAL 1 → 0 de 200, PLATE 1 → 0 de 150, pickups 2 → 0 de 63, camión
  y van 4 → 3 de 118), pero **U3 por alerta pasa de 5,6 % a 4,5 %** (cota 12,7 % → 10,9 %). DETAIL no cambia (9 de 200, 4,5 %).
- **De las 12 aceptadas en B (0,40 / 0,50), 7 son ambiguas**: fotos con rueda, neumático, `Vehicle` o `Land vehicle` anotados, que
  son casi seguro vehículos que Open Images no marcó como auto (la lista está arriba; no excluí ninguna). Otras 2 tienen un avión
  anotado (el detector ve un `car` o un `truck` en el avión) y 3 no tienen ninguna etiqueta. **La dejó pasar el detector en las 12**
  (en 2, también el segmentador); **el segmentador solo no dejó pasar ninguna** con 0,50. Las clases son `car` (8 detecciones) y
  `truck` (5): **4 de las 12 fotos entran por `truck`**, incluidas 3 ambiguas y el avión. Es el costo de mantener `truck` completo.
- **La mejor combinación en A deja S1 ≤ 1 % de cota en A, pero no en B con las ambiguas dentro**: con el segmentador barrido es
  `car` 0,65 / `seg` 0,55 (S1 en B 8 de 1 000, cota 1,4 %; sin ambiguas 3 de 975, cota 0,8 %) y con el segmentador fijo en 0,50 es
  `car` 0,75 (7 de 1 000, cota 1,3 %; sin ambiguas 3 de 975, cota 0,8 %). **Sin las ambiguas sí queda bajo el 1 %.** El costo es alto:
  **U3 por alerta de 16,1 % (cota 24,8 %) con el segmentador barrido y de 22,1 % (cota 31,2 %) con el segmentador en 0,50**, contra
  5,6 % hoy; DETAIL rechaza 23 y 34 de 200 y GENERAL 9 y 11 de 200.
- **Lo que paga ese S1 son los camiones y las pickups**: con 0,65 / 0,55 se rechazan 21 de 75 camiones válidos (28 %), 32 de 63
  pickups (51 %) y 54 de 126 pickups en total (43 %); con 0,75, **46 de 63 pickups (73 %)** y 29 de 75 camiones (39 %). v2 detecta
  las pickups y los camiones con menos confianza que a un auto, y como Open Images anota la mayoría de las pickups como `Truck`
  (107 de 126), un umbral alto para limpiar S1 deja afuera justo las camionetas que la gente fotografía. Con los umbrales actuales
  las pickups se rechazan poco: 2 de 63 en B (3,2 %; cota 9,7 %) y 3 de 126 en total; con 0,35, 0 de 63 y 1 de 126. Las vans no
  son el problema (0 de 43 con 0,40; 3 de 43 con 0,75).
- **Si el S1 ≤ 1 % se exige solo sobre las negativas sin las ambiguas** (lectura aparte), la mejor en A con el segmentador
  barrido es `car` 0,35 (con el segmentador casi apagado, 0,95: da lo mismo que 0,35 / 0,50, lo da todo el detector; U3 4,5 %) y con
  el segmentador fijo en 0,50 es `car` 0,50 (U3 10,0 %; rechaza 12 de 63 pickups). Bajar `car` de 0,40 a 0,35 es lo único que
  cambia respecto de hoy en la primera.
- **Colectivos: ya no cuentan.** Con 0,40, **64 de 65 se rechazan (98,5 %)**, 33 de 33 en B; con 0,35, 61 de 65 (93,8 %; 31 de 33 en
  B); con la foto entera, 65 de 65 (0,40) y 64 de 65 (0,35). Los que se escapan los deja pasar el detector con otra clase: con 0,40
  es 1 foto (`van` 0,47 sobre el 62 % de la foto); con 0,35 son 4 (`van` 2, `truck` y `car`). Es el efecto de sacar `bus` del mapeo.
- **Ambulancias: sin conclusión, y no se rechaza.** Solo hay 5 fotos con una ambulancia anotada en toda la partición de prueba
  y 1 admitió una ventana y no estaba en el entrenamiento: **0 de 1 rechazada**. Coincide con lo esperado (la ambulancia se
  entrenó como `van`: el detector la ve como `van` con 0,92), pero con n = 1 no se mide nada. Que dejen de contar exige reentrenar: **no lo hice**.
- **DETAIL es lo que más se rechaza** con cualquier umbral: 4,5 % con los actuales (9 de 200 en B) y 11 a 17 % con los altos. Son
  zonas del vehículo sin una parte reconocible y, con la regla de que DETAIL sigue a GENERAL, se rechazan como `NO_VEHICLE`.

#### Límites

- **Las negativas de Open Images no son exhaustivas**: son fotos sin una clase anotada y varias muestran un vehículo (7 de las 12
  aceptadas en B tienen una rueda, un neumático o un `Vehicle` anotados). La definición del 14/11/2026 las deja dentro, y es la
  pedida (sin exclusiones manuales), así que S1 es pesimista. En el otro sentido son fotos al azar (aviones, barcos, gente,
  comida), no las escenas difíciles que alguien sacaría para engañar al sistema (una calle sin autos, un estacionamiento vacío): S1
  podría ser optimista para esas.
- **Las pickups se identifican por una etiqueta de imagen, no por una caja**: la caja de vehículo que se recorta es la dominante de
  la foto y casi siempre `Truck` (107 de 126). Solo 126 de las 532 fotos con la etiqueta quedaron (las otras estaban en el
  entrenamiento, ya en otro conjunto, o sin una caja dominante que se pueda recortar). Son pocas y las cotas son anchas (63 en B).
- **Las fotos de referencia son de noche y con poca luz**; las de Open Images son sobre todo de día. El rechazo de válidas
  probablemente es **optimista para fotos nocturnas**. No se calibró nada con las de referencia.
- **Los recortes se agrandan** de ≤ 1024 px a 1600 px, así que tienen menos detalle real que una foto de celular; los modelos las ven a
  512 px (detector), pero es un límite.
- **U3 supone independencia** entre las tres fotos y su cota usa las cotas de cada tipo (es conservadora). La alerta de
  `PARKED_ON_GARAGE` tiene una foto de contexto que no está en estos conjuntos.
- **Una sola partición A/B**: otra semilla movería los números (con la misma configuración, A y B ya difieren en 4 a 5
  aceptadas), y la elección en A es optimista.
- **Vans (43 en B), colectivos (33), ambulancias (1) y pickups (63) son pocos**: las cotas son anchas.
- La **ambulancia se entrenó como `van`** y el **camión se mantiene completo** (Open Images anota pickups como `Truck`): son
  limitaciones del mapeo, no medibles con estos conjuntos más allá de lo de arriba.
- No se repitió la latencia: el modelo y el mapeo son los mismos que en la medición del 12/11/2026 (p95 de 4 943 ms por alerta de
  3 fotos), y quitar `bus` no cambia el costo.

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
también aceptaban una rueda o una patente (**la rueda salió de la regla el 11/11/2026**; ver "R1 sin la clase rueda").
Fotos válidas rechazadas (regla C) y negativas que pasan (regla A), medidas todavía con la rueda:

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
también `plate`, con el área mínima de 5 % de la foto; las cifras de esta sección son de cuando también contaba `wheel`) y la re-ID compara el recorte de la caja `car`. El
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

### R1 en tres configuraciones (`presence_configs.py`)

Con las señales que guardó `presence.py` (sin volver a inferir) evalúa R1, por rol, en las configuraciones de abajo, todas
**sin la caja de daño**. **S1** es la cantidad de fotos sin auto que la regla acepta (cota del 95 %); **U3** es la de
primeros planos válidos (PLATE y DETAIL) que rechaza. `--verify` vuelve a medir las fotos aceptadas y comprueba que los
nombres corresponden a las señales (21 re-medidas, 0 diferencias); `--sheets` arma hojas de contacto para mirarlas.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\presence-configs"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/presence_configs.py --report D:\vv-eval\out\presence-sin-segmentador\presence_report.json --fresh-negatives D:\vv-eval\presence-negativas-nuevas --verify --sheets
```

La mejor de **a** y **b** se elige por S1 (fotos sin auto nuevas aceptadas, sumadas por rol) y, si empatan, por U3.

| Configuración | Reglas (PLATE y DETAIL; GENERAL: `car`, y `seg` en b y c) | U3 de 511 (tasa; cota 95 %) | S1 nuevas, de 1 200 (cota 95 %): GENERAL / PLATE y DETAIL | S1 vistas, de 489 (cota) |
|---|---|---|---|---|
| **a** sin segmentador (la oficial) | `car`, `wheel`, `plate` | 29 (5,7 %; 7,7 %) | 15 (1,9 %) / 21 (2,5 %) | 0 (0,6 %) |
| **b** con segmentador | + `seg` | **22 (4,3 %; 6,1 %)** | 15 (1,9 %) / 21 (2,5 %) | 0 (0,6 %) |
| **c1** b sin la clase rueda | `car`, `plate`, `seg` | 22 (4,3 %; 6,1 %) | 15 (1,9 %) / **20 (2,4 %)** | 0 (0,6 %) |
| **c2** b con la rueda a confianza ≥ 0,60 | `car`, `wheel ≥ 0,60`, `plate`, `seg` | 22 (4,3 %; 6,1 %) | 15 (1,9 %) / **20 (2,4 %)** | 0 (0,6 %) |
| referencia: a sin la clase rueda | `car`, `plate` | 30 (5,9 %; 7,9 %) | 15 (1,9 %) / 20 (2,4 %) | 0 (0,6 %) |

Rechazos por rol (de 417, 137 y 374): a, 5 / 1 / 28; b, c1 y c2, 2 / 1 / 21; a sin rueda, 5 / 1 / 29. El barrido de la
rueda (0,50 a 0,90) da lo mismo que c2 desde 0,60: con 0,50 sigue pasando la foto de la rueda (0,52).

**Qué dejó pasar a cada foto sin auto** (las mismas 21 en a, b y c; ninguna configuración acepta una distinta):

| Señal | Fotos | Qué son |
|---|---|---|
| detector, clase `car` | 15 | aviones y un planeador, una lancha y una lancha rápida, una quitanieves, una cabina de avión y una cápsula espacial, una escalera de caracol, un camión (cabina), una topadora; 3 con un auto real al fondo sin anotar (y 1 dudosa) |
| detector, clase `plate` | 5 | un reloj con letras, una historieta con un rótulo, un equipo de radio, un cartel de velocidad y ciclistas |
| detector, clase `wheel` | 1 | sillas de ruedas de básquet |
| segmentador | 0 | nunca dejó pasar una foto sin auto por sí solo (solo suma `seg` donde el detector ya había marcado un auto) |

El segmentador **no suma falsas aceptaciones** y recupera 7 primeros planos (DETAIL 28 → 21, GENERAL 5 → 2). La rueda aporta
casi nada en b: quitarla no rechaza ninguna foto válida más y saca la única negativa que pasaba solo por rueda (21 → 20).
En a, la rueda salva 1 primer plano (30 → 29 rechazados).

**Latencia por alerta** (`cascade.py --long-side 4000`, 20 alertas por escenario, redacción y mosaico activos, una corrida a
la vez, nada más corriendo; AMD Ryzen 5 3600). c1 y c2 usan los mismos modelos que b, así que cuestan lo mismo:

| Escenario | a: mediana / p95 | b: mediana / p95 |
|---|---|---|
| CRASHED de 3 fotos | 5 316 / 6 139 ms | 6 491 / 6 755 ms |
| PARKED_ON_GARAGE de 3 fotos | 5 490 / 6 020 ms | 6 611 / 7 348 ms |
| CRASHED de 5 fotos | 9 526 / 11 355 ms | 11 700 / 12 712 ms |

El segmentador suma unos 1,2 s por alerta en la mediana (la etapa `burst` pasa de ~0,3 a ~1,5 s). Con 20 alertas el p95 es
casi el máximo y varía de una corrida limpia a otra (a, CRASHED de 3 fotos: 6 522 ms en la corrida anterior, 6 139 ms en esta):
conviene mirar la mediana. Una corrida de a hecha mientras corrían otras herramientas salió con p95 8 018 ms y se descartó.
Los 330 y 1 615 ms de media del detector (un pase y con mosaico) de una repetición posterior salen ~13 % por debajo de los 381 y
1 808 ms de arriba: la latencia de esta PC tiene un ruido de ese orden.

### R1 sin la clase rueda: decisor, patente y umbrales (`presence_grid.py`)

**Decisión (11/11/2026): la clase `wheel` salió de la regla R1.** `_has_car` cuenta ahora un `car` del detector (y, en
PLATE y DETAIL, una `plate`), una instancia del segmentador si lo hay y, en DETAIL, el daño si lo hay. Los helpers de
`tools/eval` (`presence.SERVICE_RULE`, `cascade.has_part`) siguen la misma regla; `presence.service_rules(wheel=True)`
reconstruye la anterior para comparar. Las tablas de arriba (a, b, c1, c2) son de antes de esa decisión y se dejan tal cual.
Sin la rueda, la regla oficial (sin segmentador ni daño) rechaza 5 / 1 / 29 válidas (GENERAL / PLATE / DETAIL) y acepta
15 / 20 / 20 de las 1 200 fotos sin auto nuevas.

`presence_grid.py` evalúa, **sin volver a inferir**, todas las combinaciones de tres dimensiones sobre las señales
guardadas (9 694 configuraciones, 20 s):

- **Decisor de "hay un auto":** `det` (solo la clase `car` del detector), `seg` (solo el segmentador), `det ∨ seg`
  (cualquiera de los dos) y `det ∧ seg` (los dos, en la misma foto).
- **Patente como señal de presencia** (la clase `plate` del detector, que se suma al decisor con un OR): `todas` las
  fotos, `PLATE y DETAIL` (la regla del servicio), `solo PLATE` o `ninguna`. No hay un modo "GENERAL solamente".
- **Umbrales solo de R1** (separados de los de redacción): `car` y `plate` del detector de 0,40 a 0,95 de a 0,05, y
  `seg` del segmentador de 0,50 a 0,95. El área mínima sigue en 5 %. Estos umbrales no están en el servicio: es una
  medición, y cambiarlos pide tu aprobación.
- **Mosaico: no es una dimensión.** R1 llama a `VehicleDetector.detect` (un solo pase, `_Run.detect`); el mosaico
  (`detect_plates`) solo se usa al redactar. Se comprobó además sobre las 5 negativas que R1 acepta por la clase patente:
  salen **las 5 sin mosaico** (0,50 a 0,76) y también con mosaico (0,60 a 0,78); ninguna es "solo de mosaico". Con el
  mosaico solo suben algunas confianzas (2e57…: 0,50 → 0,60; a6cf…: 0,74 → 0,78) y aparecen más cajas chicas.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
$env:EVAL_OUT = "D:\vv-eval\out\r1-v1"
uv run --frozen --extra dev --extra tools python tools/eval/presence_grid.py --report D:\vv-eval\out\presence-sin-segmentador\presence_report.json --fresh-negatives D:\vv-eval\presence-negativas-nuevas
```

Deja `r1_grid.json` (resumen), `r1_grid.csv` (las 9 694 filas) y `registro_por_foto.json` (qué se encendió en cada foto,
con su confianza), todo fuera del repo.

#### Qué son S1 y U3 (mi interpretación, una estimación)

El repo no define S1 ni U3; esta es la lectura que se usa acá, y las cifras son **estimaciones** que dependen de ella:

- **S1:** de las fotos sin auto, cuántas acepta la regla. Cada foto se evalúa en cada rol (GENERAL, PLATE, DETAIL) como si
  se hubiera enviado con ese rol; sale una cuenta por rol. Cota superior del 95 % de Clopper-Pearson (3/N con 0 aceptadas).
  La **marca de cumplimiento** es "como máximo 5 de 1 200 en cada uno de los tres roles", que deja la cota en 0,9 %. "Sin
  auto" quiere decir "sin `Car`, `Vehicle registration plate` ni `Wheel` anotados en Open Images", y Open Images no anota
  todo: 3 de las 1 200 nuevas (381d…, 5249…, 6aea…) tienen un auto real al fondo. Por eso se da también la cuenta **sin
  esas 3** (de 1 197).
- **U3:** de los primeros planos válidos (137 recortes de PLATE y 374 fotos de DETAIL de CarDD, 511 en total), cuántos
  rechaza. "Válido" quiere decir que se toma como foto bien tomada del rol; no se revisó una por una.
- El rechazo de GENERAL (de 417) se da aparte. "Total rechazado" suma los tres roles (928 fotos).

#### Punto actual (umbrales 0,40 / 0,40 / 0,50)

| Decisor | Patente como señal | Rechazo de válidas G / P / D (de 417 / 137 / 374) | Cota 95 % | U3 de 511 | S1 nuevas G / P / D (de 1 200) | Cota 95 % | sin las 3 con auto al fondo (de 1 197) | S1 vistas (de 489) | ≤ 5 de 1 200 |
|---|---|---|---|---|---|---|---|---|---|
| det | todas | 4 (1,0 %) / 1 (0,7 %) / 29 (7,8 %) | 2,2 % / 3,4 % / 10,4 % | 30 (5,9 %) | 20 / 20 / 20 | 2,4 % / 2,4 % / 2,4 % | 17 / 17 / 17 | 0 / 0 / 0 | no |
| det | PLATE y DETAIL (actual) | 5 (1,2 %) / 1 (0,7 %) / 29 (7,8 %) | 2,5 % / 3,4 % / 10,4 % | 30 (5,9 %) | 15 / 20 / 20 | 1,9 % / 2,4 % / 2,4 % | 12 / 17 / 17 | 0 / 0 / 0 | no |
| det | solo PLATE | 5 (1,2 %) / 1 (0,7 %) / 30 (8,0 %) | 2,5 % / 3,4 % / 10,7 % | 31 (6,1 %) | 15 / 20 / 15 | 1,9 % / 2,4 % / 1,9 % | 12 / 17 / 12 | 0 / 0 / 0 | no |
| det | ninguna | 5 (1,2 %) / 109 (79,6 %) / 30 (8,0 %) | 2,5 % / 85,1 % / 10,7 % | 139 (27,2 %) | 15 / 15 / 15 | 1,9 % / 1,9 % / 1,9 % | 12 / 12 / 12 | 0 / 0 / 0 | no |
| seg | todas | 27 (6,5 %) / 1 (0,7 %) / 79 (21,1 %) | 8,8 % / 3,4 % / 24,9 % | 80 (15,7 %) | 7 / 7 / 7 | 1,1 % / 1,1 % / 1,1 % | 5 / 5 / 5 | 0 / 0 / 0 | no (sí sin las 3) |
| seg | PLATE y DETAIL (actual) | 32 (7,7 %) / 1 (0,7 %) / 79 (21,1 %) | 10,2 % / 3,4 % / 24,9 % | 80 (15,7 %) | 2 / 7 / 7 | 0,5 % / 1,1 % / 1,1 % | 0 / 5 / 5 | 0 / 0 / 0 | no (sí sin las 3) |
| seg | solo PLATE | 32 (7,7 %) / 1 (0,7 %) / 79 (21,1 %) | 10,2 % / 3,4 % / 24,9 % | 80 (15,7 %) | 2 / 7 / 2 | 0,5 % / 1,1 % / 0,5 % | 0 / 5 / 0 | 0 / 0 / 0 | no (sí sin las 3) |
| seg | ninguna | 32 (7,7 %) / 106 (77,4 %) / 79 (21,1 %) | 10,2 % / 83,1 % / 24,9 % | 185 (36,2 %) | 2 / 2 / 2 | 0,5 % / 0,5 % / 0,5 % | 0 / 0 / 0 | 0 / 0 / 0 | sí |
| det ∨ seg | todas | 1 (0,2 %) / 1 (0,7 %) / 21 (5,6 %) | 1,1 % / 3,4 % / 8,0 % | 22 (4,3 %) | 20 / 20 / 20 | 2,4 % / 2,4 % / 2,4 % | 17 / 17 / 17 | 0 / 0 / 0 | no |
| det ∨ seg | PLATE y DETAIL (actual) | 2 (0,5 %) / 1 (0,7 %) / 21 (5,6 %) | 1,5 % / 3,4 % / 8,0 % | 22 (4,3 %) | 15 / 20 / 20 | 1,9 % / 2,4 % / 2,4 % | 12 / 17 / 17 | 0 / 0 / 0 | no |
| det ∨ seg | solo PLATE | 2 (0,5 %) / 1 (0,7 %) / 21 (5,6 %) | 1,5 % / 3,4 % / 8,0 % | 22 (4,3 %) | 15 / 20 / 15 | 1,9 % / 2,4 % / 1,9 % | 12 / 17 / 12 | 0 / 0 / 0 | no |
| det ∨ seg | ninguna | 2 (0,5 %) / 96 (70,1 %) / 21 (5,6 %) | 1,5 % / 76,5 % / 8,0 % | 117 (22,9 %) | 15 / 15 / 15 | 1,9 % / 1,9 % / 1,9 % | 12 / 12 / 12 | 0 / 0 / 0 | no |
| det ∧ seg | todas | 30 (7,2 %) / 1 (0,7 %) / 87 (23,3 %) | 9,6 % / 3,4 % / 27,1 % | 88 (17,2 %) | 7 / 7 / 7 | 1,1 % / 1,1 % / 1,1 % | 5 / 5 / 5 | 0 / 0 / 0 | no (sí sin las 3) |
| det ∧ seg | PLATE y DETAIL (actual) | 35 (8,4 %) / 1 (0,7 %) / 87 (23,3 %) | 11,0 % / 3,4 % / 27,1 % | 88 (17,2 %) | 2 / 7 / 7 | 0,5 % / 1,1 % / 1,1 % | 0 / 5 / 5 | 0 / 0 / 0 | no (sí sin las 3) |
| det ∧ seg | solo PLATE | 35 (8,4 %) / 1 (0,7 %) / 88 (23,5 %) | 11,0 % / 3,4 % / 27,4 % | 89 (17,4 %) | 2 / 7 / 2 | 0,5 % / 1,1 % / 0,5 % | 0 / 5 / 0 | 0 / 0 / 0 | no (sí sin las 3) |
| det ∧ seg | ninguna | 35 (8,4 %) / 119 (86,9 %) / 88 (23,5 %) | 11,0 % / 91,3 % / 27,4 % | 207 (40,5 %) | 2 / 2 / 2 | 0,5 % / 0,5 % / 0,5 % | 0 / 0 / 0 | 0 / 0 / 0 | sí |

- Con la patente en `ninguna`, PLATE se rechaza en 96 a 119 de 137: los recortes de patente solo se reconocen por la
  clase `plate`, y el decisor de autos casi no ve nada en ellos. Sin esa señal la regla no sirve.
- `seg` solo y `det ∧ seg` bajan S1 pero rechazan 80 a 89 primeros planos, casi todos de DETAIL.
- `det ∨ seg` es la que menos rechaza (22 de 511) con el mismo S1 que el detector solo.
- Con la patente en `todas`, S1 en GENERAL sube de 15 a 20: la patente falsa de un reloj o de un cartel pasa en una foto
  que debería mostrar el auto entero.

#### Mejor punto que cumple S1 ≤ 5 de 1 200

Entre todos los umbrales del barrido, el de menos válidas rechazadas (empate: menos aceptadas):

| Decisor | Patente | Umbrales `car` / `plate` / `seg` | Rechazo de válidas G / P / D | Cota 95 % | Total rechazado (de 928) | S1 nuevas G / P / D | Cota 95 % | S1 vistas |
|---|---|---|---|---|---|---|---|---|
| det | todas | 0,55 / 0,80 / 0,50 | 15 (3,6 %) / 14 (10,2 %) / 52 (13,9 %) | 5,5 % / 15,5 % / 17,2 % | 81 (8,7 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det | PLATE y DETAIL (actual) | 0,55 / 0,80 / 0,50 | 19 (4,6 %) / 14 (10,2 %) / 52 (13,9 %) | 6,6 % / 15,5 % / 17,2 % | 85 (9,2 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det | solo PLATE | 0,55 / 0,80 / 0,50 | 19 (4,6 %) / 14 (10,2 %) / 53 (14,2 %) | 6,6 % / 15,5 % / 17,5 % | 86 (9,3 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det | ninguna | 0,55 / 0,40 / 0,50 | 19 (4,6 %) / 119 (86,9 %) / 53 (14,2 %) | 6,6 % / 91,3 % / 17,5 % | 191 (20,6 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| seg | todas | 0,40 / 0,65 / 0,50 | 27 (6,5 %) / 5 (3,6 %) / 79 (21,1 %) | 8,8 % / 7,5 % / 24,9 % | 111 (12,0 %) | 4 / 4 / 4 | 0,8 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| seg | PLATE y DETAIL (actual) | 0,40 / 0,65 / 0,50 | 32 (7,7 %) / 5 (3,6 %) / 79 (21,1 %) | 10,2 % / 7,5 % / 24,9 % | 116 (12,5 %) | 2 / 4 / 4 | 0,5 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| seg | solo PLATE | 0,40 / 0,65 / 0,50 | 32 (7,7 %) / 5 (3,6 %) / 79 (21,1 %) | 10,2 % / 7,5 % / 24,9 % | 116 (12,5 %) | 2 / 4 / 2 | 0,5 % / 0,8 % / 0,5 % | 0 / 0 / 0 |
| seg | ninguna | 0,40 / 0,40 / 0,50 | 32 (7,7 %) / 106 (77,4 %) / 79 (21,1 %) | 10,2 % / 83,1 % / 24,9 % | 217 (23,4 %) | 2 / 2 / 2 | 0,5 % / 0,5 % / 0,5 % | 0 / 0 / 0 |
| det ∨ seg | todas | 0,55 / 0,80 / 0,50 | 5 (1,2 %) / 13 (9,5 %) / 34 (9,1 %) | 2,5 % / 14,7 % / 11,9 % | 52 (5,6 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∨ seg | PLATE y DETAIL (actual) | 0,55 / 0,80 / 0,50 | 8 (1,9 %) / 13 (9,5 %) / 34 (9,1 %) | 3,4 % / 14,7 % / 11,9 % | 55 (5,9 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∨ seg | solo PLATE | 0,55 / 0,80 / 0,50 | 8 (1,9 %) / 13 (9,5 %) / 34 (9,1 %) | 3,4 % / 14,7 % / 11,9 % | 55 (5,9 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∨ seg | ninguna | 0,55 / 0,40 / 0,50 | 8 (1,9 %) / 102 (74,5 %) / 34 (9,1 %) | 3,4 % / 80,5 % / 11,9 % | 144 (15,5 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∧ seg | todas | 0,40 / 0,65 / 0,50 | 30 (7,2 %) / 5 (3,6 %) / 87 (23,3 %) | 9,6 % / 7,5 % / 27,1 % | 122 (13,1 %) | 4 / 4 / 4 | 0,8 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| det ∧ seg | PLATE y DETAIL (actual) | 0,40 / 0,65 / 0,50 | 35 (8,4 %) / 5 (3,6 %) / 87 (23,3 %) | 11,0 % / 7,5 % / 27,1 % | 127 (13,7 %) | 2 / 4 / 4 | 0,5 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| det ∧ seg | solo PLATE | 0,40 / 0,65 / 0,50 | 35 (8,4 %) / 5 (3,6 %) / 88 (23,5 %) | 11,0 % / 7,5 % / 27,4 % | 128 (13,8 %) | 2 / 4 / 2 | 0,5 % / 0,8 % / 0,5 % | 0 / 0 / 0 |
| det ∧ seg | ninguna | 0,40 / 0,40 / 0,50 | 35 (8,4 %) / 119 (86,9 %) / 88 (23,5 %) | 11,0 % / 91,3 % / 27,4 % | 242 (26,1 %) | 2 / 2 / 2 | 0,5 % / 0,5 % / 0,5 % | 0 / 0 / 0 |

Con la misma regla, pero contando sin las 3 negativas con un auto real al fondo (≤ 5 de 1 197):

| Decisor | Patente | Umbrales `car` / `plate` / `seg` | Rechazo de válidas G / P / D | Cota 95 % | Total rechazado (de 928) | S1 nuevas G / P / D | Cota 95 % | S1 vistas |
|---|---|---|---|---|---|---|---|---|
| det | todas | 0,50 / 0,75 / 0,50 | 10 (2,4 %) / 9 (6,6 %) / 42 (11,2 %) | 4,0 % / 11,2 % / 14,3 % | 61 (6,6 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det | PLATE y DETAIL (actual) | 0,50 / 0,75 / 0,50 | 12 (2,9 %) / 9 (6,6 %) / 42 (11,2 %) | 4,6 % / 11,2 % / 14,3 % | 63 (6,8 %) | 4 / 5 / 5 | 0,8 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det | solo PLATE | 0,50 / 0,75 / 0,50 | 12 (2,9 %) / 9 (6,6 %) / 43 (11,5 %) | 4,6 % / 11,2 % / 14,6 % | 64 (6,9 %) | 4 / 5 / 4 | 0,8 % / 0,9 % / 0,8 % | 0 / 0 / 0 |
| det | ninguna | 0,50 / 0,40 / 0,50 | 12 (2,9 %) / 114 (83,2 %) / 43 (11,5 %) | 4,6 % / 88,2 % / 14,6 % | 169 (18,2 %) | 4 / 4 / 4 | 0,8 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| seg | todas | 0,40 / 0,40 / 0,50 | 27 (6,5 %) / 1 (0,7 %) / 79 (21,1 %) | 8,8 % / 3,4 % / 24,9 % | 107 (11,5 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| seg | PLATE y DETAIL (actual) | 0,40 / 0,40 / 0,50 | 32 (7,7 %) / 1 (0,7 %) / 79 (21,1 %) | 10,2 % / 3,4 % / 24,9 % | 112 (12,1 %) | 0 / 5 / 5 | 0,2 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| seg | solo PLATE | 0,40 / 0,40 / 0,50 | 32 (7,7 %) / 1 (0,7 %) / 79 (21,1 %) | 10,2 % / 3,4 % / 24,9 % | 112 (12,1 %) | 0 / 5 / 0 | 0,2 % / 0,9 % / 0,2 % | 0 / 0 / 0 |
| seg | ninguna | 0,40 / 0,40 / 0,50 | 32 (7,7 %) / 106 (77,4 %) / 79 (21,1 %) | 10,2 % / 83,1 % / 24,9 % | 217 (23,4 %) | 0 / 0 / 0 | 0,2 % / 0,2 % / 0,2 % | 0 / 0 / 0 |
| det ∨ seg | todas | 0,50 / 0,75 / 0,50 | 5 (1,2 %) / 9 (6,6 %) / 28 (7,5 %) | 2,5 % / 11,2 % / 10,1 % | 42 (4,5 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∨ seg | PLATE y DETAIL (actual) | 0,50 / 0,75 / 0,50 | 7 (1,7 %) / 9 (6,6 %) / 28 (7,5 %) | 3,1 % / 11,2 % / 10,1 % | 44 (4,7 %) | 4 / 5 / 5 | 0,8 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∨ seg | solo PLATE | 0,50 / 0,75 / 0,50 | 7 (1,7 %) / 9 (6,6 %) / 28 (7,5 %) | 3,1 % / 11,2 % / 10,1 % | 44 (4,7 %) | 4 / 5 / 4 | 0,8 % / 0,9 % / 0,8 % | 0 / 0 / 0 |
| det ∨ seg | ninguna | 0,50 / 0,40 / 0,50 | 7 (1,7 %) / 99 (72,3 %) / 28 (7,5 %) | 3,1 % / 78,5 % / 10,1 % | 134 (14,4 %) | 4 / 4 / 4 | 0,8 % / 0,8 % / 0,8 % | 0 / 0 / 0 |
| det ∧ seg | todas | 0,40 / 0,40 / 0,50 | 30 (7,2 %) / 1 (0,7 %) / 87 (23,3 %) | 9,6 % / 3,4 % / 27,1 % | 118 (12,7 %) | 5 / 5 / 5 | 0,9 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∧ seg | PLATE y DETAIL (actual) | 0,40 / 0,40 / 0,50 | 35 (8,4 %) / 1 (0,7 %) / 87 (23,3 %) | 11,0 % / 3,4 % / 27,1 % | 123 (13,3 %) | 0 / 5 / 5 | 0,2 % / 0,9 % / 0,9 % | 0 / 0 / 0 |
| det ∧ seg | solo PLATE | 0,40 / 0,40 / 0,50 | 35 (8,4 %) / 1 (0,7 %) / 88 (23,5 %) | 11,0 % / 3,4 % / 27,4 % | 124 (13,4 %) | 0 / 5 / 0 | 0,2 % / 0,9 % / 0,2 % | 0 / 0 / 0 |
| det ∧ seg | ninguna | 0,40 / 0,40 / 0,50 | 35 (8,4 %) / 119 (86,9 %) / 88 (23,5 %) | 11,0 % / 91,3 % / 27,4 % | 242 (26,1 %) | 0 / 0 / 0 | 0,2 % / 0,2 % / 0,2 % | 0 / 0 / 0 |

Cumplir S1 ≤ 1 % cuesta pasar de 23 o 24 a **52 a 55** válidas rechazadas de 928 con `det ∨ seg` (patente en `todas` o en
`PLATE y DETAIL`, `car` ≥ 0,55 y `plate` ≥ 0,80) y de 34 o 35 a **81 a 85** con el detector solo. La frontera completa
(umbrales de cada configuración, de menor a mayor rechazo) está en `r1_grid.json` (`frontera`). En esos puntos se rechazan
13 o 14 de los 137 recortes de PLATE y de 34 a 52 de las 374 fotos de DETAIL (con la patente en 0,80 y `car` en 0,55).

#### Registro por foto: detector contra segmentador (umbrales actuales)

| Conjunto | Fotos | det sí / seg sí | det sí / seg no | det no / seg sí | det no / seg no | det solo por patente |
|---|---|---|---|---|---|---|
| válidas GENERAL | 417 | 382 | 30 | 3 | 2 | 0 |
| válidas PLATE | 137 | 31 | 105 | 0 | 1 | 108 |
| válidas DETAIL | 374 | 287 | 58 | 8 | 21 | 1 |
| sin auto, vistas | 489 | 0 | 0 | 0 | 489 | 0 |
| sin auto, nuevas | 1200 | 2 | 18 | 0 | 1180 | 5 |

"det" es el auto del detector y, en PLATE, DETAIL y las negativas, también su patente; "det solo por patente" cuenta
las fotos donde el detector solo marcó una patente. El segmentador **nunca se enciende solo en una foto sin auto**
(0 en "det no / seg sí" en las 1 689 negativas), y en las válidas rescata 11 fotos (3 de GENERAL y 8 de DETAIL).

#### De las 15 negativas que pasan por la clase auto, ¿el segmentador también vio un auto?

| Foto | Confianza del auto | Segmentador: mejor instancia (cualquier tamaño) | Segmentador: cuenta para R1 (≥ 0,50 y ≥ 5 %) |
|---|---|---|---|
| `05b216c9501d4389` | 0,49 | 0,07 | no |
| `139b58bf9981102e` | 0,45 | 0,64 | no |
| `145225d35029ace7` | 0,79 | 0,91 | no |
| `1d3e7fa4223dca3e` | 0,46 | 0,39 | no |
| `1fa203101e83d731` | 0,43 | 0,02 | no |
| `373bc1f6e343749d` | 0,65 | 0,13 | no |
| `381d3671020dca70` (auto real al fondo) | 0,73 | 0,91 | **sí** (0,91) |
| `52491dfdb5474272` (auto real al fondo) | 0,52 | 0,89 | no |
| `6aea64be3ea593ec` (auto real al fondo) | 0,68 | 0,83 | **sí** (0,83) |
| `72668dba3d8886b8` | 0,48 | 0,02 | no |
| `85be4b924320e539` | 0,54 | 0,02 | no |
| `9b59c0e2c3fe60da` | 0,41 | 0,12 | no |
| `a21c9ae16959fbae` | 0,47 | 0,06 | no |
| `a5fd24fe4683492b` | 0,40 | 0,14 | no |
| `adb32fe9df8bea34` | 0,68 | 0,08 | no |

El segmentador cuenta (con el 5 % de área) en **2 de 15**, y las dos son de las 3 con un auto real al fondo. En 3 más
vio una instancia con confianza ≥ 0,50 pero de menos del 5 % de la foto (5249…, la tercera con un auto real, y 139b… y
145225…); en las otras 10 no vio nada relevante. Dicho de otro modo: en las negativas con un auto real el segmentador
acompaña al detector, y en las que no lo tienen no lo acompaña.

#### Latencia por alerta de 3 fotos

La regla R1 no cambia la latencia (compara tres números por foto): la cambian los modelos que corren. Las configuraciones
`det` usan los modelos de **a**; `seg`, `det ∨ seg` y `det ∧ seg` necesitan el segmentador y usan los de **b**. Son las
mismas corridas limpias de arriba (`cascade.py --long-side 4000`, 20 alertas por escenario, una a la vez):

| Modelos | CRASHED de 3 fotos: mediana / p95 | PARKED_ON_GARAGE de 3 fotos: mediana / p95 |
|---|---|---|
| sin segmentador (`det`) | 5 316 / 6 139 ms | 5 490 / 6 020 ms |
| con segmentador (`seg`, `det ∨ seg`, `det ∧ seg`) | 6 491 / 6 755 ms | 6 611 / 7 348 ms |

Con 20 alertas el p95 es casi el máximo y varía de una corrida limpia a otra: conviene mirar la mediana.

### R1 con la patente que ya no valida sola: P1 contra P2, por zoom (`plate_zoom.py`, `plate_rule.py`)

**Decisión (11/11/2026):** una patente sola **no valida** una foto, porque alcanza con una patente arrancada o
impresa para hacer pasar cualquier imagen. En GENERAL y DETAIL la patente **dejó de contar** como presencia
(`_has_car`: solo `car` del detector, auto del segmentador y, en DETAIL, daño). En PLATE hay dos variantes, que
se miden acá sin aplicar ninguna (el servicio sigue con P1):

- **P1** (la actual): un auto **o** una patente.
- **P2**: una patente **y** un auto (det ∨ seg), con el umbral de auto en barrido desde 0,30.

**Cómo se armó el set de primeros planos realistas** (`plate_zoom.py`). Los recortes de siempre (137, de
`presence.py`) llevan medio ancho de margen: la patente ocupa el 50 % del ancho, mucho más ajustado que lo que
saca la app. El set nuevo sale de **imágenes completas**: las del dataset de Roboflow "patentes" (todas sus
particiones, 135 imágenes) y las fotos con patente de la **partición de prueba de Open Images** de v1 (204 fotos).
De cada patente anotada (lado menor ≥ 16 px: 252 en total) se recorta, **en memoria**, una ventana **4:3
horizontal centrada en la patente** cuyo ancho es `ancho_de_la_patente / zoom`, con zoom 25 %, 40 % y 60 % (la
patente ocupa ese porcentaje del ancho de la foto). Si la ventana se sale de la imagen se la corre hasta que
entre; si no cabe, o queda de menos de 128 px de ancho, esa patente no entra. **Se usan solo las patentes que
admiten los tres zooms**, para que la comparación entre niveles sea sobre las mismas: **151 patentes** (84 de
Roboflow y 67 de Open Images), 151 fotos por zoom. Los identificadores no llevan nombres de archivo. Límites: 151
patentes dan cotas anchas (de 5 a 9 puntos), las fotos de 128 a 400 px de ancho son de menor resolución que las
de la app, y la ventana siempre es 4:3 horizontal y centrada: una app real puede inclinar, descentrar y variar.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"; $env:EVAL_DATA_DIR = "D:\vv-eval"; $env:MODELS_DIR = "models\weights"
$env:EVAL_OUT = "D:\vv-eval\out\zoom-v1"
uv run --frozen --extra dev --extra tools python tools/eval/plate_zoom.py   # ~8 min con v1; --detector <modelo> para otro
$env:EVAL_OUT = "D:\vv-eval\out\r1-regla-v1"
uv run --frozen --extra dev --extra tools python tools/eval/plate_rule.py --report D:\vv-eval\out\presence-sin-segmentador\presence_report.json --zoom-report D:\vv-eval\out\zoom-v1\plate_zoom_report.json --fresh-negatives D:\vv-eval\presence-negativas-nuevas
```

`plate_rule.py` no vuelve a inferir: barre decisor (`det`, `seg`, `det ∨ seg`, `det ∧ seg`), variante y umbrales
de auto, patente y segmentador de 0,30 a 0,95 (11 760 configuraciones). Son umbrales **solo de la medición**:
ninguno se aplica. El área mínima de cada hallazgo es la del servicio (5 % de la foto).

#### Rechazo de válidas por tipo y zoom (umbrales actuales 0,40 / 0,40 / 0,50; entre paréntesis la cota del 95 %)

| Decisor | Regla | GENERAL (de 417) | PLATE 50 % (de 137) | PLATE 25 % (de 151) | PLATE 40 % (de 151) | PLATE 60 % (de 151) | DETAIL (de 374) |
|---|---|---|---|---|---|---|---|
| det | P1 | 5 (≤ 2,5 %) | 1 (≤ 3,4 %) | 8 (≤ 9,4 %) | 3 (≤ 5,1 %) | 4 (≤ 6,0 %) | 30 (≤ 10,7 %) |
| det | P2 | 5 (≤ 2,5 %) | 109 (≤ 85,1 %) | 129 (≤ 89,9 %) | 47 (≤ 37,9 %) | 99 (≤ 72,0 %) | 30 (≤ 10,7 %) |
| det ∨ seg | P1 | 2 (≤ 1,5 %) | 1 (≤ 3,4 %) | 3 (≤ 5,1 %) | 3 (≤ 5,1 %) | 4 (≤ 6,0 %) | 21 (≤ 8,0 %) |
| det ∨ seg | P2 | 2 (≤ 1,5 %) | 96 (≤ 76,5 %) | 128 (≤ 89,4 %) | 33 (≤ 28,1 %) | 85 (≤ 63,1 %) | 21 (≤ 8,0 %) |

- **P2 rechaza casi todos los primeros planos de patente**, y por dos motivos distintos. En los **ajustados (50 %)
  y en el 60 %** no hay auto que ver: el detector no encuentra un auto en una foto que es casi solo la patente
  (96 y 85 de 151 con det ∨ seg). En el **25 %** sí hay auto, pero la patente ocupa cerca del **3 % del área** de
  la foto y el mínimo del servicio es **5 %**: no cuenta (128 de 151). Con un mínimo de 1 % para la patente, P2 en
  el 25 % baja de 128 a 13 rechazadas (última tabla), pero eso es otro cambio que nadie pidió. P2 solo anda bien
  en el término medio (40 %: 33 de 151).
- **P1** rechaza entre 1 y 4 de 151 con det ∨ seg a cualquier zoom, porque la patente sola (o el auto solo)
  alcanza.

#### Fotos sin auto aceptadas y U3 por alerta (mismos umbrales)

La columna "nuevas" cuenta las aceptadas de 1 200 en GENERAL / PLATE / DETAIL (cada foto evaluada como si se
hubiera enviado en ese rol); "sin las 3" las cuenta sin las 3 que tienen un auto real al fondo (de 1 197); en las
489 vistas, 0 en todas. **U3 por alerta** = `1 - (1 - pG)(1 - pP)(1 - pD)` con una foto de cada rol (GENERAL +
PLATE + un DETAIL; una alerta `PARKED_ON_GARAGE` trae CONTEXT en vez de DETAIL y no está medida), con la PLATE del
zoom indicado.

| Decisor | Regla | nuevas G / P / D | sin las 3 | U3 por alerta (PLATE 25 / 40 / 60 %) | S1 ≤ 5 |
|---|---|---|---|---|---|
| det | P1 | 15 / 20 / 15 | 12 / 17 / 12 | 13,9 % · 10,9 % · 11,5 % | no |
| det | P2 | 15 / 0 / 15 | 12 / 0 / 12 | 86,8 % · 37,4 % · 68,7 % | no |
| det ∨ seg | P1 | 15 / 20 / 15 | 12 / 17 / 12 | 7,9 % · 7,9 % · 8,6 % | no |
| det ∨ seg | P2 | 15 / 0 / 15 | 12 / 0 / 12 | 85,7 % · 26,6 % · 58,9 % | no |

**El supuesto de U3.** Se supone que las fotos de una misma alerta **fallan de forma independiente**. Es una
estimación, no una medición: no hay alertas reales (las tres fotos salen de conjuntos distintos), y en la
práctica las fallas de una alerta pueden estar correlacionadas (mismo auto, misma luz), lo que haría la unión
real **menor**. La cota usa las cotas de cada foto en la misma fórmula: es conservadora, no un intervalo exacto.

#### Configuraciones que cumplen S1 (≤ 5 de 1 200 en cada rol) y su costo

La mejor de cada decisor y variante: la que minimiza el U3 medio de los tres zooms entre las que cumplen S1
(optimizar un zoom solo mueve un umbral; todas están en `plate_rule.json`). Umbrales `car` / `plate` / `seg`.
Columnas de rechazo: rechazadas de 417 / 137 / 151 / 151 / 151 / 374.

| Decisor | Regla | Umbrales | GENERAL | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % | DETAIL | nuevas G/P/D | U3 por alerta 25 / 40 / 60 % |
|---|---|---|---|---|---|---|---|---|---|---|
| det | P1 | 0,55 / 0,80 / – | 19 | 14 | 17 | 19 | 27 | 53 | 5 / 5 / 5 | 27,3 · 28,4 · 32,7 % |
| det | P2 | 0,55 / 0,30 / – | 19 | 119 | 130 | 65 | 116 | 53 | 5 / 0 / 5 | 88,6 · 53,3 · 81,0 % |
| det ∨ seg | P1 | 0,55 / 0,80 / 0,30 | 4 | 10 | 6 | 6 | 18 | 27 | 5 / 5 / 5 | **11,8 · 11,8 · 19,1 %** |
| det ∨ seg | P2 | 0,55 / 0,30 / 0,30 | 4 | 81 | 127 | 29 | 74 | 27 | 5 / 0 / 5 | 85,4 · 25,8 · 53,1 % |

En las 489 vistas, 0 aceptadas. Sin las 3 con auto real al fondo (≤ 5 de 1 197) la mejor es det ∨ seg con P1 y
0,55 / 0,65 / 0,30: rechaza 4 / 5 / 4 / 3 / 8 / 27 y da un U3 de 10,5 · 9,9 · 13,0 %. Notas:

- **Costo de cumplir S1, contra el punto actual (det ∨ seg, P1):** el U3 por alerta pasa de ~8 % a ~12 % (zoom 25
  y 40 %) o 19 % (60 %), y DETAIL de 21 a 27 rechazadas de 374. Con det solo (sin segmentador) el costo es mucho
  mayor: 27 a 33 % por alerta.
- **P2 no se recupera con ningún umbral:** cumple S1 sin esfuerzo (0 negativas aceptadas en PLATE) pero rechaza el
  85 % de las alertas con un PLATE al 25 % y la mitad al 60 %, porque el costo no viene del umbral sino de que el
  auto no está en la foto (o de que la patente es chica para el mínimo de área).
- El umbral de segmentador 0,30 que eligen las mejores configuraciones está **por debajo del 0,50 vigente**: es la
  primera vez que se barre por debajo. El segmentador no se enciende solo en ninguna negativa (ver arriba), así
  que bajar su umbral casi no cuesta falsos de S1.
- El criterio S1 ≤ 5 y la lectura de U3 son mi interpretación (ver más arriba). Los umbrales se eligieron sobre
  las mismas negativas con las que se cuentan: es una cota optimista de lo que daría con fotos nuevas.

#### Sensibilidad del área mínima de la patente (det ∨ seg, umbrales actuales; rechazadas de 137 / 151)

| Área mínima de la patente | Regla | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % |
|---|---|---|---|---|---|
| 5 % (servicio) | P1 | 1 | 3 | 3 | 4 |
| 5 % (servicio) | P2 | 96 | 128 | 33 | 85 |
| 1 % | P1 | 1 | 1 | 2 | 4 |
| 1 % | P2 | 96 | 13 | 30 | 85 |
| 0 % | P1 | 1 | 1 | 2 | 4 |
| 0 % | P2 | 96 | 12 | 30 | 85 |

Es un diagnóstico para explicar por qué P2 rechaza el 25 %; no se propone cambiar `PRESENCE_MIN_AREA`.

### Detector v2 contra v1

El detector v2 (`rfdetr_openimages_v2`, ver [tools/training/detector_v2/README.md](../training/detector_v2/README.md)) se
compara con v1 sobre **la prueba de v1**: las 204 fotos del split `test` de Open Images que ninguno de los dos vio. Mismas
condiciones que arriba: el adaptador real, los umbrales del servicio (autos 0,40, patentes 0,20), cada foto reducida a
1 024 px, IoU 0,5. Las cuatro corridas (v1 y v2, con y sin `PLATE_TILING`) se hicieron seguidas, una por vez, con la PC
sin otras cargas.

```powershell
$env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"; $env:MODELS_DIR = "models\weights"; $env:EVAL_OUT = "D:\vv-eval\out\cmp-v2-sin"
uv run --frozen --extra dev --extra tools python tools/eval/detector.py --split test --model rfdetr_openimages_v2 --dump-fp
# con mosaico: --plate-tiling (EVAL_OUT distinto). Para v1: --model rfdetr_openimages
```

**Cómo leerla.** La verdad de la prueba solo anota `Car`, no `Truck`, `Bus` ni `Van`: v2 detecta esas clases como `car`
(son presencia), así que sus camiones, vans y colectivos cuentan como falsos positivos de la precisión **oficial**. Por eso
se da también la precisión corregida (abajo). v2 no tiene rueda ni rostro: solo se comparan autos y patentes.

| Métrica | v1, sin mosaico | v2, sin mosaico | v1, con mosaico | v2, con mosaico |
|---|---|---|---|---|
| **Auto**: recall | 0,898 | 0,881 | 0,898 | 0,881 |
| Auto: precisión oficial | 0,747 | 0,754 | 0,747 | 0,754 |
| Auto: AP50 | 0,897 | 0,873 | 0,897 | 0,873 |
| Auto: recall en chicos (< 32 px) | 0,333 (6 cajas) | 0,167 | 0,333 | 0,167 |
| **Patente**: recall | 0,830 | 0,807 | 0,904 | 0,874 |
| Patente: precisión oficial | 0,737 | 0,727 | 0,632 | 0,634 |
| Patente: AP50 | 0,787 | 0,809 | 0,831 | 0,835 |
| Patente: recall en chicas | 0,608 | 0,627 | 0,765 | 0,745 |
| **Auto + patente (micro)**: recall | 0,873 | 0,855 | 0,900 | 0,879 |
| Auto + patente: precisión oficial | 0,744 | 0,745 | 0,702 | 0,707 |
| Auto + patente: AP50 | 0,851 | 0,832 | 0,866 | 0,852 |
| Falsos positivos (auto / patente) | 74 / 40 | 70 / 41 | 74 / 71 | 70 / 68 |
| **Latencia por foto** (media · mediana · p95, ms) | 324 · 315 · 394 | 334 · 317 · 430 | 1 569 · 1 616 · 1 800 | 1 620 · 1 675 · 1 864 |

(Cajas reales: 244 autos y 135 patentes; el auto no cambia con el mosaico, que solo afecta a las patentes. Latencia en
CPU, ONNX Runtime, AMD Ryzen 5 3600; sin contar decodificación ni reducción.)

**Precisión corregida, por revisión manual de un solo revisor** (`fp_audit.py`, junto a la oficial, que no se
reemplaza). La corregida de v1 es la ya revisada. Para v2 **reutilicé la revisión de v1** en cada falso positivo que
coincide (misma foto, misma clase, IoU ≥ 0,5: 73 de 111 sin mosaico y 88 de 138 con mosaico) y **revisé a mano los
nuevos** (38 y 50, con las cajas en hojas de contacto). La corregida cuenta como acierto un auto o patente real que el
dataset no anotó; para v2 se da además una columna donde los camiones, vans y colectivos (`otro_vehiculo`) cuentan como
acierto, porque en v2 son presencia de vehículo.

| Auto + patente (micro) | Oficial | Corregida | Corregida con los dudosos | Corregida con otros vehículos como acierto |
|---|---|---|---|---|
| v1, sin mosaico | 0,744 | 0,888 | 0,924 | 0,910 |
| v2, sin mosaico | 0,745 | 0,874 | 0,906 | **0,906** |
| v1, con mosaico | 0,702 | 0,844 | 0,889 | 0,864 |
| v2, con mosaico | 0,707 | 0,843 | 0,900 | **0,873** |

Por clase, sin mosaico: **auto** corregida 0,904 (v1) y 0,902 (v2); con otros vehículos como acierto 0,939 y **0,951**.
**Patente** corregida 0,855 (v1) y 0,820 (v2).

Límites: es **un solo revisor**, y los veredictos de v2 no son idénticos en criterio a los de v1: sobre los mismos tipos
de caja marqué más "dudosos" (9 contra 5 patentes sin mosaico; 22 contra 11 con mosaico), así que la cifra "con dudosos"
es la más comparable. Los falsos positivos que heredé de v1 conservan su veredicto aunque v2 los hubiera visto distinto.
Las diferencias de unas décimas entre v1 y v2 están dentro de lo que mueve esa revisión.

**Lectura.** v2 es **un poco peor que v1 en recall** (auto 0,881 contra 0,898, patente 0,807 contra 0,830; micro 0,855
contra 0,873) y en AP50 de auto (0,873 contra 0,897), y **un poco mejor en AP50 de patente** (0,809 contra 0,787). Su
precisión oficial es igual (0,745 contra 0,744) y con camiones, vans y colectivos como aciertos la corregida de auto sube
(0,951 contra 0,939). El recall de autos chicos baja de 0,33 a 0,17, pero son 6 cajas. La latencia es la misma
(334 contra 324 ms; 1 620 contra 1 569 ms con mosaico). Estas diferencias son chicas para 204 fotos y 379 cajas; no
alcanzan para decir que uno detecta mejor que el otro en esta prueba. Lo que v2 agrega no se ve acá: autos que no son
`Car` (camiones, vans, colectivos) y las confusoras. Eso se mide en R1, abajo.

### R1 con el detector v2: la regla de patente, v1 y v2 lado a lado

Misma medición que la sección anterior (`presence.py`, `plate_zoom.py` y `plate_rule.py`, una por vez), repetida con
`--detector rfdetr_openimages_v2`. En v2 **"auto" es cualquier clase de presencia** (`car`, `truck`, `van`, `bus`: el
adaptador devuelve una caja `car`); v2 no tiene rueda, así que R1 nunca la usó. Las señales de v2 están en
`D:\vv-eval\out\presence-v2` y `zoom-v2`, el análisis en `r1-regla-v2`.

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\presence-v2"
uv run --frozen --extra dev --extra tools python tools/eval/presence.py --detector rfdetr_openimages_v2 --fresh-negatives D:\vv-eval\presence-negativas-nuevas --fresh-limit 1200   # ~1 h
$env:EVAL_OUT = "D:\vv-eval\out\zoom-v2"
uv run --frozen --extra dev --extra tools python tools/eval/plate_zoom.py --detector rfdetr_openimages_v2
$env:EVAL_OUT = "D:\vv-eval\out\r1-regla-v2"
uv run --frozen --extra dev --extra tools python tools/eval/plate_rule.py --report D:\vv-eval\out\presence-v2\presence_report.json --zoom-report D:\vv-eval\out\zoom-v2\plate_zoom_report.json --fresh-negatives D:\vv-eval\presence-negativas-nuevas --exclude-negatives D:\vv-eval\out\neg-v2\vehiculos_reales.json
```

#### Las "negativas" de v2 no son todas sin vehículo (revisión a mano)

Las negativas (489 vistas y 1 200 nuevas) son fotos de Open Images **sin `Car`, `Wheel` ni patente anotados**. No excluyen
`Truck`, `Bus` ni `Van`, que para v2 **sí son presencia**. Con los umbrales actuales v2 acepta 16 de las 1 200 (v1, 15) y 3
de las 489 (v1, 0). Revisé a mano las **22 fotos** donde v2 ve un auto con confianza ≥ 0,30 (área ≥ 5 %), con la misma
regla que usé para los falsos positivos (un solo revisor):

| Qué hay en la foto | Fotos |
|---|---|
| **Un vehículo real** (camión, colectivo, van, ambulancia, auto parcial o al fondo) | 11 |
| Dudoso (una mujer dentro de un auto, gente arriba de un colectivo de dos pisos, un auto de carrera, un quitanieves con un auto cubierto de nieve al fondo) | 4 |
| Error de v2 (un vehículo de nieve, un blindado, un autoelevador, una topadora, un flipper, un avión, una escalera caracol) | 7 |

Las 11 con un vehículo real **dejan de ser "sin auto"**: v2 hace bien en aceptarlas. Se excluyen de las columnas "sin las
excluidas" (que para v1 ya excluían las 3 con un auto real al fondo, dentro de estas 11) y de `cumple_s1_sin_fondo`. Las 4
dudosas y las 7 errores **se siguen contando** como aceptadas. **Esta exclusión es una decisión mía sobre 11 fotos, de un
solo revisor**, y favorece a v2: sin ella, v2 no cumple S1 con ninguna configuración razonable (abajo, "S1 literal").
La lista (`vehiculos_reales.json`) queda fuera del repo con las fotos. Se aplica igual a v1, que acepta solo 4 de
esas 11: v1 hoy **rechaza** como `NO_VEHICLE` la mayoría de los camiones, vans y colectivos reales.

#### Rechazo de válidas por tipo y zoom (umbrales actuales 0,40 / 0,40 / 0,50; cota del 95 % entre paréntesis)

**v1** (`rfdetr_openimages`):

| Decisor | Regla | GENERAL (417) | PLATE 50 % (137) | PLATE 25 % (151) | PLATE 40 % (151) | PLATE 60 % (151) | DETAIL (374) |
|---|---|---|---|---|---|---|---|
| det | P1 | 5 (≤ 2,5 %) | 1 (≤ 3,4 %) | 8 (≤ 9,4 %) | 3 (≤ 5,1 %) | 4 (≤ 6,0 %) | 30 (≤ 10,7 %) |
| det | P2 | 5 (≤ 2,5 %) | 109 (≤ 85,1 %) | 129 (≤ 89,9 %) | 47 (≤ 37,9 %) | 99 (≤ 72,0 %) | 30 (≤ 10,7 %) |
| det ∨ seg | P1 | 2 (≤ 1,5 %) | 1 (≤ 3,4 %) | 3 (≤ 5,1 %) | 3 (≤ 5,1 %) | 4 (≤ 6,0 %) | 21 (≤ 8,0 %) |
| det ∨ seg | P2 | 2 (≤ 1,5 %) | 96 (≤ 76,5 %) | 128 (≤ 89,4 %) | 33 (≤ 28,1 %) | 85 (≤ 63,1 %) | 21 (≤ 8,0 %) |

**v2** (`rfdetr_openimages_v2`):

| Decisor | Regla | GENERAL (417) | PLATE 50 % (137) | PLATE 25 % (151) | PLATE 40 % (151) | PLATE 60 % (151) | DETAIL (374) |
|---|---|---|---|---|---|---|---|
| det | P1 | 5 (≤ 2,5 %) | 9 (≤ 11,2 %) | 4 (≤ 6,0 %) | 7 (≤ 8,5 %) | 13 (≤ 13,3 %) | 30 (≤ 10,7 %) |
| det | P2 | 5 (≤ 2,5 %) | 91 (≤ 73,1 %) | 129 (≤ 89,9 %) | 48 (≤ 38,6 %) | 88 (≤ 65,0 %) | 30 (≤ 10,7 %) |
| det ∨ seg | P1 | 3 (≤ 1,8 %) | 9 (≤ 11,2 %) | 4 (≤ 6,0 %) | 6 (≤ 7,7 %) | 12 (≤ 12,6 %) | 22 (≤ 8,3 %) |
| det ∨ seg | P2 | 3 (≤ 1,8 %) | 85 (≤ 69,0 %) | 129 (≤ 89,9 %) | 40 (≤ 33,1 %) | 81 (≤ 60,6 %) | 22 (≤ 8,3 %) |

P2 sigue rechazando casi todos los primeros planos con v2, por las mismas razones que con v1 (no hay auto en una foto que
es la patente, y la patente al 25 % no llega al 5 % del área). **v2 con P1 rechaza más primeros planos de patente
ajustados que v1** (9 contra 1 de 137 al 50 %; 12 contra 4 de 151 al 60 %): le cuesta más encontrar una patente cuando casi
llena la foto (no investigué la causa). En GENERAL rechaza 3 contra 2 de 417 y en DETAIL
22 contra 21 de 374: iguales.

#### Fotos sin auto aceptadas y U3 por alerta (mismos umbrales)

**v1:**

| Decisor | Regla | nuevas G / P / D (de 1 200) | sin las excluidas | vistas (de 489) | U3 por alerta, PLATE 25 / 40 / 60 % |
|---|---|---|---|---|---|
| det | P1 | 15 / 20 / 15 | 11 / 16 / 11 | 0 / 0 / 0 | 13,9 · 10,9 · 11,5 % |
| det | P2 | 15 / 0 / 15 | 11 / 0 / 11 | 0 / 0 / 0 | 86,8 · 37,4 · 68,7 % |
| det ∨ seg | P1 | 15 / 20 / 15 | 11 / 16 / 11 | 0 / 0 / 0 | 7,9 · 7,9 · 8,6 % |
| det ∨ seg | P2 | 15 / 0 / 15 | 11 / 0 / 11 | 0 / 0 / 0 | 85,7 · 26,6 · 58,9 % |

**v2:**

| Decisor | Regla | nuevas G / P / D (de 1 200) | sin las excluidas | vistas (de 489) | U3 por alerta, PLATE 25 / 40 / 60 % |
|---|---|---|---|---|---|
| det | P1 | 16 / 16 / 16 | 7 / 7 / 7 | 3 / 3 / 3 | 11,5 · 13,3 · 17,0 % |
| det | P2 | 16 / 0 / 16 | 7 / 0 / 7 | 3 / 0 / 3 | 86,8 · 38,0 · 62,1 % |
| det ∨ seg | P1 | 16 / 16 / 16 | 7 / 7 / 7 | 3 / 3 / 3 | 9,0 · 10,3 · 14,0 % |
| det ∨ seg | P2 | 16 / 0 / 16 | 7 / 0 / 7 | 3 / 0 / 3 | 86,4 · 31,3 · 56,7 % |

(Columna "sin las excluidas": sin las 11 fotos con un vehículo real. U3 por alerta: GENERAL + PLATE + un DETAIL, con
independencia entre fotos; es una estimación, ver el supuesto arriba.)

#### Configuraciones que cumplen S1 (≤ 5 de 1 200 sin las excluidas) y su costo

La mejor de cada decisor y variante: la que minimiza el U3 medio de los tres zooms (las demás están en `plate_rule.json`).
Columnas de rechazo: rechazadas de 417 / 137 / 151 / 151 / 151 / 374.

**v1:**

| Decisor | Regla | Umbrales car / plate / seg | GENERAL | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % | DETAIL | nuevas G/P/D | U3 por alerta 25 / 40 / 60 % |
|---|---|---|---|---|---|---|---|---|---|---|
| det | P1 | 0,50 / 0,75 / – | 12 | 9 | 15 | 10 | 18 | 43 | 4 / 5 / 4 | 22,6 · 19,7 · 24,3 % |
| det | P2 | 0,50 / 0,30 / – | 12 | 114 | 128 | 60 | 109 | 43 | 4 / 0 / 4 | 86,9 · 48,2 · 76,1 % |
| det ∨ seg | P1 | 0,55 / 0,65 / 0,30 | 4 | 5 | 4 | 3 | 8 | 27 | 3 / 5 / 3 | 10,5 · 9,9 · 13,0 % |
| det ∨ seg | P2 | 0,50 / 0,30 / 0,30 | 4 | 80 | 127 | 27 | 72 | 24 | 4 / 0 / 4 | 85,3 · 23,9 · 51,5 % |

**v2:**

| Decisor | Regla | Umbrales car / plate / seg | GENERAL | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % | DETAIL | nuevas G/P/D | U3 por alerta 25 / 40 / 60 % |
|---|---|---|---|---|---|---|---|---|---|---|
| det | P1 | 0,45 / 0,30 / – | 6 | 4 | 4 | 5 | 9 | 31 | 5 / 5 / 5 | 12,0 · 12,6 · 15,0 % |
| det | P2 | 0,45 / 0,30 / – | 6 | 94 | 130 | 48 | 87 | 31 | 5 / 0 / 5 | 87,4 · 38,3 · 61,7 % |
| det ∨ seg | P1 | 0,45 / 0,30 / 0,30 | 1 | 4 | 4 | 3 | 8 | 19 | 5 / 5 / 5 | 7,8 · 7,2 · 10,3 % |
| det ∨ seg | P2 | 0,45 / 0,30 / 0,30 | 1 | 73 | 129 | 36 | 68 | 19 | 5 / 0 / 5 | 86,2 · 27,9 · 47,9 % |

- **La mejor configuración de cada detector, det ∨ seg con P1, da un U3 por alerta de 10,5 · 9,9 · 13,0 % con v1 y de
  7,8 · 7,2 · 10,3 % con v2** (zoom 25 · 40 · 60 %). Con v2 alcanza con `car` ≥ 0,45, `plate` ≥ 0,30 y `seg` ≥ 0,30 (v1
  pide 0,55 / 0,65 / 0,30). Es una mejora de ~3 puntos que sale de rechazar menos GENERAL (1 contra 4 de 417) y menos
  DETAIL (19 contra 27 de 374), y se apoya en la exclusión de 11 fotos y en umbrales elegidos sobre las mismas negativas
  con las que se miden: es una cota optimista, no una promesa.
- **Sin segmentador (solo det)**: v2 da 12,0 · 12,6 · 15,0 % y v1 22,6 · 19,7 · 24,3 %. Ahí la diferencia es más grande:
  v2 rechaza 6 contra 12 de GENERAL y 31 contra 43 de DETAIL.
- **P2 no cumple en la práctica con ningún detector**: cumple S1 (0 negativas aceptadas en PLATE) pero rechaza 85 a 87 % de
  las alertas con una PLATE al 25 %, 24 a 48 % al 40 % y 48 a 76 % al 60 %.
- El umbral de segmentador de 0,30 está por debajo del 0,50 vigente (ver la sección anterior).

**S1 literal (≤ 5 de 1 200 contando todas, también las de vehículo real).** Para ver cuánto depende v2 de la revisión:

**v1:**

| Decisor | Regla | Umbrales car / plate / seg | GENERAL | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % | DETAIL | nuevas G/P/D | U3 por alerta 25 / 40 / 60 % |
|---|---|---|---|---|---|---|---|---|---|---|
| det | P1 | 0,55 / 0,80 / – | 19 | 14 | 17 | 19 | 27 | 53 | 5 / 5 / 5 | 27,3 · 28,4 · 32,7 % |
| det | P2 | 0,55 / 0,30 / – | 19 | 119 | 130 | 65 | 116 | 53 | 5 / 0 / 5 | 88,6 · 53,3 · 81,0 % |
| det ∨ seg | P1 | 0,55 / 0,80 / 0,30 | 4 | 10 | 6 | 6 | 18 | 27 | 5 / 5 / 5 | 11,8 · 11,8 · 19,1 % |
| det ∨ seg | P2 | 0,55 / 0,30 / 0,30 | 4 | 81 | 127 | 29 | 74 | 27 | 5 / 0 / 5 | 85,4 · 25,8 · 53,1 % |

**v2:**

| Decisor | Regla | Umbrales car / plate / seg | GENERAL | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % | DETAIL | nuevas G/P/D | U3 por alerta 25 / 40 / 60 % |
|---|---|---|---|---|---|---|---|---|---|---|
| det | P1 | 0,90 / 0,30 / – | 398 | 5 | 126 | 19 | 14 | 368 | 2 / 2 / 2 | 100,0 · 99,9 · 99,9 % |
| det | P2 | 0,90 / 0,30 / – | 398 | 137 | 151 | 151 | 151 | 368 | 2 / 0 / 2 | 100,0 · 100,0 · 100,0 % |
| det ∨ seg | P1 | 0,90 / 0,30 / 0,30 | 11 | 4 | 11 | 3 | 9 | 47 | 4 / 4 / 4 | 21,1 · 16,6 · 20,0 % |
| det ∨ seg | P2 | 0,90 / 0,30 / 0,30 | 11 | 83 | 130 | 42 | 76 | 47 | 4 / 0 / 4 | 88,2 · 38,6 · 57,7 % |

Con v2 y sin la exclusión, S1 solo se cumple con `car` ≥ 0,90: ahí rechaza 11 de 417 GENERAL y 47 de 374 DETAIL con
det ∨ seg (U3 por alerta de 16,6 a 21,1 %), y con det solo rechaza casi todo. Esos altos umbrales existen porque
camiones y colectivos reales salen con confianza 0,97: no son errores del detector.

#### Sensibilidad del área mínima de la patente (det ∨ seg, umbrales actuales; rechazadas de 137 / 151)

**v1:**

| Área mínima de la patente | Regla | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % |
|---|---|---|---|---|---|
| 5 % (servicio) | P1 | 1 | 3 | 3 | 4 |
| 5 % (servicio) | P2 | 96 | 128 | 33 | 85 |
| 1 % | P1 | 1 | 1 | 2 | 4 |
| 1 % | P2 | 96 | 13 | 30 | 85 |
| 0 % | P1 | 1 | 1 | 2 | 4 |
| 0 % | P2 | 96 | 12 | 30 | 85 |

**v2:**

| Área mínima de la patente | Regla | PLATE 50 % | PLATE 25 % | PLATE 40 % | PLATE 60 % |
|---|---|---|---|---|---|
| 5 % (servicio) | P1 | 9 | 4 | 6 | 12 |
| 5 % (servicio) | P2 | 85 | 129 | 40 | 81 |
| 1 % | P1 | 9 | 3 | 5 | 12 |
| 1 % | P2 | 85 | 22 | 34 | 81 |
| 0 % | P1 | 9 | 3 | 5 | 12 |
| 0 % | P2 | 85 | 21 | 34 | 81 |

Es un diagnóstico; no se propone cambiar `PRESENCE_MIN_AREA`.

#### Lo que esta medición no mide

- Fotos válidas de **camiones, vans y colectivos**: no hay un set de reportes válidos de esos vehículos. v1 los rechaza como
  `NO_VEHICLE` (las negativas que acepta v2 lo muestran) y v2 los acepta, pero cuánto cambia el rechazo de reportes
  válidos no está medido.
- El costo de latencia por alerta: v1 y v2 son iguales (ver "Detector v2 contra v1"). Cualquier ganancia de v2 en el rechazo
  viene del modelo, no de la regla.

### Precisión corregida del detector, por revisión manual de un solo revisor (`fp_audit.py`)

Open Images no anota todo lo que se ve, así que parte de los "falsos positivos" son autos o patentes reales sin anotar.
`detector.py --dump-fp` escribe cada falso positivo (de auto y de patente, con los umbrales del servicio) y `fp_audit.py`
arma hojas de contacto (fuera del repo) con la caja en rojo y las reales en verde. La precisión corregida es **una corrección por
revisión manual de un solo revisor**: un solo revisor miró los 145 distintos (114 sin mosaico, más 33 que solo aparecen
con mosaico; el resto se hereda por IoU ≥ 0,5), sin segunda opinión. El veredicto es subjetivo y por eso se da con y sin
los dudosos. **La precisión oficial no se reemplaza**: la corregida va aparte, al lado de la oficial.

```powershell
uv run --frozen --extra dev --extra tools python tools/eval/detector.py --split test --dump-fp
uv run --frozen --extra dev --extra tools python tools/eval/fp_audit.py sheets --dump <falsos_positivos.json> --out <carpeta>
# se mira cada hoja y se anota veredictos.json ({categoría: [números de panel]}) en <carpeta>
uv run --frozen --extra dev --extra tools python tools/eval/fp_audit.py corrected --report <detector_report.json> --audit <carpeta>
```

| Clase | Falsos positivos | Auto o patente real sin anotar | Otro vehículo (camión, pickup, furgoneta) | Duplicada o mal ubicada | No es | Dudoso | Precisión oficial | **Corregida** (con dudosos) |
|---|---|---|---|---|---|---|---|---|
| Auto | 74 | 46 | 10 | 5 | 2 | 11 | 0,747 | **0,904** (0,942) |
| Patente, sin mosaico | 40 | 18 | 0 | 5 | 12 | 5 | 0,737 | **0,855** (0,888) |
| Patente, con mosaico | 71 | 23 | 0 | 10 | 27 | 11 | 0,632 | **0,751** (0,808) |
| Auto + patente (micro), sin mosaico | 114 | 64 | 10 | 10 | 14 | 16 | 0,744 | **0,888** (0,924) |
| Auto + patente (micro), con mosaico | 145 | 69 | 10 | 15 | 29 | 22 | 0,702 | **0,844** (0,889) |

La corregida cuenta como acierto solo lo de la columna "real sin anotar". Los "otros vehículos" no cuentan: Open Images los
anota con otra clase (`Truck`, `Van`), aunque para el servicio una pickup o una furgoneta sean un vehículo. Los falsos
positivos de patente son sobre todo textos o piezas con forma de patente (marcas, insignias, carteles, faros): 12 de 40
sin mosaico y 27 de 71 con mosaico, así que el mosaico agrega ruido real, no solo patentes sin anotar. Los falsos
positivos de auto que no lo son casi no existen (2 de 74: un scooter y un cartel con la foto de un auto).

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
| R1 | `NO_VEHICLE` (fraude) | En alguna foto no hay un vehículo (la misma regla para GENERAL, PLATE, DETAIL y CONTEXT): una caja `car` del detector o una instancia del segmentador (det ∨ seg) que ocupa al menos `PRESENCE_MIN_AREA` de la foto. La patente, la rueda y el daño no cuentan. |
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
chequeo de **daño**: el daño tampoco cuenta como presencia en DETAIL (decisión del 12/11/2026), que sigue exactamente la
regla de GENERAL. Verificar el daño o la rampa es trabajo futuro. `DAMAGE_PRESENCE_T` queda solo como la confianza
mínima del adaptador de daño (`MODEL_DAMAGE`, experimental).

## Rebajes: ¿el auto bloquea el rebaje? (`curb.py`)

> **Congelado y experimental, atado al detector v1 (decisión del 11/11/2026).** Es lo único que sigue usando
> la clase `wheel` (`domain/ramp.py::wheel_diameter` mide la rueda para la escala del rebaje) y esa clase
> existe solo en los pesos de v1 (`rfdetr_openimages`). El detector v2 no la tiene: con v2 no hay diámetro de
> rueda y se usa la escala de respaldo. No se modifica ni se mantiene hasta que se retome el rebaje.

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
