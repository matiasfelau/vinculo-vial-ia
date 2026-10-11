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

1. Abrí <https://universe.roboflow.com/patentes-obs9r/patentes-n81go>.
2. **Download Dataset** → formato **COCO** → **Download zip to computer**.
3. Descomprimilo en una carpeta fuera del repo, por ejemplo `D:\vv-eval\ocr-roboflow`. Tienen que
   quedar las carpetas `train`, `valid` y `test`, cada una con su `_annotations.coco.json`.

### 2. Correrlo

Elegí dónde se guardan los resultados (otra carpeta fuera del repo) y avisale a la terminal:

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval\ocr-roboflow"
$env:EVAL_OUT = "D:\vv-eval\out\ocr"
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/eval/ocr.py
```

Tarda unos segundos. Las variables se pierden al cerrar la terminal. También se pueden pasar como
`--data-dir`, `--out-dir` y `--models-dir`.

### 3. Qué hace el script

1. Lee las cajas de las patentes del COCO. La etiqueta de cada caja es el **nombre de su clase**, sin
   espacios y en mayúsculas.
2. Descarta los **marcadores de posición**: etiquetas cuyos dígitos son todos ceros, como `AA000AA`.
3. Recorta cada caja de la foto igual que el servicio y se la pasa al OCR.
4. Compara la lectura con la etiqueta usando la misma función que el servicio (`plate_check`).

### 4. Cómo leer el resultado

La consola muestra algo así (los números dependen del dataset):

```text
Modelo parseq@7ff186b1f743 · 137 cajas · descartadas {'marcador de posición': 3}
Lecturas: 134 · fallas del modelo: 0
Coincidencia exacta: 104 (77.6 %)
Distribución con OCR_MIN_CONF=0.6: {'PLATE_MATCH': 105, 'PLATE_FUZZY': 12, 'PLATE_MISMATCH': 3, 'PLATE_UNREADABLE': 14}
Propuesta (sin 4 lecturas de otros formatos): OCR_MIN_CONF=0.78 (74.6 % de lecturas aceptadas)
```

| Resultado | Qué significa en el servicio |
|---|---|
| `PLATE_MATCH` | La lectura coincide con la patente (permitiendo las confusiones típicas O/0, I/1, B/8…). |
| `PLATE_FUZZY` | Difiere en un solo carácter: alerta amarilla. |
| `PLATE_MISMATCH` | Difiere en más: **alerta roja de fraude**. Es el error caro. |
| `PLATE_UNREADABLE` | Lectura vacía o con confianza menor a `OCR_MIN_CONF`: alerta amarilla. |

*Coincidencia exacta* es más estricta que `MATCH`: no permite ninguna corrección por confusión.

En `EVAL_OUT` queda además `report.json`, con las mismas cifras desglosadas por split, por formato de
patente (Mercosur, anterior, otro) y un barrido de umbrales.

### 5. Auditar a ojo

Abrí `audit.html` en el navegador. Muestra cada recorte con su etiqueta, la lectura, la confianza y el
resultado. Incluye **todo lo que no fue un acierto exacto** y una muestra al azar de aciertos
(`--match-samples`, 20 por defecto). Sirve para dos cosas:

- ver *por qué* falla el modelo (foto borrosa, patente cortada, moto de dos líneas…);
- detectar **etiquetas mal cargadas** en el dataset: si el modelo lee bien y la etiqueta está mal, el
  error es del dataset, no del modelo.

### 6. El umbral `OCR_MIN_CONF`

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
  `test` un nivel.
- **"No se pudo empezar: ... sha256 ..."**: el archivo de `MODELS_DIR` no es el del manifest; ver
  [tools/export/README.md](../export/README.md).
