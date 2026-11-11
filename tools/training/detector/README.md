# Entrenar el detector (autos, ruedas, patentes y rostros)

> **Este es el detector v1.** Existe también el **v2** (`tools/training/detector_v2/`): presencia de vehículo,
> patente y confusoras, **sin rostro ni rueda**. Decisiones del 11/11/2026:
>
> - El camino `MODEL_FACE=rfdetr_openimages` (rostros del detector) **queda solo para v1**. v2 no lo ofrece
>   (`MODEL_FACE=rfdetr_openimages_v2` no arranca) y su adaptador toma los ids de clase del manifest.
> - La clase **rueda** salió de todo camino activo del servicio (ya no cuenta en R1). Solo sigue dentro de los
>   pesos de v1 y en el chequeo de rebajes (`domain/ramp.py::wheel_diameter`, `tools/eval/curb.py`), que está
>   **congelado y es experimental, atado a v1**: no se modifica.

Un **detector** recibe una foto y devuelve cajas: dónde hay un auto, una rueda, una patente o un rostro.
El servicio lo usa para dos cosas: encontrar la patente que hay que leer y **tapar las patentes y los
rostros** antes de guardar la evidencia. Un solo modelo (RF-DETR Small, licencia Apache 2.0) hace los
dos trabajos: `MODEL_DETECTOR` y `MODEL_FACE` apuntan al mismo archivo.

Entrenar es mostrarle al modelo muchas fotos donde ya sabemos dónde están las cajas, para que aprenda a
encontrarlas solo. No se parte de cero: se parte de un modelo que ya sabe detectar objetos de uso general
(entrenado con COCO) y se lo **ajusta** a nuestras cuatro clases con fotos de Open Images V7.

Todo lo que se baja y se genera acá tiene **fotos de terceros con rostros y patentes**: vive fuera del
repo y los scripts se niegan a escribir dentro de él.

## Antes de empezar (una sola vez)

1. Dependencias y carpeta de pesos: seguí "Antes de empezar" de
   [tools/export/README.md](../../export/README.md).
2. Definí `TRAIN_DATA_DIR`, una carpeta fuera del repo con unos 10 GB libres, y avisale a la terminal:

   ```powershell
   $env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
   $env:MODELS_DIR = "models\weights"
   ```

   Dentro de esa carpeta los scripts crean `openimages-detector\` (las fotos y anotaciones),
   `fiftyone-cache\` (lo que baja FiftyOne) y `runs\detector\` (los resultados del entrenamiento).

## 1. Bajar las fotos

```powershell
uv run --frozen --extra dev --extra tools python tools/training/detector/download.py
```

Baja de Open Images V7, con FiftyOne, fotos de tres tipos por cada split (`train`, `valid`, `test`): con
patentes, con rostros y con autos. Hacerlo por etapas evita que las patentes, que son lo más escaso,
queden con pocos ejemplos. Cada foto conserva todas sus cajas de las cuatro clases.

Por defecto baja `500 400 400` fotos de entrenamiento (patentes, rostros, autos), `100 80 80` de
validación y `100 80 80` de prueba. Para otro tamaño:

```powershell
uv run --frozen --extra dev --extra tools python tools/training/detector/download.py --train 1000 800 800
```

La primera vez baja unos 1,5 GB de anotaciones y tarda varios minutos. Si un split ya existe lo
omite; `--force` lo rehace.

**Qué se descarta:** las cajas que cubren un grupo entero (`IsGroupOf`), dibujos o juguetes
(`IsDepiction`), fotos tomadas desde adentro del objeto (`IsInside`) y cajas de menos de 8 px en la foto
original. El resumen queda en `dataset_info.json`, con cantidades por clase.

## 2. Entrenar

Antes de empezar, **cerrá lo que use la GPU** (juegos, navegadores, OBS). RF-DETR Small necesita unos
3 GB de memoria de video y el script se detiene si hay menos de 3,5 GB libres. Se ve con `nvidia-smi`.

```powershell
uv run --frozen --extra dev --extra tools python tools/training/detector/train.py
```

Por defecto: 20 épocas como máximo, lote de 2 fotos con acumulación de 8 (equivale a un lote de 16
sin necesitar la memoria de 16), **FP32** y parada temprana: si el modelo no mejora en 5 épocas seguidas
sobre las fotos de validación, termina solo. Una *época* es una pasada completa por las fotos de
entrenamiento.

| Opción | Para qué |
|---|---|
| `--device cuda\|cpu` | Dónde corre. En CPU funciona pero tarda muchísimo más. |
| `--epochs N` | Máximo de épocas. |
| `--batch-size N --grad-accum N` | Fotos por paso y pasos que se acumulan. Si falta memoria, bajá `--batch-size` a 1 y subí `--grad-accum` a 16. |
| `--amp` | Precisión mixta FP16. Apagada: la GTX 1660 no tiene tensor cores y casi no ayuda. |
| `--patience N` | Épocas sin mejorar antes de parar. |
| `--resume ARCHIVO` | Retoma desde un checkpoint, por ejemplo después de un corte de luz. |

Si se corta, volvé a correrlo con `--resume` apuntando a `runs\detector\last.ckpt`.

### Qué queda en `runs\detector\`

- `checkpoint_best_total.pth`: el mejor modelo según validación. Es el que se exporta.
- `metrics.csv`: una fila por época, con las métricas de entrenamiento y de validación.
- `training_summary.json`: tiempo total, pico de memoria de la GPU y versiones.

### Cómo leer las métricas

- `val/mAP_50`: qué tan bien encuentra las cajas con un solapamiento de al menos el 50 % (IoU 0,5). Va de
  0 a 1; más es mejor.
- `val/mAP_50_95`: lo mismo pero exigiendo cajas cada vez más exactas. Siempre es menor.
- `val/recall`: de todas las cajas reales, qué fracción encontró. Para tapar rostros y patentes importa
  más que la precisión: una caja de más tapa de más, una caja de menos deja un rostro a la vista.
- `train/loss` tiene que bajar; si `val/mAP_50` deja de subir mientras la pérdida sigue bajando, el
  modelo está memorizando (sobreajuste) y la parada temprana lo frena.

## 3. Exportar a ONNX

```powershell
uv run --frozen --extra dev --extra tools python tools/training/detector/export.py
```

Toma el mejor checkpoint, lo convierte a `MODELS_DIR\rfdetr_openimages.onnx` (unos 115 MB) y hace tres
controles: que las clases sean `car, wheel, plate, face` en ese orden, que el ONNX reproduzca a PyTorch
(al menos el 95 % de sus consultas) y que la salida tenga la forma que espera el adaptador. Muestra el
`sha256` que va en `models/manifest.json`.

**El entrenamiento no es reproducible bit a bit:** otro entrenamiento da otros pesos y otro `sha256`. El
archivo que se usa es el que se exportó y se registró en el manifest; para tenerlo en otra máquina,
copialo.

## 4. Evaluar

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\detector"
uv run --frozen --extra dev --extra tools python tools/eval/detector.py --split test
```

Usa el adaptador real (el mismo código que el servicio) sobre las fotos de prueba, que el modelo nunca vio, y
muestra por clase el **recall** y la **precisión** con los umbrales del servicio, el **AP50**, el recall
en objetos chicos (menos de 32 px) y cuánto tarda en CPU. Ver [tools/eval/README.md](../../eval/README.md).

## 5. Usarlo en el servicio

En el `.env`:

```dotenv
MODEL_DETECTOR=rfdetr_openimages
MODEL_FACE=rfdetr_openimages
MODELS_DIR=./models/weights
DETECTOR_MIN_CONF=0.40
```

`REDACTION_REQUIRED` sigue en `false` en local hasta validar que tapa bien rostros y patentes.

## En Colab o en una GPU alquilada

El entrenamiento solo usa fotos de Open Images, que se pueden subir a Colab. Con el repo clonado:

```bash
pip install uv==0.12.23
uv sync --frozen --extra dev --extra tools
export TRAIN_DATA_DIR=/content/vv-train
uv run --frozen --extra dev --extra tools python tools/training/detector/download.py --train 1500 1200 1200
uv run --frozen --extra dev --extra tools python tools/training/detector/train.py --batch-size 8 --grad-accum 2
```

Con una GPU de 16 GB conviene un lote mayor. Al terminar, bajá `runs/detector/checkpoint_best_total.pth` a
tu máquina y corré `export.py` ahí. **No subas nunca fotos propias**: con rostros o patentes de terceros
solo se suben las estrictamente necesarias y se borran al terminar.

## Problemas frecuentes

- **"La GPU tiene solo N MB libres"**: cerrá lo que la usa y mirá `nvidia-smi`.
- **`CUDA out of memory` a mitad del entrenamiento**: bajá `--batch-size` (a 1) y subí `--grad-accum`; no
  cambies a un modelo más chico.
- **"Falta train/_annotations.coco.json"**: corré antes `download.py`.
- **"no encontré ningún checkpoint"**: el entrenamiento no terminó ninguna época; mirá `runs\detector`.
- **El entrenamiento se queda esperando en Windows**: probá `--num-workers 0`.

## Resultados de referencia

Los números de la primera corrida, para saber qué esperar. Se midieron en una GTX 1660 de 6 GB con las
fotos que baja `download.py` por defecto (`TRAIN_DATA_DIR` = `D:\vv-eval\cardd`). **El entrenamiento no es
reproducible bit a bit: los tuyos van a ser parecidos, no iguales.**

| | |
|---|---|
| Fotos | entrenamiento 1057 (1721 autos, 1481 ruedas, 688 patentes, 1121 rostros) · validación 201 · prueba 204 |
| Configuración | RF-DETR Small a 512 px, FP32, lote 2 con acumulación de 8, hasta 20 épocas con paciencia 5 |
| Duración | **55 minutos** (3310 s); paró solo tras 13 evaluaciones, a unos 4 o 5 minutos por época |
| Memoria de la GPU | pico de **2,8 GB** asignados (3,3 GB reservados); `nvidia-smi` ve unos 4,1 GB en total |
| Validación (mejor época) | mAP50 0,83 · mAP50:95 0,62 · recall 0,80 · precisión 0,81 |
| Prueba (nunca vista) | mAP50 0,83 · mAP50:95 0,59 · recall 0,79 · precisión 0,81 |

El modelo ya partía de buenos pesos (COCO): con una sola época la validación daba mAP50 0,73, y en la cuarta
época ya estaba cerca de su techo. Más épocas no ayudaron; lo que puede ayudar son **más fotos**.

Con el adaptador del servicio y sus umbrales (`tools/eval/detector.py --split test`, 204 fotos que el
modelo nunca vio; 280 ms por foto en CPU):

| Clase | Cajas reales | Umbral | Recall | Precisión | AP50 | Recall en chicas (< 32 px) |
|---|---|---|---|---|---|---|
| `car` | 244 | 0,40 | 0,90 | 0,75 | 0,90 | 0,33 (6 cajas) |
| `wheel` | 379 | 0,40 | 0,54 | 0,89 | 0,78 | 0,14 (91 cajas) |
| `plate` | 135 | 0,20 | 0,83 | 0,74 | 0,79 | 0,61 (51 cajas) |
| `face` | 135 | 0,30 | 0,76 | 0,89 | 0,83 | 0,15 (13 cajas) |

Cómo leerlo:

- Los **rostros** se encuentran en tres de cada cuatro (0,88 en validación); los chicos casi no. Para tapar
  rostros esto importa mucho: Open Images anota menos las caras chicas, así que el modelo aprendió poco de
  ellas. La comparación con otros detectores sobre fotos propias está en
  [tools/eval/README.md](../../eval/README.md); hasta entonces `REDACTION_REQUIRED` sigue en `false` en local.
- La **precisión** de autos y patentes (0,75) es la más baja. Puede deberse en parte a objetos reales que Open
  Images no anotó (el dataset no es exhaustivo), pero no lo verifiqué mirando las cajas; para tapar patentes
  una caja de más cuesta poco.
- Las **ruedas** se pierden más (recall 0,54): se usan solo para estimar la escala del rebaje, y sin rueda el
  servicio usa una escala de respaldo.
