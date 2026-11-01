# Entrenar el detector de daños (CarDD)

Este modelo mira la foto de **detalle** de una alerta y dibuja una caja en cada daño que encuentra:
abolladuras, rayones, grietas, vidrios rotos, ópticas rotas y neumáticos pinchados. El servicio lo usa
para saber si la foto de detalle realmente muestra un daño. Es RF-DETR Small (licencia Apache 2.0),
ajustado con CarDD a partir de un modelo que ya detecta objetos de uso general (COCO).

## Condiciones de CarDD (leelas antes de seguir)

CarDD es un dataset del PIC Lab (Chinese Academy of Sciences). Lo tenés por consentimiento previo del
titular, **para investigación**:

- el dataset **nunca** va al repositorio, a la imagen de Docker ni a la nube;
- **no se redistribuye** a terceros;
- los **pesos entrenados con él no se publican**: viven en `MODELS_DIR`, nunca en git;
- el **uso comercial**, incluido probar sistemas comerciales, requiere autorización previa del PIC
  Lab. Por eso el modelo figura como `escalable` en `models/manifest.json`.

Cita: Wang, Li y Wu. *CarDD: A New Dataset for Vision-Based Car Damage Detection*. IEEE Transactions on
Intelligent Transportation Systems, 24(7):7202-7214, 2023.

## Antes de empezar (una sola vez)

1. Dependencias y carpeta de pesos: seguí "Antes de empezar" de
   [tools/export/README.md](../../export/README.md).
2. `TRAIN_DATA_DIR` es la carpeta fuera del repo que contiene `CarDD_release\` (el `CarDD_COCO` que
   descargaste). Avisale a la terminal:

   ```powershell
   $env:TRAIN_DATA_DIR = "D:\vv-eval\cardd"
   $env:MODELS_DIR = "models\weights"
   ```

## 1. Preparar el dataset

```powershell
uv run --frozen --extra dev --extra tools python tools/training/damage/prepare.py
```

Reordena CarDD para que RF-DETR lo entienda: crea `cardd-detector\train`, `valid` y `test` con las
fotos (enlazadas al original si están en el mismo disco, así que **no ocupan espacio de más**) y un
`_annotations.coco.json` con las cajas, sin las máscaras. Si las categorías no son las de CarDD en su
orden, se detiene. Imprime cuántas fotos y cuántas cajas de cada clase hay en cada split.

## 2. Entrenar

Cerrá antes lo que use la GPU (juegos, navegadores, OBS); se ve con `nvidia-smi`. El script se detiene
si hay menos de 3,5 GB de memoria de video libres.

```powershell
uv run --frozen --extra dev --extra tools python tools/training/damage/train.py
```

Por defecto: hasta 20 épocas, lote de 2 fotos con acumulación de 8 (lote efectivo 16), **FP32** y
parada temprana: termina solo si el modelo no mejora en 5 épocas seguidas sobre las fotos de
validación. Las opciones son las mismas que las del detector de vehículos
([tools/training/detector/README.md](../detector/README.md)): `--device`, `--epochs`, `--batch-size`,
`--grad-accum`, `--amp`, `--patience`, `--resume`.

Los resultados quedan en `runs\damage\`: `checkpoint_best_total.pth` (el que se exporta), `metrics.csv`
(una fila por época) y `training_summary.json` (tiempo, memoria de la GPU y métricas).

Si falta memoria no cambies a un modelo más chico: bajá `--batch-size` a 1 y subí `--grad-accum` a 16; si
no alcanza, usá CPU para lo que se pueda o una GPU alquilada.

## 3. Exportar a ONNX

```powershell
uv run --frozen --extra dev --extra tools python tools/training/damage/export.py
```

Convierte el mejor checkpoint a `MODELS_DIR\rfdetr_cardd.onnx` y comprueba que las clases sean las de
CarDD en su orden y que el ONNX reproduzca a PyTorch. Muestra el `sha256` para el manifest. El
entrenamiento no es reproducible bit a bit: el archivo que vale es el que se exportó y se registró.

## 4. Evaluar

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\damage"
uv run --frozen --extra dev --extra tools python tools/eval/damage.py --split test
```

Muestra por clase el recall, la precisión, el AP50 y el recall en daños chicos sobre las 374 fotos de
prueba, que el modelo no vio. Ver [tools/eval/README.md](../../eval/README.md).

## 5. Usarlo en el servicio

```dotenv
MODEL_DAMAGE=rfdetr_cardd
MODELS_DIR=./models/weights
DAMAGE_T_LOW=0.969
DAMAGE_T_HIGH=0.969
```

`DAMAGE_T_LOW` y `DAMAGE_T_HIGH` son los dos umbrales de confianza del daño (ver `tools/eval/README.md`). Los
defaults (0,969) son **provisorios** y **no cumplen** la regla de que un caso válido no termine en rojo: dejan en
rojo el 80 % de las fotos con daño de CarDD. Las cifras de abajo se midieron con 0,40.

## Problemas frecuentes

- **"No encuentro las anotaciones de CarDD"**: `TRAIN_DATA_DIR` tiene que contener
  `CarDD_release\CarDD_COCO\annotations`. Si está en otro lado, usá `--source`.
- **"las categorías no son las de CarDD"**: el JSON no es el original de CarDD.
- **"La GPU tiene solo N MB libres"**: cerrá lo que la usa.
- **`CUDA out of memory`**: `--batch-size 1 --grad-accum 16`.

## Resultados de referencia

Los números de la primera corrida, para saber qué esperar. Se midieron en una GTX 1660 de 6 GB. **El
entrenamiento no es reproducible bit a bit: los tuyos van a ser parecidos, no iguales.**

| | |
|---|---|
| Fotos | entrenamiento 2816 (6211 cajas) · validación 810 · prueba 374 (de CarDD) |
| Configuración | RF-DETR Small a 512 px, FP32, lote 2 con acumulación de 8, 20 épocas como máximo con paciencia 5 |
| Duración | **3 h 32 min** (12 737 s) para las 20 épocas, a unos 10 minutos por época |
| Memoria de la GPU | pico de **2,8 GB** asignados (3,3 GB reservados) |
| Validación (mejor época, la 19) | mAP50 0,78 · mAP50:95 0,62 · recall 0,72 · precisión 0,78 |
| Prueba (nunca vista) | mAP50 0,80 · mAP50:95 0,64 · recall 0,77 · precisión 0,78 |

**Llegó al tope de 20 épocas sin que la parada temprana lo frenara**, pero la curva está casi plana: el
mAP50:95 de validación pasó de 0,618 en la época 12 a 0,623 en la 19. Más épocas darían una mejora marginal a
cambio de muchas horas de GPU. Si querés probar, `--epochs 30 --resume runs\damage\last.ckpt`.

Con el adaptador del servicio y `DAMAGE_MIN_CONF=0,40` (`tools/eval/damage.py --split test`, 374 fotos de
prueba; 330 ms por foto en CPU):

| Clase | Cajas reales | Recall | Precisión | AP50 |
|---|---|---|---|---|
| `dent` (abolladura) | 236 | 0,58 | 0,62 | 0,66 |
| `scratch` (rayón) | 307 | 0,65 | 0,66 | 0,70 |
| `crack` (grieta) | 70 | 0,59 | 0,59 | 0,58 |
| `glass_shatter` (vidrio roto) | 71 | 0,99 | 0,95 | 1,00 |
| `lamp_broken` (óptica rota) | 69 | 0,84 | 0,83 | 0,90 |
| `tire_flat` (neumático pinchado) | 32 | 0,97 | 0,97 | 0,99 |

**Por foto, que es lo que usa el servicio: el 99,5 % de las fotos con daño (372 de 374) tiene al menos una caja
con confianza de 0,40 o más.** Esta medida no mira falsos positivos: todas las fotos de CarDD tienen algún
daño, así que no dice qué pasa con una foto de detalle sin daño. Para eso hacen falta fotos propias sin daño.

Las clases difíciles son las de siempre en este dataset: las abolladuras, los rayones y las grietas son sutiles
y de límites poco claros. Los daños grandes y bien definidos (vidrios, ópticas, neumáticos) se encuentran
casi siempre.
