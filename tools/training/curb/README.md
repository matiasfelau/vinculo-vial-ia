# Entrenar el modelo de rebajes

El servicio necesita saber si el auto de la foto de contexto **bloquea el tramo rebajado del cordón** (la
entrada de un garage o una rampa de accesibilidad). Para eso un modelo de segmentación marca, en la foto, dos
cosas: el **cordón** (`curb`) y el **tramo rebajado** (`curb_cut`). Después el servicio mide cuánto del rebaje
tapa el auto.

El modelo es RF-DETR-Seg Small (licencia Apache 2.0) y se entrena con **tus fotos**. Parte de un modelo que ya
sabe segmentar objetos de uso general (COCO) y se lo ajusta a estas dos clases. Es un modelo que se
**reentrena a medida que el dataset crece**.

Las fotos tienen rostros y patentes de terceros: todo vive en `EVAL_DATA_DIR\curb\`, fuera del repo, y no se
sube a ninguna nube. Todos los scripts se niegan a escribir dentro del repo.

```text
D:\vv-eval\curb\          <- EVAL_DATA_DIR\curb
  raw\                    las fotos, tal como las sacaste
  fotos.csv               qué es cada foto (ver abajo)
  prelabel\               el preetiquetado con SAM 3
  cvat-export\            lo que exportás de CVAT, corregido a mano
  splits\                 qué foto fue a cada conjunto (test.json no cambia nunca)
  dataset\                el dataset armado para entrenar
  runs\                   los resultados de cada entrenamiento
```

## 1. Describir las fotos: `fotos.csv`

Una fila por foto de `raw\`. Columnas, en este orden:

| Columna | Valores |
|---|---|
| `archivo` | Nombre del archivo en `raw\`. |
| `tipo` | `positivo_claro`, `positivo_marginal`, `negativo_auto_junto_rebaje`, `negativo_auto_sin_rebaje`, `negativo_rebaje_sin_auto`, `negativo_parecido` o `descartada`. |
| `rebaje` | `garage`, `accesibilidad` o `ninguno`. |
| `parecido` | Solo con `tipo = negativo_parecido` (y ahí es obligatorio): `cordon_pintado`, `cordon_roto`, `boca_tormenta`, `porton_sin_rebaje`, `desnivel_obra` u `otro`. |
| `iluminacion` | `dia`, `atardecer` o `noche`. |
| `clima` | `seco`, `mojado` o `lluvia`. |
| `oclusion` | `ninguna`, `parcial` o `fuerte` (definición abajo). |
| `notas` | Texto libre, opcional. |

Qué es cada `tipo`:

- `positivo_claro`: el auto tapa claramente el rebaje. Hay que detectarlo como bloqueo (`BLOCKING`).
- `positivo_marginal`: lo tapa apenas o está en el borde. Tiene que dar `BLOCKING`: la decisión es binaria.
- `negativo_auto_junto_rebaje`: hay un auto al lado del rebaje, **sin** taparlo (`NOT_BLOCKING`).
- `negativo_auto_sin_rebaje`: hay un auto y no hay rebaje (`NO_DRIVEWAY`).
- `negativo_rebaje_sin_auto`: hay un rebaje y no hay auto (`NO_CAR`; además el modelo tiene que ver el rebaje).
- `negativo_parecido`: algo que parece un rebaje y no lo es: un cordón pintado, una boca de tormenta, etc.
  (`NO_DRIVEWAY`, o `NO_CAR` si no hay auto).
- `descartada`: una foto que no sirve (movida, tapada...). Se excluye de todo.

Qué es cada `oclusion`. La **zona** es el rebaje, el cordón a sus lados y el borde inferior del auto:

- `ninguna`: nada se interpone en la zona.
- `parcial`: algo tapa **menos de la mitad** de la zona y el encuentro entre el auto y el rebaje sigue visible.
- `fuerte`: tapa **la mitad o más**, o justo el encuentro entre el auto y el rebaje.

Una rueda tapada por un objeto es oclusión; una rueda fuera de cuadro o escondida por el ángulo de la foto no lo es.

El script **rechaza cualquier valor que no esté en estas listas**, indicando el número de fila (la fila 2 es la
primera de datos). Mejor un error acá que un dataset con etiquetas mal escritas.

## 2. Etiquetar

Seguí [tools/labeling/README.md](../../labeling/README.md): SAM 3 preetiqueta, vos corregís en CVAT y
exportás en formato "COCO 1.0" a `cvat-export\`.

## 3. Armar el dataset

```powershell
$env:EVAL_DATA_DIR = "D:\vv-eval"
uv run --frozen --extra dev --extra tools python tools/training/curb/prepare.py
```

Valida el CSV, une las anotaciones de CVAT con la clase de cada foto y reparte las fotos así:

- **Cada foto se reparte sola**: no se agrupan por lugar ni por escena.
- **Al azar con semilla fija** (`--seed`, 0 por defecto, así que repetir el comando da lo mismo),
  aproximadamente **60 % entrenamiento, 20 % validación y 20 % prueba**, y **estratificado por tipo**: cada
  conjunto tiene proporciones parecidas de cada tipo de foto. Con pocas fotos de un tipo las proporciones se
  corren: el script muestra lo que logró.
- **El test se guarda la primera vez en `splits	est.json` y no cambia nunca más**, así podés comparar modelos
  entrenados en momentos distintos con exactamente las mismas fotos. Las fotos nuevas van solo a entrenamiento
  y validación.

Te avisa también si una foto de un tipo que muestra un rebaje no tiene ningún `curb_cut` anotado (suele ser un
olvido al etiquetar) y cuántas anotaciones descartó (por ejemplo las que están como máscara en vez de polígono).
Con menos de 3 fotos no se puede repartir.

Si tu CVAT exporta en otra carpeta: `--export RUTA` (la carpeta o el `.json`). `--seed N` cambia el reparto de
train y valid; el test guardado no cambia.

## 4. Entrenar

Cerrá lo que use la GPU (`nvidia-smi` muestra qué hay). Hacen falta unos 3,5 GB libres.

```powershell
uv run --frozen --extra dev --extra tools python tools/training/curb/train.py
```

Por defecto: hasta 50 épocas, **lote de 2 con acumulación de 8** (equivale a 16), **FP32** y **parada
temprana**: termina solo si el modelo no mejora en 10 épocas seguidas sobre las fotos de validación. Una
*época* es una pasada completa por las fotos de entrenamiento. Las opciones son las del detector
([tools/training/detector/README.md](../detector/README.md)): `--device`, `--epochs`, `--batch-size`,
`--grad-accum`, `--amp`, `--patience`, `--resume`.

Los resultados quedan en `runs\rfdetr-seg-curb\`: `checkpoint_best_total.pth` (el que se exporta),
`metrics.csv` y `training_summary.json` (tiempo, memoria de la GPU y métricas).

**Con pocas fotos** (la guía académica pide unas 400, con 45 % de positivas) las métricas de validación son
ruidosas: una foto más o menos las mueve mucho. Tomalas como una señal de que el pipeline funciona, no como
una medida final.

## 5. Exportar

```powershell
$env:MODELS_DIR = "models\weights"
uv run --frozen --extra dev --extra tools python tools/training/curb/export.py
```

Genera `MODELS_DIR\rfdetr_seg_curb.onnx`, comprueba que las clases sean `curb` y `curb_cut` y que el ONNX
reproduzca a PyTorch, y muestra el `sha256`. Para usarlo hay que **agregarlo a `models/manifest.json`** (el
sha256 más la licencia Apache 2.0 y el paper de RF-DETR; los datos son propios) y poner en el `.env`:

```dotenv
MODEL_CURB=rfdetr_seg_curb
CURB_MIN_CONF=0.50
MODELS_DIR=./models/weights
```

## 6. Evaluar

```powershell
$env:EVAL_OUT = "D:\vv-eval\out\curb"
uv run --frozen --extra dev --extra tools python tools/eval/curb.py
```

Compara, sobre las fotos de **prueba**, tres sistemas:

- `sam3`: SAM 3 sin entrenar (requiere el acceso de Hugging Face; con `--no-sam3` se omite);
- `entrenado`: el modelo que entrenaste;
- `oraculo`: las máscaras que anotaste vos. No es un modelo: sirve para separar errores.

Ver [tools/eval/README.md](../../eval/README.md) para leer los resultados.

## Problemas frecuentes

- **"fotos.csv tiene errores"**: la lista dice qué fila y qué valor. Corregilo en el CSV y repetí.
- **"N fotos de fotos.csv no están en la exportación de CVAT"**: te faltó incluirlas en la task o
  exportar de nuevo.
- **"la exportación no tiene las clases curb_cut"**: en CVAT, las etiquetas tienen que llamarse exactamente
  `curb` y `curb_cut`.
- **"hacen falta al menos 3 fotos"**: el reparto necesita al menos tres fotos.
- **`CUDA out of memory`**: `--batch-size 1 --grad-accum 16`. No cambies a un modelo más chico.
