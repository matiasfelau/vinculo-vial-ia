# Exportar modelos a ONNX

El servicio no usa PyTorch: corre los modelos con ONNX Runtime en CPU. "Exportar" es convertir un
modelo de PyTorch a un archivo `.onnx` que el servicio pueda cargar. Se hace una sola vez por
modelo, en tu máquina, fuera de Docker.

Los archivos `.onnx` son pesos: **nunca se suben a git**. Van a una carpeta tuya (`MODELS_DIR`) y
el repo solo guarda su huella (sha256) en [models/manifest.json](../../models/manifest.json). Al
arrancar, el servicio compara la huella del archivo con la del manifest y no arranca si difieren.

## Antes de empezar (una sola vez)

Necesitás Windows o Linux, conexión a internet y unos 8 GB libres en disco.

1. Instalá `uv`, que maneja Python y las dependencias:

   ```powershell
   pip install uv==0.12.23
   ```

2. Parado en la raíz del repo, instalá las dependencias del servicio y las herramientas:

   ```powershell
   uv sync --frozen --extra dev --extra tools
   ```

   La primera vez baja PyTorch (unos 2,5 GB) y tarda varios minutos. Todo queda en la carpeta
   `.venv` del repo; no toca Docker ni el resto del sistema.

3. Elegí la carpeta donde vas a guardar los pesos y avisale a la terminal. La carpeta
   `models/weights` del repo sirve: está ignorada por git.

   ```powershell
   $env:MODELS_DIR = "models\weights"
   ```

   En Linux o Colab: `export MODELS_DIR=models/weights`.

## DINOv2 ViT-S/14 (puerto `embedder`)

Es el modelo que convierte la foto de un auto en un vector, para decidir si dos fotos muestran el
mismo auto. Se usa tal como lo publicó Meta (licencia Apache 2.0): no hay nada que entrenar.

1. Corré el export:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/export/dinov2.py
   ```

   Baja los pesos de Hugging Face (88 MB, públicos: no hace falta cuenta ni token) y tarda menos
   de un minuto. No necesita GPU. Si igual querés usarla, agregá `--device cuda`.

2. Vas a ver varias advertencias (`TracerWarning`, `DeprecationWarning`, avisos de Hugging Face).
   Son esperables. Lo que importa son las últimas líneas:

   ```text
   Diferencia máxima entre ONNX (CPU) y PyTorch (cpu): 5.44e-05
   Listo: models\weights\dinov2_vits14.onnx (88.4 MB), salida (1, 257, 384)
   sha256: 0ada57dd13c4e98d025aeefe3cc181dd0c76a9016d939cc9d7d3814557c416d9
   ```

   La primera línea es el control de calidad: el script le pasa la misma entrada al modelo
   original y al exportado y compara las salidas. Si la diferencia es grande, borra el archivo y
   termina con error.

3. Compará el `sha256` que te mostró con el de `dinov2_vits14` en `models/manifest.json`.

   - Si es igual, no hay nada más que hacer.
   - Si es distinto, tu archivo no es idéntico byte a byte al que se registró (suele pasar con otra
     versión de PyTorch). Lo más simple es copiar el archivo desde la máquina donde se registró.
     La alternativa es reemplazar el valor en el manifest y commitear ese cambio, y entonces todos
     los entornos tienen que usar tu archivo.

4. Probá el modelo real con el test de integración:

   ```powershell
   uv run --frozen --extra dev --extra tools pytest -m models
   ```

5. Para que el servicio lo use, en tu `.env`:

   ```dotenv
   MODEL_EMBEDDER=dinov2_vits14
   MODELS_DIR=./models/weights
   ```

   En `/health` vas a ver la versión cargada, por ejemplo `dinov2_vits14@0ada57dd13c4`.

### Opciones

| Opción | Para qué |
|---|---|
| `--out-dir CARPETA` | Guardar en otra carpeta en vez de `MODELS_DIR`. |
| `--device cuda\|cpu` | Dónde corre PyTorch durante el export. Por defecto `cpu`. |
| `--force` | Pisar el archivo si ya existe. |
| `--revision COMMIT` | Exportar otra revisión de Hugging Face. Cambia el sha256. |

### Problemas frecuentes

- **"Ya existe ... Usá --force"**: ya lo exportaste. Si el sha256 coincide con el manifest, no hace
  falta repetirlo.
- **"PyTorch no ve ninguna GPU"**: usá `--device cpu`; para este modelo da lo mismo.
- **"indicá --out-dir o definí MODELS_DIR"**: te faltó el paso 3 de "Antes de empezar". La variable
  se pierde al cerrar la terminal.
- **El servicio no arranca y dice que el sha256 no coincide**: el archivo de `MODELS_DIR` no es el
  que está registrado en el manifest. Ver el paso 3.

### En Colab

No hace falta para este modelo. Si igual querés hacerlo ahí, con el repo ya clonado:

```bash
pip install uv==0.12.23
uv sync --frozen --extra dev --extra tools
export MODELS_DIR=models/weights
uv run --frozen --extra dev --extra tools python tools/export/dinov2.py
```

Después descargá `models/weights/dinov2_vits14.onnx` a tu máquina. En este paso no interviene
ninguna foto ni dataset: solo los pesos públicos del modelo.

## RF-DETR-Seg con COCO (puerto `segmenter`)

Es el modelo que marca, píxel por píxel, dónde está cada auto de la foto. Se usa tal como lo
publica Roboflow, entrenado con COCO (licencia Apache 2.0): no hay nada que entrenar. Se exporta el
tamaño **Small**; el Medium también funciona pero tarda más en CPU. Nunca se usan XL ni 2XL, que
tienen otra licencia (PML 1.0): el script ni siquiera los ofrece.

1. Con las dependencias y `MODELS_DIR` ya listas (ver "Antes de empezar"), corré:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/export/rfdetr_seg.py
   ```

   La primera vez baja unos 129 MB de pesos a `~\.roboflow\models` (carpeta de la librería, fuera
   del repo, con verificación de MD5) y tarda un par de minutos. No necesita GPU, aunque acepta
   `--device cuda`. Con `--size medium` exporta el tamaño Medium.

2. Las últimas líneas tienen que parecerse a estas:

   ```text
   Diferencia máxima ONNX (CPU) vs PyTorch (cpu) en ('dets', 'labels', 'masks'): [0.0002, 0.0004, 0.0085]
   Listo: models\weights\rfdetr_seg_coco.onnx (123.4 MB)
   sha256: 202b2050c1776e7cc5b08e0632b7fee1b5fc52cb6fcc4bbcd56d36973351a335
   ```

   Las diferencias son el control de calidad: si alguna supera 0,05 el script no guarda el
   archivo. Pesa 123 MB.

3. Compará el `sha256` con el de `rfdetr_seg_coco` en `models/manifest.json`. Si no coincide, mismo
   criterio que con DINOv2: copiá el archivo desde la máquina donde se registró, o reemplazá el
   valor en el manifest y commiteá ese cambio.

4. Probá el modelo real: `uv run --frozen --extra dev --extra tools pytest -m models`.

5. Para usarlo, en tu `.env`:

   ```dotenv
   MODEL_SEGMENTER=rfdetr_seg_coco
   SEGMENTER_MIN_CONF=0.50
   MODELS_DIR=./models/weights
   ```

   `SEGMENTER_MIN_CONF` es la confianza mínima para aceptar un auto. Más alta, menos falsos autos
   pero más autos perdidos.

### Problemas frecuentes (RF-DETR-Seg)

- **"Ya existe ... Usá --force"**: ya lo exportaste; si el sha256 coincide con el manifest, no hace
  falta repetirlo.
- **Falla la descarga de los pesos**: necesita internet solo esta vez. Si la red bloquea
  `storage.googleapis.com`, bajá `rf-detr-seg-s-ft.pth` desde otra red y copialo como
  `rf-detr-seg-small.pt` en `~\.roboflow\models`.

## PARSeq (puerto `ocr`)

Es el modelo que lee el texto de la foto de una patente. Se usa preentrenado, sin reentrenar. Código y pesos son
Apache 2.0: la licencia de los pesos la declara el autor en la ficha `baudm/parseq-small` de Hugging Face
(<https://huggingface.co/baudm/parseq-small>), cuyo `pytorch_model.bin` es byte a byte el `parseq-bb5792a6.pt` de la
release v1.0.0 que baja este exportador (SHA-256 `bb5792a68e36...cd00`, verificado el 16/11/2026; ver
[THIRD_PARTY.md](../../THIRD_PARTY.md)). Por eso figura como `yes`.

Limitación: lee una sola línea de texto. Las patentes de dos líneas (motos) están fuera de alcance.

1. Con las dependencias y `MODELS_DIR` ya listas (ver "Antes de empezar"), corré:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/export/parseq.py
   ```

   La primera vez baja de GitHub el código de PARSeq (en un commit fijo, a `~\.cache\torch\hub`) y
   los pesos (unos 91 MB, con verificación de hash). Tarda un par de minutos. Exporta siempre en
   CPU: no hace falta GPU.

2. Las últimas líneas tienen que parecerse a estas:

   ```text
   Diferencia máxima entre ONNX (CPU) y el forward original de PARSeq: 9.54e-06
   Listo: models\weights\parseq.onnx (96.7 MB)
   sha256: 7ff186b1f7433e62980d01d647dce74ab338ebe43eefe6ec5e031a2c06315991
   ```

   La primera línea es el control de calidad: el script compara los logits del ONNX con los del
   modelo original y también que lean los mismos caracteres. Si la diferencia supera 0,001 no
   guarda el archivo.

3. Compará el `sha256` con el de `parseq` en `models/manifest.json`. Si no coincide, mismo criterio
   que con DINOv2.

4. Probá el modelo real: `uv run --frozen --extra dev --extra tools pytest -m models`. Esas pruebas
   dibujan texto con formato de patente en una imagen sintética; no usan fotos reales.

5. Para usarlo, en tu `.env`:

   ```dotenv
   MODEL_OCR=parseq
   MODELS_DIR=./models/weights
   ```

   `OCR_MIN_CONF` (0,60 por defecto) es la confianza mínima para aceptar una lectura. El servicio
   la compara con el mínimo de las probabilidades de los caracteres leídos; el valor definitivo se
   propone con la evaluación del OCR.

### Qué hace el adaptador con el texto leído

Si el texto contiene una patente argentina (Mercosur `AB123CD` o anterior `ABC123`), devuelve solo
esa parte y su confianza sale de sus propios caracteres. Tolera las confusiones típicas del OCR (O/0,
I/1, B/8, Z/2, S/5, G/6) dentro de la patente. Si no hay ninguna, devuelve el texto tal cual y la
confianza mínima de todo el texto.

### Problemas frecuentes (PARSeq)

- **Falla la descarga**: necesita internet solo esta vez, hacia `github.com` y `objects.githubusercontent.com`.
- **"Ya existe ... Usá --force"**: ya lo exportaste; si el sha256 coincide con el manifest, no hace
  falta repetirlo.
