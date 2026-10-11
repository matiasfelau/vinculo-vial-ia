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
