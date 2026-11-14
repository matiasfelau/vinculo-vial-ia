# Ajustar el embedder de re-ID con 3DRealCar

El servicio decide que dos fotos son de **otro auto** (R2, `BURST_MISMATCH`) cuando el coseno entre sus
embeddings es menor que `REID_T`. Este modelo es DINOv2 ViT-S/14 (Apache 2.0) ajustado con
[3DRealCar](https://github.com/xiaobiaodu/3DRealCar_Dataset) para que ese coseno separe mejor a los autos. Tiene la
misma entrada y la misma salida que el `dinov2_vits14` sin ajustar, así que el adaptador del servicio es el mismo: se
elige con `MODEL_EMBEDDER=dinov2_vits14_reid`.

**El modelo ajustado no está activado.** `MODEL_EMBEDDER` y `REID_T` siguen como estaban; cambiarlos es una
decisión aparte (ver "Resultados" en [tools/eval/README.md](../../eval/README.md#re-id-con-3drealcar-reid_3drealcarpy)).

## Condiciones de 3DRealCar

- **Licencia:** Apache-2.0, con uso comercial permitido (el `README` del dataset lo dice y el repositorio trae el
  `LICENSE`). Detalle y cita en [THIRD_PARTY.md](../../../THIRD_PARTY.md).
- **Los datos nunca van al repositorio** ni a la imagen. Se guardan en `REID_DATA_DIR`, fuera de git.
- **Los pesos no van a git:** se montan desde `MODELS_DIR` en solo lectura y su `sha256` se registra en
  `models/manifest.json`.
- Solo se usan las fotos RGB y la pose de la cámara (para el ángulo de vista). Nada del escaneo 3D.
- **Cuota de Google Drive.** Cada tramo del dataset pesa 55 a 110 GB y se baja por pedidos por rango (solo las fotos
  elegidas), pero Drive limita las descargas por archivo y respondió "Quota exceeded" a mitad de la bajada. Quedaron 406
  autos completos (175 de luz estándar y 231 oscuros) y **ningún reflectivo**. `fetch.py pull` retoma lo que falta cuando la
  cuota se libera (el pedido espera entre reintentos y escribe cada archivo entero o no lo escribe).

## Qué ve el embedder

Lo mismo que en la cascada sin segmentador: la foto se reduce a `MODEL_MAX_SIDE_PX` (1024), el detector
(`rfdetr_openimages`) busca autos y se recorta la **caja `car` más grande**; el adaptador achica ese recorte a
224 x 224 (sin respetar la proporción) y lo normaliza. `prepare.py` guarda esos recortes en PNG (sin pérdida) y el
entrenamiento los lee con el mismo preproceso (`vvi_ia.adapters.models.dinov2.preprocess`).

## Pasos

```powershell
$env:REID_DATA_DIR = "D:\vv-eval\3drealcar"
$env:MODELS_DIR = "models\weights"
$env:EVAL_OUT = "D:\vv-eval\out\reid"
```

1. **Anotaciones, índice y selección** (red; no baja fotos):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py annotations
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py index
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py select --frames 12 --standard 600
   ```

   El dataset se distribuye en zips de 55 a 110 GB con todo el escaneo 3D. `index` lee solo el índice de cada zip
   (Google Drive acepta pedidos por rango) y `select` elige los autos: todos los de luz oscura y reflectiva y una
   muestra de los de luz estándar, con 12 vistas repartidas a lo largo de la caminata alrededor de cada uno.

2. **Bajar las vistas** (solo `frame_N.jpg` y `frame_N.json` de cada vista elegida):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py pull --threads 8
   ```

   Se puede repetir: no vuelve a bajar lo que ya está.

3. **Recortes, ángulos y partición:**

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/prepare.py --workers 3 --threads 2
   ```

   Escribe `crops/` y `crops.json`. La **partición** es agrupada por auto (ningún auto está en dos particiones),
   estratificada por luz, tipo de carrocería y color, con semilla fija (`--seed`, 0 por defecto): 70 % entrenamiento,
   15 % validación y 15 % prueba.

4. **Entrenar** (GTX 1660 de 6 GB, FP32, lote chico con acumulación; cerrá antes lo que use la GPU):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/train.py
   ```

5. **Exportar a ONNX y verificar** que reproduce a PyTorch:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/export.py --checkpoint "$env:EVAL_OUT\best.pt"
   ```

   Imprime el `sha256` para `models/manifest.json` y falla (borrando el archivo) si la diferencia máxima por
   componente del embedding pasa de 1e-3 o el coseno mínimo entre ONNX y PyTorch baja de 0,9999.

6. **Medir** contra el modelo sin ajustar, en 3DRealCar y en VeRi-776: ver
   [tools/eval/README.md](../../eval/README.md#re-id-con-3drealcar-reid_3drealcarpy).

## Pares, ángulos y métricas

- **Ángulo de vista.** Cada foto trae la pose de la cámara de ARKit. El auto es el punto del plano horizontal más
  cercano a todos los ejes ópticos y el ángulo de una vista es el azimut de la cámara alrededor de ese punto. Dos
  vistas del mismo auto se separan por la diferencia de azimut; no hace falta saber hacia dónde "mira" el auto.
- **Pares del mismo auto:** vistas con una separación angular mínima de 30°, y los resultados se reportan por rango
  (30-60°, 60-90°, 90-135° y 135-180°).
- **Pares de autos distintos:** todos (una muestra al azar) y **difíciles** (misma marca y mismo color).
- **U1:** pares del mismo auto marcados como distintos (coseno < umbral): penaliza a un usuario honesto.
  **S4:** pares de autos distintos detectados. Las cotas del 95 % son Clopper-Pearson (con 0 fallas, 3/N).
- **Calibración:** el mayor umbral cuya cota del 95 % de U1 sobre validación no pasa de 1 %; con eso fijo se reporta
  el S4 sobre prueba.

## Pérdida: CosFace

Se entrena con **CosFace** (margen aditivo en el coseno, `s = 30`, `m = 0,30`) sobre un prototipo por auto de
entrenamiento. Por qué:

- El servicio decide con un umbral sobre el coseno; CosFace separa a los autos justo en esa métrica, con un margen
  que se traduce en un `REID_T` con holgura.
- Los prototipos funcionan como negativos de todos los autos en cada paso. Con 6 GB de memoria el lote tiene que ser
  chico y se acumula gradiente: una pérdida por pares dentro del lote (triplet, contrastiva) vería muy pocos
  negativos y acumular no los aumenta, mientras que con prototipos el resultado es el mismo con lote 16 que con 64.
- Los prototipos arrancan en el promedio de los embeddings del modelo sin ajustar de cada auto, así que el ajuste
  parte de un modelo que ya separa y solo lo afina.

El embedding es el token CLS de la última capa, normalizado L2, igual que en el servicio. Aumentos: recorte
aleatorio (80-100 % del recorte, para imitar la imprecisión de la caja del detector), brillo, contraste, saturación
y espejo. Parada temprana sobre validación (S4 con U1 acotado a 1 %, paciencia de 3 épocas).

## Limitaciones

- **U1 es optimista.** Los pares del mismo auto salen de una misma sesión de captura (el mismo lugar, el mismo día y
  la misma luz): se parecen más entre sí que dos fotos tomadas en momentos distintos.
- **Puede haber diferencias con el parque automotor argentino.** Las marcas y modelos son los de las calles donde se
  capturó 3DRealCar (anotados en chino, p. ej. BYD o Changan).
- **No hay primeros planos de partes del auto.** Todas las vistas muestran el auto entero, caminando alrededor.
- Hay una foto de cada vista, no un par de fotos de personas distintas con celulares distintos: el cambio de
  cámara entre GENERAL y PLATE en una alerta real no está representado.
