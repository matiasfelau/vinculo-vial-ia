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
- **Descarga completa (15/11/2026).** Se bajaron los 10 zips desde la copia propia de Drive (`fetch.py pull --source drive`, un zip
  por vez, sin cortes de cuota): **2 009 autos usables** (1 321 de luz estándar, 424 reflectivos y 264 oscuros), con CRC y conteo de
  autos verificados por zip. 3DRealCar repite 17 autos entre `200-400.zip` y `1200-1400.zip`: la verificación acepta un auto de más si
  el índice lo tiene en otro zip con las mismas vistas. (La primera versión, con 411 autos, se había cortado por la cuota del folder
  público de Google y no tenía ningún reflectivo.) Cada tramo del dataset pesa 55 a 110 GB, pero solo se leen las vistas elegidas
  (1,4 a 3,6 GB por zip).

## Qué ve el embedder

Lo mismo que en la cascada sin segmentador: la foto se reduce a `MODEL_MAX_SIDE_PX` (1024), el detector busca vehículos
y se recorta la **caja del vehículo dominante** (la `car` más grande); el adaptador achica ese recorte a 224 x 224
(sin respetar la proporción) y lo normaliza. **Desde el 15/11/2026 el detector es v2** (`rfdetr_openimages_v2`, con la
confianza de auto del servicio, 0,35; v1 ya no se usa para esto) y **sin máscaras del segmentador**. `prepare.py` guarda
esos recortes en PNG (sin pérdida) en `crops-v2/` y escribe `crops_v2.json`; el entrenamiento los lee con el mismo
preproceso (`vvi_ia.adapters.models.dinov2.preprocess`). Los recortes de la primera versión (`crops/`, `crops.json`,
detector v1) quedan sin usar.

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

   **Desde una copia propia en Google Drive** (cuando el folder público corta por cuota): cada zip se copia con
   "Crear una copia" al Drive de quien entrena, y `fetch.py` lo lee por rangos con una cuenta de servicio de **solo
   lectura**. No baja el zip: lee el índice (unos 0,1 GB) y las vistas elegidas (1,5 a 3,5 GB por zip).

   Configuración, una sola vez:

   1. En Google Cloud Console: un proyecto con la API **Google Drive** habilitada y una **cuenta de servicio** sin
      roles. Creale una clave JSON y guardala **fuera del repositorio** (`fetch.py` se niega a leerla desde dentro).
   2. En Drive: una carpeta (p. ej. `3drealcar-copias`) compartida con el correo de la cuenta de servicio como
      **Lector**. Ve solo esa carpeta, no puede escribir ni borrar.
   3. Copiá cada zip del folder público con "Crear una copia" y movelo a esa carpeta. El nombre
      `Copia de 1400-1600.zip` se reconoce solo; si hay dos copias del mismo zip, se frena.
   4. Variables (el id es lo que sigue a `/folders/` en la URL de la carpeta):

      ```powershell
      $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\<vos>\.vv\drive-lector.json"
      $env:REID_DRIVE_FOLDER_ID = "<id de la carpeta>"
      ```

   Después, para sumar todos los autos (las vistas de los ya elegidos no cambian) y bajar un zip por vez:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py select --frames 12 --standard 2000
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py drive-check --source drive
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py pull --source drive --only 1400-1600
   uv run --frozen --extra dev --extra tools python tools/training/reid/fetch.py status
   ```

   - `drive-check` valida el acceso (lista la carpeta y lee 1 KiB al final de cada zip) y muestra el MD5 de cada
     copia, para compararlo con el de Drive.
   - `pull` verifica antes de bajar: el zip abre, tiene los mismos autos que `index.json` y las vistas elegidas
     están donde el índice dice. Cada foto y cada pose se comparan contra su CRC32. Si algo no coincide, no baja
     nada de ese zip. `verify --source drive` hace solo esa verificación.
   - Al terminar un zip dice **"ya podés borrar su copia de Drive"**. Si Google corta por cuota termina con código
     3 y lo bajado se conserva: se retoma después con el mismo comando.
   - Al terminar, borrá la clave de la cuenta de servicio y dejá de compartir la carpeta.

3. **Recortes, ángulos y partición** (después de bajar todo: la partición se congela al correrlo):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/prepare.py --workers 3 --threads 2
   ```

   Escribe `crops-v2/` y `crops_v2.json` (solo entran los autos con todas sus vistas elegidas bajadas). La **partición** es agrupada por auto (ningún auto está en dos particiones),
   estratificada por luz, tipo de carrocería y color, con semilla fija (`--seed`, 0 por defecto): 70 % entrenamiento,
   15 % validación y 15 % prueba. La partición se **congela** en `split.json`: al sumar autos (otro zip, más autos
   estándar) los que ya estaban no cambian de partición y solo se sortean los nuevos, de modo que cada estrato
   vuelva a quedar en 70/15/15. Para empezar de cero hay que borrar `split.json` y `crops.json`.

   **Pares de patente** (validación y prueba; ver "Pares de patente" abajo):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/partial.py
   ```

   Escribe `partial-v2/`, `partial_v2.json` y `partial_v2_stats.json` (solo patentes).

4. **Entrenar** (GTX 1660 de 6 GB, FP32, lote chico con acumulación; cerrá antes lo que use la GPU):

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/train.py
   ```

   `train.py` guarda `last.pt` en cada época: si la corrida se corta (por ejemplo, por falta de memoria de la máquina), se sigue
   donde quedó con `--resume`.

5. **Exportar a ONNX y verificar** que reproduce a PyTorch:

   ```powershell
   uv run --frozen --extra dev --extra tools python tools/training/reid/export.py --checkpoint "$env:EVAL_OUT\best.pt"
   ```

   Imprime el `sha256` para `models/manifest.json` y falla (borrando el archivo) si la diferencia máxima por
   componente del embedding pasa de 1e-3 o el coseno mínimo entre ONNX y PyTorch baja de 0,9999.

6. **Medir** contra el modelo sin ajustar, con los pares general-patente y en VeRi-776: ver
   [tools/eval/README.md](../../eval/README.md#re-id-con-3drealcar-general-contra-patente-reid_parcialespy).

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

## Pares de patente

R2 compara la foto GENERAL contra la de PLATE (la trompa o la cola, cortada entre el capot y el parabrisas, con la patente
al 8-15 % del ancho); **no compara DETAIL ni CONTEXT** (decisión del 15/11/2026: la foto de detalle probablemente se elimine, así
que no se calibra, no se mide y no se simula). 3DRealCar solo tiene vistas del auto entero, así que `partial.py` **simula** la foto de
patente a partir de ellas, solo para **validación y prueba** (el entrenamiento sigue con autos enteros):

1. **Vistas frontales y traseras por la pose.** El auto es una caja más ancha de costado que de frente. Con el azimut de
   cada vista (de la pose de ARKit) y la relación ancho/alto de la caja del vehículo, se ajusta `aspecto ≈ c + a·cos(2·az) +
   b·sin(2·az)`: el máximo cae en el azimut de la vista de costado y las frontales y traseras están a ±90°. Se aceptan las
   vistas a menos de 30° de ese eje. Si el ajuste es débil (amplitud < 20 % del aspecto medio o R² < 0,5), el auto no
   aporta pares de patente.
2. **Patente.** De esas vistas, las que el detector v2 ve con una patente (confianza >= 0,30) dentro de la caja del vehículo
   y con un ancho de 10 a 30 % del ancho del vehículo; hasta 2 por auto (las de mayor confianza, separadas al menos 30°).
   Se recorta la ventana 16:9 de `capture_sets.plate_window` (la misma de los conjuntos de R1): patente al 8-15 % del
   ancho, vehículo entero a lo ancho, corte a la altura del capot.
3. El recorte se lleva a 1600 px y JPEG de <= 900 KB (lo que envía la app) y pasa por el mismo camino que una foto real:
   se reduce a 1024 px, el detector busca el vehículo dominante y se recorta su caja, **sin máscaras**. Si no encuentra un
   vehículo, el servicio no tendría embedding (duda, GREEN): se cuenta como "sin vehículo" y no entra a los pares.

Pares: **general contra patente** del mismo auto con una separación mínima de 30° entre la vista entera y la vista de la que
salió el recorte de patente, y de autos distintos (todos y difíciles: misma marca y color). Los pares de auto entero se
arman como siempre y quedan solo como referencia.

**Umbral.** Un **único umbral** calibrado en validación con los pares general-patente. Activo (15/11/2026): **0,2565, el mayor con la cota del
95 % de U1 en 1 % o menos** (en prueba: U1 3 de 778, cota 0,99 %, S4 98,0 %). Opción conservadora: **0,0638, el menor con S4 >= 70 %** (U1 0 de 778, S4 69,6 %).
Como hay una sola comparación por alerta, el U1 por par es también el U1 por alerta. R2 solo rechaza con certeza; ante la duda, GREEN. Resultados y
límites en [tools/eval/README.md](../../eval/README.md#re-id-con-3drealcar-general-contra-patente-reid_parcialespy).

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
- **Los pares parciales son simulados.** Salen de recortes de vistas del auto entero, no de fotos tomadas con el encuadre
  de la app: la resolución, el desenfoque y la iluminación de una foto de patente real son otros.
- **No hay primeros planos reales de partes del auto.** Todas las vistas muestran el auto entero, caminando alrededor.
- Hay una foto de cada vista, no un par de fotos de personas distintas con celulares distintos: el cambio de
  cámara entre GENERAL y PLATE en una alerta real no está representado.
