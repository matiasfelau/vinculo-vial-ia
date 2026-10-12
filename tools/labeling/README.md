# Preetiquetar rebajes con SAM 3

**Etiquetar** una foto es marcar, a mano, dónde está cada cosa que el modelo tiene que aprender: acá, el
cordón (`curb`) y el tramo rebajado del cordón (`curb_cut`). Hacerlo desde cero, píxel por píxel, es lo más
lento de entrenar un modelo. Por eso se **preetiqueta**: un modelo grande, SAM 3, hace un primer intento, y
vos lo corregís en CVAT en lugar de dibujar todo.

SAM 3 solo se usa acá, en `tools/`. No es parte del servicio. Sus propuestas son un punto de partida: **no
son la verdad** y hay que revisarlas una por una.

Las fotos tienen rostros y patentes de terceros: todo queda en tu carpeta, fuera del repo, y no se sube a
ninguna nube (SAM 3 corre en tu máquina).

## Antes de empezar (una sola vez)

1. Dependencias: seguí "Antes de empezar" de [tools/export/README.md](../export/README.md).
2. **Acceso a SAM 3.** El modelo `facebook/sam3` es de acceso restringido en Hugging Face. Con tu cuenta ya
   aprobada, iniciá sesión **en esta máquina**, una sola vez:

   ```powershell
   uv run --frozen --extra dev --extra tools hf auth login
   ```

   Te pide un token (Hugging Face → Settings → Access Tokens, con permiso de lectura). Se guarda en tu perfil
   de usuario, no en el repo. También sirve definir la variable de entorno `HF_TOKEN`. Para comprobar:
   `uv run --frozen --extra dev --extra tools hf auth whoami`.
3. La primera corrida baja el modelo (unos 3,5 GB) a `%USERPROFILE%\.cache\huggingface`.
4. Las fotos van en `EVAL_DATA_DIR\curb\raw\`:

   ```powershell
   $env:EVAL_DATA_DIR = "D:\vv-eval"
   ```

## 1. Probar los prompts

SAM 3 segmenta lo que le describís con texto, y **las palabras importan**. Antes de procesar todo, probá
variantes en unas pocas fotos (en inglés, que es como mejor entiende):

```powershell
uv run --frozen --extra dev --extra tools python tools/labeling/prelabel_curb.py --explore "curb" "street curb" "dropped curb" "curb ramp" "driveway apron" --explore-photos 5
```

Por cada prompt muestra en cuántas fotos encontró algo, cuántas instancias y con qué confianza media, y
guarda las fotos con lo que encontró en `curb\prelabel\explore\`. **Miralas:** el número solo no dice si el
rebaje es el que querés. Elegí el prompt que mejor marque el tramo rebajado (`curb_cut`) y el que mejor marque
el cordón sin el rebaje (`curb`).

## 2. Preetiquetar

```powershell
uv run --frozen --extra dev --extra tools python tools/labeling/prelabel_curb.py --curb-prompts "curb" --curb-cut-prompts "dropped curb" "curb ramp"
```

Si no pasás prompts usa `curb` para el cordón y `dropped curb` y `curb ramp` para el rebaje.

| Opción | Para qué |
|---|---|
| `--device cuda\|cpu` | GPU (rápido) o CPU (mucho más lento: sirve para dejarlo corriendo de noche). |
| `--max-images N` | Procesa como máximo N fotos nuevas por corrida. Para trabajar **por tandas**. |
| `--threshold 0.3` | Confianza mínima de SAM 3 para aceptar una instancia. Más alta: menos propuestas, más seguras. |
| `--simplify 1.5` | Qué tanto se simplifican los polígonos (en px). Más alto: menos puntos, menos fieles. |
| `--min-area 0.0005` | Instancias más chicas que esta fracción de la foto se descartan por ruido. |
| `--samples 6` | Cuántas fotos de muestra se guardan con las máscaras dibujadas. |

**Si se corta, no se pierde nada.** Cada foto se guarda al terminar. Al volver a correr el mismo comando,
sigue donde quedó. Si cambiás los prompts o los umbrales, empieza de nuevo (los resultados viejos quedan
guardados aparte).

En la memoria de la GPU, SAM 3 ocupa unos 3,5 GB: cerrá lo que use la GPU (juegos, navegadores) o usá
`--device cpu`.

Al final queda en `curb\prelabel\`:

- `prelabel_coco.zip`: lo que se importa en CVAT.
- `annotations\instances_default.json`: lo mismo, sin comprimir.
- `samples\`: fotos con las máscaras dibujadas (azul: cordón, naranja: rebaje) para mirar a ojo.

## 3. Corregir en CVAT

1. En CVAT creá una *task* con las fotos de `curb\raw\`. Las etiquetas tienen que llamarse **exactamente**
   `curb` y `curb_cut`, de tipo *Polygon* (no *Mask*).
2. **Menu de la task → Upload annotations → formato "COCO 1.0"** y elegí `prelabel_coco.zip`.
3. Foto por foto: borrá lo que esté mal, corregí los bordes y agregá lo que falte.
   - `curb` es el cordón visible, **sin** el tramo rebajado.
   - `curb_cut` es el tramo rebajado, sea de garage o de accesibilidad (es la misma clase).
   - Una foto sin cordón ni rebaje queda sin anotaciones: es válido, y ayuda al modelo.
4. Cuando termines: **Menu de la task → Export task dataset → formato "COCO 1.0"**, sin guardar las imágenes.
   Descomprimí el zip en `curb\cvat-export\`.

El resto del camino (el `fotos.csv`, el reparto y el entrenamiento) está en
[tools/training/curb/README.md](../training/curb/README.md).

## Problemas frecuentes

- **`GatedRepoError` o "Cannot access gated repo"**: no hay sesión de Hugging Face en esta máquina o la
  cuenta no tiene acceso. Ver "Acceso a SAM 3".
- **"No se pudo cargar SAM 3 (OutOfMemoryError...)"**: poca memoria de video libre. Cerrá programas o usá `--device cpu`.
- **Muchas fotos sin resultados**: bajá `--threshold` o probá otros prompts con `--explore`.
- **Los polígonos salen rotados o desplazados en CVAT**: la foto tiene una orientación guardada (EXIF) y CVAT la
  trata distinto. Probá exportando la foto ya girada.

## Validación técnica del preetiquetado

Se probó de punta a punta con **6 fotos de calle de Open Images** (no con fotos de rebajes: todavía no hay).
Sirve para saber que el pipeline funciona, **no** para saber qué tan bien marca SAM 3 tus rebajes.

| | |
|---|---|
| Velocidad | unos **5 segundos por foto** en la GTX 1660 |
| Memoria de la GPU | pico de **3,9 GB** (los pesos de SAM 3 en FP32 ocupan unos 3,4) |
| Retome | probado: una tanda de 3 fotos y otra que procesó solo las 3 que faltaban |
| COCO generado | 6 fotos, 21 instancias; cargó con `pycocotools` y las 21 máscaras reconstruidas coinciden con su área |

Prompts probados con `--explore` (6 fotos de autos en la calle):

| Prompt | Fotos con resultado | Instancias | Confianza media |
|---|---|---|---|
| `curb` | 5 de 6 | 10 | 0,57 |
| `street curb` | 5 de 6 | 12 | 0,56 |
| `dropped curb` | 4 de 6 | 11 | 0,52 |
| `curb ramp` | 0 | 0 | — |
| `driveway apron` | 0 | 0 | — |
| `sidewalk` | 6 de 6 | 13 | 0,60 |

Mirando las superposiciones, `curb` marca bien las tiras de cordón al pie de los autos (confianza 0,7 a 0,8) y
no inventa cordón donde no lo hay. `curb ramp` y `driveway apron` no encontraron nada en estas fotos; **no
significa que no sirvan** con tus fotos de rebajes, donde sí hay rampas: por eso conviene correr `--explore` con
tus fotos antes de procesar todo.
