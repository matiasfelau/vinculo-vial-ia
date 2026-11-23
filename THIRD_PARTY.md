# Componentes y datos de terceros

Licencias, papers y atribuciones de todo lo que usa `vinculo-vial-ia`. Los pesos y los datasets
nunca están en este repositorio ni en la imagen: los pesos se montan desde `MODELS_DIR` y se
verifican contra [models/manifest.json](models/manifest.json).

**Criterio del proyecto (16/11/2026).** El producto final solo usa componentes y datos con camino comercial, también
para elegir modelos y calibrar umbrales. Lo único que puede faltar es pagar licencias. Lo que no cumple el criterio
queda como **módulo experimental**, fuera del producto (sección 2): el código está en el repo, el adaptador es el nulo por
defecto y la cascada no lo llama.

El campo `commercial` del manifest usa tres valores:

| Valor | Significado |
|---|---|
| `yes` | Licencia que permite uso comercial. |
| `escalable` | Uso académico hoy; el titular puede autorizar el uso comercial (puede requerir pagar una licencia). |
| `no` | Sin camino comercial. Solo se usa en `tools/` para comparar. |

## 1. El producto

### 1.1 Modelos

| Puerto | Modelo (`models/manifest.json`) | Qué es | Licencia | Camino comercial |
|---|---|---|---|---|
| `detector` | `rfdetr_openimages_v2` (activo) y `rfdetr_openimages` (v1, respaldo) | RF-DETR Small ajustado por este proyecto con un subconjunto de Open Images V7, partiendo de los pesos COCO de Roboflow | Código y pesos base Apache-2.0; anotaciones de Open Images CC BY 4.0 | `yes` |
| `segmenter` | `rfdetr_seg_coco` | RF-DETR-Seg Small con los pesos COCO que publica Roboflow, sin reentrenar | Apache-2.0 | `yes` |
| `embedder` | `dinov2_vits14_reid_v2` (activo); `dinov2_vits14` y `dinov2_vits14_reid` (no activados) | DINOv2 ViT-S/14 (Meta) ajustado por este proyecto con 3DRealCar y una pérdida CosFace | Apache-2.0 (DINOv2 y 3DRealCar) | `yes` |
| `ocr` | `parseq` | PARSeq preentrenado, sin reentrenar (`baudm/parseq-small` en Hugging Face) | Apache-2.0, código y pesos (la declara el autor en la ficha del modelo) | `yes` |

Papers: RF-DETR (Robinson, Robicheaux, Popov, Ramanan y Peri. *RF-DETR: Neural Architecture Search for Real-Time
Detection Transformers*. ICLR 2026); DINOv2 (Oquab et al. *DINOv2: Learning Robust Visual Features without
Supervision*. TMLR 2024); PARSeq (Bautista y Atienza. *Scene Text Recognition with Permuted Autoregressive Sequence
Models*. ECCV 2022).

Notas:

- **RF-DETR:** solo se usan los tamaños con licencia Apache 2.0. Los tamaños XL y 2XL se distribuyen con la licencia
  PML 1.0 y no se usan en ninguna parte del proyecto.
- **Detector (`rfdetr_openimages_v2`, el activo desde el 14/11/2026):** RF-DETR Small ajustado con un subconjunto de
  Open Images V7 (autos, camiones, vans, colectivos, patentes y confusoras), sin datos de COCO ni de otras fuentes, con
  [tools/training/detector_v2/](tools/training/detector_v2/README.md). v1 (`rfdetr_openimages`) queda como respaldo. Los
  pesos son propios y no se redistribuyen con el repositorio.
- **Segmentador:** pesos de RF-DETR-Seg Small entrenados con COCO que publica Roboflow
  (`https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth`), convertidos a ONNX sin modificar con
  [tools/export/rfdetr_seg.py](tools/export/rfdetr_seg.py). Solo se aceptan los tamaños Small y Medium. Entra a la
  cascada (R1: presencia de vehículo, detector o segmentador).
- **Embedder:** pesos `facebook/dinov2-small` publicados por Meta en Hugging Face
  (<https://huggingface.co/facebook/dinov2-small>). `dinov2_vits14` es ese modelo convertido a ONNX sin modificar
  ([tools/export/dinov2.py](tools/export/dinov2.py)); `dinov2_vits14_reid_v2` es el ajustado con los 2.009 autos de
  3DRealCar ([tools/training/reid/](tools/training/reid/README.md)). El umbral de re-ID (`REID_T`) sale solo de 3DRealCar. Los
  pesos ajustados no se redistribuyen.
- **PARSeq (verificado el 16/11/2026):** los pesos son `pytorch_model.bin` de <https://huggingface.co/baudm/parseq-small>
  (revisión `a1526c3d63740e460153987f9aaf6b86aa199dc1`; la ficha declara `license: apache-2.0`), con SHA-256
  `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`. Ese archivo es **byte a byte** el `parseq-bb5792a6.pt` de
  la release v1.0.0 de <https://github.com/baudm/parseq> que usa [tools/export/parseq.py](tools/export/parseq.py) (el
  mismo SHA-256), y reexportar el modelo desde esos pesos reproduce el ONNX del servicio: SHA-256
  `7ff186b1f7433e62980d01d647dce74ab338ebe43eefe6ec5e031a2c06315991` (el de `models/manifest.json`). Lee una sola línea de
  texto: las patentes de dos líneas (motos) están fuera de alcance. Los pesos se entrenaron con datasets reales de texto en
  escena (los del autor, ver `Datasets.md` de su repositorio) cuyas licencias propias este proyecto no auditó: se confía en
  la licencia que declara el autor. (La ficha `baudm/parseq-tiny`, también Apache-2.0, es otro modelo y no se usa.)

### 1.2 Bibliotecas del servicio

Dependencias de ejecución de `pyproject.toml` (licencia según los metadatos de cada paquete, versiones instaladas el
16/11/2026). Las herramientas de entrenamiento y evaluación (`tools`) no entran a la imagen.

| Biblioteca | Licencia | Camino comercial |
|---|---|---|
| aio-pika | Apache-2.0 | `yes` |
| aiohttp | Apache-2.0 y MIT | `yes` |
| boto3 | Apache-2.0 | `yes` |
| numpy | BSD-3-Clause (con 0BSD, MIT, Zlib y CC0-1.0 en componentes incluidos) | `yes` |
| onnxruntime | MIT | `yes` |
| opencv-python-headless | Apache-2.0 | `yes` |
| pydantic y pydantic-settings | MIT | `yes` |
| pyhpke | MIT | `yes` |
| rapidfuzz | MIT | `yes` |

### 1.3 Datasets usados para entrenar, elegir o calibrar el producto

| Dataset | Para qué se usa | Licencia | Camino comercial |
|---|---|---|---|
| Open Images V7 | Ajuste del detector (v1 y v2); evaluación del detector y de la presencia de vehículo (R1); umbrales `DETECTOR_MIN_CONF`, `SEGMENTER_MIN_CONF`, `PRESENCE_MIN_AREA` y `PLATE_REDACT_MIN_CONF` | Anotaciones CC BY 4.0; imágenes CC BY 2.0 | `yes`, con atribución |
| COCO | Solo a través de los pesos preentrenados de RF-DETR-Seg (el dataset no se descarga) | Anotaciones CC BY 4.0; las imágenes tienen las licencias de Flickr de cada foto | Los pesos los publica Roboflow con Apache-2.0; las licencias de las imágenes no se auditaron |
| 3DRealCar | Ajuste de la re-identificación (`dinov2_vits14_reid_v2`) y calibración de `REID_T` | Apache-2.0 | `yes` |
| "patentes" (Roboflow Universe) | Evaluación del OCR y calibración de `OCR_MIN_CONF`; no se usa para entrenar | CC BY 4.0 | `yes`, con atribución |
| "Patentes Argentinas - Master" (Roboflow Universe) | Evaluación del OCR (300 patentes recortadas y etiquetadas a mano); no se usa para entrenar | Public Domain, declarada por quien lo publicó | `yes`; el origen de las fotos no se auditó (solo se evalúa con él y no se distribuye) |
| Fotos de referencia propias | Solo para medir el encuadre de captura (1-2 m); no calibran ningún umbral y no entran a ningún conjunto | Propias | No aplica |

Los datasets viven fuera del repositorio y de la imagen.

#### Open Images V7

- **Uso:** ajuste de `detector` v1 (clases Car, Wheel, Vehicle registration plate y Human face) y del detector v2
  (`rfdetr_openimages_v2`, solo `detector`) con Car, Limousine, Taxi, Truck, Van, Ambulance, Bus, Vehicle registration plate
  y las confusoras Airplane, Boat (y Barge, Gondola, Canoe), Bicycle, Motorcycle, Wheelchair, Traffic sign (y Stop sign) y
  Clock (y sus variantes), más fotos de fondo. La partición de prueba, que ningún entrenamiento vio, sirve para evaluar
  el detector y la presencia de vehículo, y para elegir los umbrales.
- **Licencia:** anotaciones CC BY 4.0. Las imágenes figuran con licencia CC BY 2.0.
- **Cita:** Kuznetsova et al. *The Open Images Dataset V4: Unified image classification, object
  detection, and visual relationship detection at scale*. IJCV, 2020.
- **Sitio:** <https://storage.googleapis.com/openimages/web/index.html>

#### COCO

- **Uso:** solo a través de los pesos preentrenados de RF-DETR-Seg y de RF-DETR (los pesos de partida del detector). El
  dataset no se descarga.
- **Licencia:** anotaciones CC BY 4.0. Las imágenes provienen de Flickr y cada una tiene su licencia (algunas no
  permiten uso comercial): este proyecto no las usa; usa los pesos que Roboflow publica con Apache-2.0.
- **Cita:** Lin et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.

#### 3DRealCar

- **Uso:** ajuste y evaluación de la re-identificación del `embedder` (DINOv2 ViT-S/14), y calibración de `REID_T`. Solo se
  usan las fotos RGB (`frame_N.jpg`) y la pose de la cámara de cada una; no se usa el escaneo 3D (depth, mallas, mapas del
  escáner).
- **Licencia:** Apache-2.0. El `README` del repositorio dice "3DRealCar dataset is allowed commercial usage for
  free. The dataset license is Apache-2.0", y el `LICENSE` del repositorio es el texto de Apache-2.0 (verificado
  el 10/11/2026 en <https://github.com/xiaobiaodu/3DRealCar_Dataset>).
- **Cita:** Du, Sun, Wang, Wu, Sheng, Ying, Lu, Zhu, Zhan y Yu. *3DRealCar: An In-the-wild RGB-D Car Dataset with
  360-degree Views*. ICCV 2025 (arXiv:2406.04875).
- **Sitio:** <https://xiaobiaodu.github.io/3drealcar/>
- **En este proyecto:** los datos viven fuera del repositorio (`REID_DATA_DIR`) y no se versionan ni entran a la
  imagen. Los pesos ajustados con él no van a git: se montan desde `MODELS_DIR` y se verifican contra el `sha256` del
  manifest.
- **Límites:** cada auto se capturó en una sola sesión (el mismo lugar, el mismo día), así que los pares del mismo
  auto son más fáciles que los de la vida real; hay diferencias con el parque automotor argentino (las
  marcas y los modelos están anotados en chino, p. ej. BYD o Changan, y son los de las calles donde se capturó); y no hay
  primeros planos de partes del auto.

#### "patentes" (Roboflow Universe)

- **Uso:** evaluación del OCR y calibración de `OCR_MIN_CONF`. No se usa para entrenar.
- **Proyecto:** `patentes-obs9r/patentes-n81go`,
  <https://universe.roboflow.com/patentes-obs9r/patentes-n81go>
- **Licencia:** CC BY 4.0.
- **Atribución:** dataset "patentes", publicado en Roboflow Universe por el workspace
  `patentes-obs9r`.
- **Cambios respecto del original:** las clases se unificaron, las imágenes se recortaron por
  caja y las etiquetas se normalizaron (mayúsculas, solo letras y dígitos). Además, se
  corrigieron etiquetas y se excluyeron fotos ilegibles mediante un archivo de correcciones
  (`label_overrides.csv`) que se guarda fuera del repositorio; el export original no se modificó.

#### "Patentes Argentinas - Master" (Roboflow Universe)

- **Uso:** evaluación del OCR: 300 patentes recortadas y etiquetadas a mano para medir la lectura de PARSeq (conjunto
  armado con `tools/eval/ocr_set.py`). No se usa para entrenar.
- **Proyecto:** `xenon-bo5e9/patentes-argentinas-master`,
  <https://universe.roboflow.com/xenon-bo5e9/patentes-argentinas-master>. Versión v1 del 19/06/2025, exportada en
  formato COCO el 08/01/2026: 656 fotos y 800 cajas de una sola clase (`patente`).
- **Licencia:** Public Domain, según la declara quien lo publicó en Roboflow Universe.
- **Cita:** xenon. *Patentes Argentinas - Master* (v1, 2025-06-19). Roboflow Universe.
  <https://universe.roboflow.com/xenon-bo5e9/patentes-argentinas-master>. Lo publica un usuario de Roboflow; no tiene
  artículo asociado.
- **Cambios respecto del original:** cada caja se recortó con un 15 % de margen; se descartaron las cajas de menos de
  40x24 píxeles, las fotos y las patentes que ya están en el dataset "patentes" (con el que se calibró el OCR) y los
  duplicados y casi duplicados; de lo que quedó se eligieron 300 con semilla fija, al menos 50 de noche. El export
  original no se modificó. Los recortes, las etiquetas y las hojas de contacto viven fuera del repositorio
  (`EVAL_DATA_DIR/ocr-ar-set`).
- **Límites:** "Public Domain" lo declara quien subió el dataset, que no necesariamente es el titular de las fotos:
  muchas parecen tomadas de la web (se ven marcas de agua y muestras de patentes), y no se auditó su origen. Por eso se
  usa solo para evaluar y no se distribuye ni se versiona. Incluye patentes de dos líneas (motos), que el servicio no
  lee, y muestras gráficas de patentes que no son fotos.

## 2. Módulos experimentales (fuera del producto)

No cumplen el criterio de camino comercial sin pagar una licencia o sin autorización del titular, o todavía no tienen
un uso en el producto. El código y las herramientas están en el repo, el adaptador es el nulo por defecto (o no se carga)
y **la cascada no los llama**: no influyen en el veredicto ni en la redacción.

| Módulo | Modelo | Datos | Licencia | `commercial` |
|---|---|---|---|---|
| Daños (`damage`, `MODEL_DAMAGE`) | `rfdetr_cardd`: RF-DETR Small ajustado con CarDD | CarDD | Modelo Apache-2.0; los pesos heredan las condiciones de CarDD | `escalable` |
| Rebajes (`curb`, `MODEL_CURB`) | RF-DETR-Seg Small ajustado con un dataset propio (pesos no registrados en el manifest) | Dataset propio de rebajes; preetiquetado con SAM 3 | Apache-2.0; datos propios; SAM License (solo preetiqueta, no se sirve) | `yes` (el modelo), sin uso en el producto |
| Rostros (`face`, `MODEL_FACE`) | `scrfd_10g`: SCRFD-10G de InsightFace, sin reentrenar | Los de los pesos de InsightFace | Pesos para investigación; InsightFace ofrece licencias comerciales | `escalable` |

El producto **no redacta rostros** (riesgo aceptado: las fotos se toman a 1-2 m centradas en el vehículo); solo tapa
patentes de terceros.

### Daños (CarDD)

- **Uso:** entrenamiento de `damage`. Sin uso en la cascada: el daño no cuenta como presencia ni en ninguna regla.
- **Titular:** PIC Lab (Chinese Academy of Sciences).
- **Condiciones:** uso para investigación, con consentimiento previo del titular (ya otorgado).
  El uso comercial, incluido probar sistemas comerciales, requiere autorización previa del
  PIC Lab. No se puede redistribuir a terceros.
- **En este proyecto:** el dataset nunca entra al repositorio, a la imagen ni a la nube, y los
  pesos entrenados con él no se publican. Se generan con [tools/training/damage/](tools/training/damage/README.md) a partir
  de un CarDD que cada persona tiene en su máquina.
- **Cita:** Wang, Li y Wu. *CarDD: A New Dataset for Vision-Based Car Damage Detection*. IEEE
  Transactions on Intelligent Transportation Systems, 24(7):7202-7214, 2023.
- **Modelo:** RF-DETR (ICLR 2026), Apache 2.0.

### Rebajes

- **Uso:** entrenamiento y evaluación de `curb` (el chequeo de rebajes salió del veredicto).
- **Condiciones del dataset propio:** fotos propias con rostros y patentes de terceros. No se versionan y se guardan
  fuera del repositorio; nunca se suben a la nube.
- **SAM 3** (`facebook/sam3`): preetiquetado de cordones y rebajes, y referencia sin entrenar en la evaluación de `curb`.
  Carion et al. *SAM 3: Segment Anything with Concepts*. ICLR 2026. SAM License. No se sirve.

### Rostros (SCRFD)

- **SCRFD-10G (`scrfd_10g`):** el `det_10g.onnx` del paquete `buffalo_l` de InsightFace, sin modificar, copiado de
  <https://huggingface.co/public-data/insightface>. Guo et al. *Sample and Computation Redistribution for Efficient Face
  Detection*. ICLR 2022. Los pesos son para investigación y el uso comercial requiere la licencia de InsightFace.
- **Alternativa con camino comercial:** `rfdetr_openimages` (v1, `commercial = yes`) también detecta rostros (clase
  *Human face* de Open Images), con peor recall en caras chicas (ver [tools/eval/README.md](tools/eval/README.md)).
- **Comparadores solo en `tools/` (`commercial = no`):** RetinaFace-R50 de biubug6 (Deng et al. *RetinaFace: Single-Shot
  Multi-Level Face Localisation in the Wild*. CVPR 2020; código MIT, sin licencia comercial confirmada para los pesos) y
  YuNet (Wu et al. *YuNet: A Tiny Millisecond-level Face Detector*. Machine Intelligence Research, 2023; código MIT, sin
  licencia comercial confirmada para los pesos). Se cargan desde un ONNX que tiene cada persona y no se redistribuyen.
- **Datos de evaluación:** el split de prueba de Open Images con solo la clase *Human face* (ver arriba).
