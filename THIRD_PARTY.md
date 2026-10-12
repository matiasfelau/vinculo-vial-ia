# Modelos y datasets de terceros

Licencias, papers y atribuciones de todo lo que usa `vinculo-vial-ia`. Los pesos y los datasets
nunca están en este repositorio ni en la imagen: los pesos se montan desde `MODELS_DIR` y se
verifican contra [models/manifest.json](models/manifest.json).

El campo `commercial` del manifest usa tres valores:

| Valor | Significado |
|---|---|
| `yes` | Licencia que permite uso comercial. |
| `escalable` | Uso académico hoy; el titular puede autorizar el uso comercial. |
| `no` | Sin camino comercial. No se puede seleccionar en el servicio; solo se usa en `tools/` para comparar. |

## Modelos del servicio

| Puerto | Modelo | Paper | Licencia | `commercial` |
|---|---|---|---|---|
| `detector`, `face` | RF-DETR (Small o Medium), entrenado con Open Images V7 | Robinson, Robicheaux, Popov, Ramanan y Peri. *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers*. ICLR 2026 | Código y pesos base Apache 2.0. Datos de entrenamiento CC BY 4.0 | `yes` |
| `segmenter` | RF-DETR-Seg (Small o Medium), pesos preentrenados con COCO | Mismo paper, ICLR 2026 | Apache 2.0 | `yes` |
| `curb` | RF-DETR-Seg Small, entrenado con el dataset propio de rebajes | Mismo paper, ICLR 2026 | Apache 2.0. Datos propios | `yes` |
| `embedder` | DINOv2 ViT-S/14, sin entrenar | Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024 | Apache 2.0 | `yes` |
| `ocr` | PARSeq, pesos preentrenados | Bautista y Atienza. *Scene Text Recognition with Permuted Autoregressive Sequence Models*. ECCV 2022 | Código Apache 2.0. El repositorio no declara una licencia para los pesos | `escalable` |
| `damage` | RF-DETR, entrenado con CarDD | RF-DETR: ICLR 2026. Datos: ver CarDD | Modelo Apache 2.0. Los pesos heredan las condiciones de CarDD | `escalable` |

Notas:

- **RF-DETR:** solo se usan los tamaños con licencia Apache 2.0. Los tamaños XL y 2XL se
  distribuyen con la licencia PML 1.0 y no se usan en ninguna parte del proyecto.
- **Detector (`detector` y `face`):** RF-DETR Small ajustado por este proyecto con un subconjunto de Open Images
  V7 (clases Car, Wheel, Vehicle registration plate y Human face), partiendo de los pesos COCO de Roboflow
  (Apache 2.0). Los pesos resultantes son propios y no se redistribuyen con el repositorio; se generan con
  [tools/training/detector/](tools/training/detector/README.md). Open Images: anotaciones CC BY 4.0 e imágenes CC BY
  2.0 (ver la sección del dataset).
- **DINOv2:** pesos `facebook/dinov2-small` publicados por Meta en Hugging Face
  (<https://huggingface.co/facebook/dinov2-small>), convertidos a ONNX sin modificar con
  [tools/export/dinov2.py](tools/export/dinov2.py).
- **RF-DETR-Seg (`segmenter`):** pesos de RF-DETR-Seg Small entrenados con COCO que publica
  Roboflow (`https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth`), convertidos a ONNX sin
  modificar con [tools/export/rfdetr_seg.py](tools/export/rfdetr_seg.py). Solo se aceptan los
  tamaños Small y Medium.
- **PARSeq:** antes de un uso comercial hay que confirmar con los autores la licencia de los pesos.
  Se usan los pesos de la release v1.0.0 de <https://github.com/baudm/parseq> (`parseq-bb5792a6.pt`),
  convertidos a ONNX sin reentrenar con [tools/export/parseq.py](tools/export/parseq.py), que carga el
  código del commit `1902db043c02`. Lee una sola línea de texto: las patentes de dos líneas (motos)
  están fuera de alcance.
- **`damage`:** los pesos entrenados con CarDD no se publican ni se redistribuyen.

## Modelos que solo viven en `tools/`

No van al servicio. Se usan para preetiquetar o como referencia en las evaluaciones.

| Modelo | Uso | Paper | Licencia | `commercial` |
|---|---|---|---|---|
| SAM 3 (`facebook/sam3`) | Preetiquetado de cordones y rebajes, y referencia sin entrenar en la evaluación de `curb` | Carion et al. *SAM 3: Segment Anything with Concepts*. ICLR 2026 | SAM License | No aplica: no se sirve |
| SCRFD | Comparación de detectores de rostros | Guo et al. *Sample and Computation Redistribution for Efficient Face Detection*. ICLR 2022 | Solo investigación. InsightFace ofrece licencias comerciales | `escalable`: solo entra al servicio si gana la comparación y se confirma esa licencia |
| RetinaFace-R50 (biubug6) | Comparación de detectores de rostros | Deng et al. *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild*. CVPR 2020 | Código MIT. Entrenado con WIDER FACE (CC BY-NC-ND 4.0) | `no` |
| YuNet | Comparación de detectores de rostros | Wu et al. *YuNet: A Tiny Millisecond-level Face Detector*. Machine Intelligence Research, 2023 | Código MIT. Entrenado con WIDER FACE (CC BY-NC-ND 4.0) | `no` |

## Datasets

### Open Images V7

- **Uso:** entrenamiento de `detector` y `face` con las clases Car, Wheel, Vehicle registration
  plate y Human face.
- **Licencia:** anotaciones CC BY 4.0. Las imágenes figuran con licencia CC BY 2.0.
- **Cita:** Kuznetsova et al. *The Open Images Dataset V4: Unified image classification, object
  detection, and visual relationship detection at scale*. IJCV, 2020.
- **Sitio:** <https://storage.googleapis.com/openimages/web/index.html>

### COCO

- **Uso:** solo a través de los pesos preentrenados de RF-DETR-Seg. El dataset no se descarga.
- **Licencia:** anotaciones CC BY 4.0.
- **Cita:** Lin et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.

### CarDD

- **Uso:** entrenamiento de `damage`.
- **Titular:** PIC Lab (Chinese Academy of Sciences).
- **Condiciones:** uso para investigación, con consentimiento previo del titular (ya otorgado).
  El uso comercial, incluido probar sistemas comerciales, requiere autorización previa del
  PIC Lab. No se puede redistribuir a terceros.
- **En este proyecto:** el dataset nunca entra al repositorio, a la imagen ni a la nube, y los
  pesos entrenados con él no se publican.
- **Cita:** Wang, Li y Wu. *CarDD: A New Dataset for Vision-Based Car Damage Detection*. IEEE
  Transactions on Intelligent Transportation Systems, 24(7):7202-7214, 2023.

### "patentes" (Roboflow Universe)

- **Uso:** evaluación del OCR. No se usa para entrenar.
- **Proyecto:** `patentes-obs9r/patentes-n81go`,
  <https://universe.roboflow.com/patentes-obs9r/patentes-n81go>
- **Licencia:** CC BY 4.0.
- **Atribución:** dataset "patentes", publicado en Roboflow Universe por el workspace
  `patentes-obs9r`.
- **Cambios respecto del original:** las clases se unificaron, las imágenes se recortaron por
  caja y las etiquetas se normalizaron (mayúsculas, solo letras y dígitos). Además, se
  corrigieron etiquetas y se excluyeron fotos ilegibles mediante un archivo de correcciones
  (`label_overrides.csv`) que se guarda fuera del repositorio; el export original no se modificó.

### Dataset propio de rebajes

- **Uso:** entrenamiento y evaluación de `curb`.
- **Condiciones:** fotos propias con rostros y patentes de terceros. No se versionan y se guardan
  fuera del repositorio; nunca se suben a la nube.
