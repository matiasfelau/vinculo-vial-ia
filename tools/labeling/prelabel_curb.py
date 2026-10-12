"""Preetiqueta con SAM 3 las fotos de rebajes: cordón (`curb`) y tramo rebajado (`curb_cut`).

SAM 3 (`facebook/sam3`, con acceso otorgado) segmenta lo que describe un prompt de texto. Acá se corre sobre
las fotos de `<curb>/raw/` con los prompts de cada clase y se genera un COCO con polígonos, listo para
importar en CVAT como "COCO 1.0" y **corregir a mano**: es un punto de partida, no la verdad. SAM 3 solo vive
en `tools/`: no se sirve en el servicio.

- **Tandas con retome:** el resultado de cada foto se guarda al terminarla; si se corta, al volver a correr
  sigue donde quedó. `--max-images N` limita cuántas fotos nuevas procesa cada corrida.
- **Prompts configurables** (`--curb-prompts`, `--curb-cut-prompts`) y un modo `--explore` para probar
  variantes de prompt en pocas fotos y compararlas antes de procesar todo.
- Corre en GPU (`--device cuda`) o en CPU (`--device cpu`, mucho más lento: sirve para tandas nocturnas).

Las fotos y las salidas tienen rostros y patentes de terceros: todo queda fuera del repo. Uso: ver
`tools/labeling/README.md`.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
CURB_FOLDER = "curb"
CLASSES = ("curb", "curb_cut")
DEFAULT_PROMPTS = {"curb": ("curb",), "curb_cut": ("dropped curb", "curb ramp")}
HF_MODEL = "facebook/sam3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_THRESHOLD = 0.30  # confianza mínima de una instancia de SAM 3
DEFAULT_MASK_THRESHOLD = 0.50
DEFAULT_SIMPLIFY_PX = 1.5  # tolerancia al simplificar los polígonos
DEFAULT_MIN_AREA = 0.0005  # fracción de la foto: debajo de esto una instancia es ruido
DEDUPE_IOU = 0.70
DEFAULT_SAMPLES = 6
MIN_POLYGON_POINTS = 3
CLASS_COLORS = {"curb": (0, 114, 178), "curb_cut": (230, 159, 0)}  # azul y naranja (RGB)
OVERLAY_ALPHA = 0.45
PROGRESS_EVERY = 5
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class Instance:
    """Una instancia que encontró SAM 3: clase, prompt, confianza y polígonos (coordenadas de la foto)."""

    label: str
    prompt: str
    score: float
    polygons: list[list[float]]
    area: float
    bbox: list[float]


class Backend(Protocol):
    """Lo que necesita el script de un segmentador por texto (SAM 3 en producción, un falso en los tests)."""

    def segment(
        self, image: NDArray[np.uint8], prompts: Sequence[str]
    ) -> dict[str, list[tuple[float, NDArray[np.bool_]]]]:
        """Por cada prompt, las instancias `(confianza, máscara booleana del tamaño de la foto)`."""
        ...


# --- polígonos y COCO (sin modelo) -------------------------------------------------------------------


def mask_to_polygons(
    mask: NDArray[np.bool_], simplify_px: float = DEFAULT_SIMPLIFY_PX, min_area: float = DEFAULT_MIN_AREA
) -> list[list[float]]:
    """Polígonos `[x1, y1, x2, y2, ...]` del contorno exterior de cada parte de la máscara."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum = min_area * mask.size
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum:
            continue
        simplified = cv2.approxPolyDP(contour, simplify_px, True).reshape(-1, 2)
        if len(simplified) >= MIN_POLYGON_POINTS:
            polygons.append([float(value) for value in simplified.ravel()])
    return polygons


def polygons_area(polygons: Sequence[Sequence[float]], shape: tuple[int, int]) -> float:
    canvas = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.array(polygon, np.int32).reshape(-1, 2)], 1)
    return float(canvas.sum())


def polygons_to_mask(polygons: Sequence[Sequence[float]], shape: tuple[int, int]) -> NDArray[np.bool_]:
    canvas = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.array(polygon)).astype(np.int32).reshape(-1, 2)], 1)
    return canvas.astype(bool)


def to_instance(
    label: str, prompt: str, score: float, mask: NDArray[np.bool_], simplify_px: float, min_area: float
) -> Instance | None:
    polygons = mask_to_polygons(mask, simplify_px, min_area)
    if not polygons:
        return None
    ys, xs = np.nonzero(mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    return Instance(label, prompt, round(float(score), 4), polygons, float(mask.sum()), box)


def mask_iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def dedupe(
    instances: Sequence[Instance], shape: tuple[int, int], threshold: float = DEDUPE_IOU
) -> list[Instance]:
    """Quita las instancias de una misma clase que se pisan: dos prompts suelen encontrar lo mismo."""
    kept: list[tuple[Instance, NDArray[np.bool_]]] = []
    for instance in sorted(instances, key=lambda item: -item.score):
        mask = polygons_to_mask(instance.polygons, shape)
        if any(item.label == instance.label and mask_iou(mask, other) >= threshold for item, other in kept):
            continue
        kept.append((instance, mask))
    return [instance for instance, _ in kept]


def build_coco(photos: dict[str, tuple[tuple[int, int], Sequence[Instance]]]) -> dict[str, Any]:
    """COCO con polígonos (`{foto: ((alto, ancho), instancias)}`) en el formato que importa CVAT."""
    images, annotations = [], []
    for name, ((height, width), instances) in sorted(photos.items()):
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        for instance in instances:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASSES.index(instance.label) + 1,
                    "segmentation": instance.polygons,
                    "bbox": instance.bbox,
                    "area": instance.area,
                    "iscrowd": 0,
                    "attributes": {"score": instance.score, "prompt": instance.prompt},
                }
            )
    return {
        "info": {"description": "Preetiquetado con SAM 3 para corregir en CVAT"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(CLASSES)],
    }


def draw_overlay(image: NDArray[np.uint8], instances: Sequence[Instance]) -> NDArray[np.uint8]:
    """Copia de la foto con cada clase pintada (azul: cordón, naranja: rebaje) y su confianza."""
    out = image.copy()
    layer = image.copy()
    for instance in instances:
        for polygon in instance.polygons:
            points = np.round(np.array(polygon)).astype(np.int32).reshape(-1, 2)
            cv2.fillPoly(layer, [points], CLASS_COLORS[instance.label])
            cv2.polylines(out, [points], True, CLASS_COLORS[instance.label], 2)
    out = cv2.addWeighted(layer, OVERLAY_ALPHA, out, 1 - OVERLAY_ALPHA, 0)
    for instance in instances:
        x, y = int(instance.bbox[0]), int(instance.bbox[1]) + 14
        cv2.putText(
            out,
            f"{instance.label} {instance.score:.2f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"{instance.label} {instance.score:.2f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            CLASS_COLORS[instance.label],
            1,
        )
    return out


# --- proceso por foto, con retome ---------------------------------------------------------------------


def config_hash(
    prompts: dict[str, Sequence[str]],
    threshold: float,
    mask_threshold: float,
    simplify_px: float,
    min_area: float,
) -> str:
    """Identifica la configuración: si cambia, los resultados guardados dejan de valer."""
    payload = json.dumps(
        {
            "prompts": {k: list(v) for k, v in sorted(prompts.items())},
            "umbral": threshold,
            "umbral_mascara": mask_threshold,
            "simplificar": simplify_px,
            "area_minima": min_area,
            "modelo": HF_MODEL,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def read_image(path: Path) -> NDArray[np.uint8] | None:
    """Foto RGB (con la orientación EXIF aplicada), o `None` si no se puede leer."""
    data = np.fromfile(str(path), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # type: ignore[no-any-return]


def process_photo(
    image: NDArray[np.uint8],
    backend: Backend,
    prompts: dict[str, Sequence[str]],
    simplify_px: float,
    min_area: float,
) -> list[Instance]:
    """Instancias de las dos clases de una foto, sin duplicados."""
    every_prompt = [prompt for label in CLASSES for prompt in prompts[label]]
    found = backend.segment(image, every_prompt)
    instances: list[Instance] = []
    for label in CLASSES:
        for prompt in prompts[label]:
            for score, mask in found.get(prompt, []):
                instance = to_instance(label, prompt, score, mask, simplify_px, min_area)
                if instance is not None:
                    instances.append(instance)
    return dedupe(instances, image.shape[:2])


def load_cached(path: Path) -> tuple[tuple[int, int], list[Instance]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data["alto"], data["ancho"]), [Instance(**item) for item in data["instancias"]]


def save_cached(path: Path, shape: tuple[int, int], instances: Sequence[Instance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"alto": shape[0], "ancho": shape[1], "instancias": [asdict(i) for i in instances]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def list_photos(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def run_batch(
    photos: Sequence[Path],
    out_dir: Path,
    backend: Backend,
    prompts: dict[str, Sequence[str]],
    settings: tuple[float, float, float, float],
    max_images: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Procesa las fotos que faltan (hasta `max_images`) y guarda cada resultado al terminarla."""
    threshold, mask_threshold, simplify_px, min_area = settings
    cache = out_dir / "cache" / config_hash(prompts, threshold, mask_threshold, simplify_px, min_area)
    pending = [p for p in photos if not (cache / f"{p.name}.json").is_file()]
    todo = pending if max_images is None else pending[:max_images]
    stats: dict[str, Any] = {
        "fotos": len(photos),
        "ya_hechas": len(photos) - len(pending),
        "procesadas": 0,
        "ilegibles": 0,
    }
    started = time.perf_counter()
    for index, photo in enumerate(todo, start=1):
        image = read_image(photo)
        if image is None:
            stats["ilegibles"] += 1
            continue
        instances = process_photo(image, backend, prompts, simplify_px, min_area)
        save_cached(cache / f"{photo.name}.json", image.shape[:2], instances)
        stats["procesadas"] += 1
        if index % PROGRESS_EVERY == 0 or index == len(todo):
            log(f"  {index}/{len(todo)} fotos ({(time.perf_counter() - started) / index:.1f} s por foto)")
    stats["faltan"] = len(pending) - stats["procesadas"] - stats["ilegibles"]
    stats["segundos"] = round(time.perf_counter() - started)
    return stats


def collect(
    photos: Sequence[Path], out_dir: Path, hash_: str
) -> dict[str, tuple[tuple[int, int], list[Instance]]]:
    """Resultados ya guardados de las fotos."""
    result = {}
    for photo in photos:
        cached = load_cached(out_dir / "cache" / hash_ / f"{photo.name}.json")
        if cached is not None:
            result[photo.name] = cached
    return result


def write_outputs(results: dict[str, tuple[tuple[int, int], list[Instance]]], out_dir: Path) -> Path:
    """`annotations/instances_default.json` y el zip `prelabel_coco.zip` para importar en CVAT."""
    coco = build_coco(results)
    annotations = out_dir / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / "instances_default.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    archive = out_dir / "prelabel_coco.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.write(path, "annotations/instances_default.json")
    return archive


def write_samples(
    photos: Sequence[Path],
    results: dict[str, tuple[tuple[int, int], list[Instance]]],
    out_dir: Path,
    count: int,
) -> int:
    """Imágenes de muestra con las máscaras superpuestas, para revisar a ojo."""
    folder = out_dir / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    written = 0
    for photo in photos:
        if written >= count:
            break
        if photo.name not in results:
            continue
        image = read_image(photo)
        if image is None:
            continue
        overlay = draw_overlay(image, results[photo.name][1])
        cv2.imwrite(str(folder / f"{photo.stem}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written += 1
    return written


def summarize(results: dict[str, tuple[tuple[int, int], list[Instance]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(i.label for _, items in results.values() for i in items)
    with_class = {
        name: sum(any(i.label == name for i in items) for _, items in results.values()) for name in CLASSES
    }
    return {
        "fotos": len(results),
        "instancias": {name: counts.get(name, 0) for name in CLASSES},
        "fotos_con_la_clase": with_class,
        "fotos_sin_nada": sum(not items for _, items in results.values()),
    }


def explore(
    photos: Sequence[Path],
    out_dir: Path,
    backend: Backend,
    prompts: Sequence[str],
    settings: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Prueba cada prompt por separado en pocas fotos: cuántas instancias y con qué confianza."""
    _, _, simplify_px, min_area = settings
    report: dict[str, Any] = {}
    folder = out_dir / "explore"
    folder.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        scores: list[float] = []
        with_hits = 0
        for photo in photos:
            image = read_image(photo)
            if image is None:
                continue
            found = backend.segment(image, [prompt]).get(prompt, [])
            instances = [
                i for s, m in found if (i := to_instance("curb", prompt, s, m, simplify_px, min_area))
            ]
            scores.extend(i.score for i in instances)
            with_hits += bool(instances)
            safe = "".join(c if c.isalnum() else "_" for c in prompt)
            overlay = draw_overlay(image, instances)
            cv2.imwrite(str(folder / f"{safe}__{photo.stem}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        report[prompt] = {
            "fotos_con_resultado": with_hits,
            "instancias": len(scores),
            "confianza_media": round(float(np.mean(scores)), 3) if scores else None,
        }
    return report


# --- SAM 3 -------------------------------------------------------------------------------------------


class Sam3Backend:
    """SAM 3 de `transformers`: los *embeddings* de la foto se calculan una vez y se reusan en cada prompt."""

    def __init__(self, device: str, threshold: float, mask_threshold: float) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self._device = device
        self._threshold = threshold
        self._mask_threshold = mask_threshold
        self._model = Sam3Model.from_pretrained(HF_MODEL).to(device).eval()
        self._processor = Sam3Processor.from_pretrained(HF_MODEL)

    def peak_memory_mb(self) -> int | None:
        """Pico de memoria de la GPU en MB, o `None` si corre en CPU."""
        if self._device != "cuda":
            return None
        return int(self._torch.cuda.max_memory_allocated() // BYTES_PER_MB)

    def segment(
        self, image: NDArray[np.uint8], prompts: Sequence[str]
    ) -> dict[str, list[tuple[float, NDArray[np.bool_]]]]:
        from PIL import Image

        torch = self._torch
        height, width = image.shape[:2]
        with torch.no_grad():
            pixel_values = self._processor(
                images=Image.fromarray(image), return_tensors="pt"
            ).pixel_values.to(self._device)
            vision = self._model.get_vision_features(pixel_values=pixel_values)
            found: dict[str, list[tuple[float, NDArray[np.bool_]]]] = {}
            for prompt in prompts:
                text = self._processor(text=prompt, return_tensors="pt").to(self._device)
                outputs = self._model(
                    vision_embeds=vision, input_ids=text.input_ids, attention_mask=text.attention_mask
                )
                result = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self._threshold,
                    mask_threshold=self._mask_threshold,
                    target_sizes=[(height, width)],
                )[0]
                found[prompt] = [
                    (float(score), mask.cpu().numpy().astype(bool))
                    for score, mask in zip(result["scores"], result["masks"], strict=True)
                ]
        return found


# --- línea de comandos --------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--images", type=Path, help="carpeta con las fotos (por defecto <curb>/raw)")
    parser.add_argument("--out-dir", type=Path, help="salida (por defecto <curb>/prelabel)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--curb-prompts", nargs="+", default=list(DEFAULT_PROMPTS["curb"]))
    parser.add_argument("--curb-cut-prompts", nargs="+", default=list(DEFAULT_PROMPTS["curb_cut"]))
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="confianza mínima de una instancia"
    )
    parser.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument(
        "--simplify", type=float, default=DEFAULT_SIMPLIFY_PX, help="tolerancia de los polígonos, en px"
    )
    parser.add_argument(
        "--min-area", type=float, default=DEFAULT_MIN_AREA, help="área mínima, como fracción de la foto"
    )
    parser.add_argument("--max-images", type=int, help="fotos nuevas por corrida (para trabajar por tandas)")
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES, help="imágenes de muestra con las máscaras"
    )
    parser.add_argument(
        "--explore", nargs="+", metavar="PROMPT", help="probar prompts sueltos en pocas fotos"
    )
    parser.add_argument("--explore-photos", type=int, default=5, help="fotos para --explore")
    args = parser.parse_args(argv)
    if args.data_dir is None and (args.images is None or args.out_dir is None):
        parser.error("indicá --data-dir (EVAL_DATA_DIR) o --images y --out-dir")
    curb = Path(args.data_dir) / CURB_FOLDER if args.data_dir else None
    args.images = args.images or (curb / "raw" if curb else None)
    args.out_dir = args.out_dir or (curb / "prelabel" if curb else None)
    for flag, value in (("--threshold", args.threshold), ("--mask-threshold", args.mask_threshold)):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{flag} debe estar entre 0 y 1")
    return args


def make_backend(args: argparse.Namespace) -> Backend:
    return Sam3Backend(args.device, args.threshold, args.mask_threshold)


def main(
    argv: list[str] | None = None, backend_factory: Callable[[argparse.Namespace], Backend] = make_backend
) -> int:
    args = parse_args(argv)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.images, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    if not Path(args.images).is_dir():
        print(f"No existe la carpeta de fotos {args.images}.", file=sys.stderr)
        return 1
    photos = list_photos(Path(args.images))
    if not photos:
        print("No hay fotos en la carpeta.", file=sys.stderr)
        return 1
    settings = (args.threshold, args.mask_threshold, args.simplify, args.min_area)
    prompts = {"curb": args.curb_prompts, "curb_cut": args.curb_cut_prompts}
    try:
        backend = backend_factory(args)
    except Exception as exc:  # noqa: BLE001 - se informa el tipo y el motivo, sin datos
        print(
            f"No se pudo cargar SAM 3 ({type(exc).__name__}): {str(exc).splitlines()[0] if str(exc) else ''}",
            file=sys.stderr,
        )
        return 1
    out_dir = Path(args.out_dir)
    if args.explore:
        report = explore(photos[: args.explore_photos], out_dir, backend, args.explore, settings)
        for prompt, row in report.items():
            print(f"  {prompt!r}: {row}")
        print(f"Superposiciones en {out_dir / 'explore'}")
        return 0
    stats = run_batch(photos, out_dir, backend, prompts, settings, args.max_images)
    peak = getattr(backend, "peak_memory_mb", lambda: None)()
    if peak is not None:
        stats["pico_memoria_gpu_mb"] = peak
    results = collect(photos, out_dir, config_hash(prompts, *settings))
    archive = write_outputs(results, out_dir)
    written = write_samples(photos, results, out_dir, args.samples)
    print(f"Fotos: {stats}")
    print(f"Resultado: {summarize(results)}")
    print(f"COCO para CVAT: {archive} · muestras: {written} en {out_dir / 'samples'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
