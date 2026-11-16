"""Etiquetas del set de evaluación del OCR en Excel: arma `etiquetas.xlsx`, lo lee y lo convierte a CSV.

`ocr_set.py` deja en su carpeta un `etiquetas.csv` con las filas por id. Para etiquetar cómodo este módulo
arma `etiquetas.xlsx` (una fila por id con la miniatura del recorte y de la foto entera, desplegables y
validaciones) y después lo lee de vuelta: `ocr.py --labeled-set` y las guardas de `ocr_set.py` usan el xlsx
si existe, sin pasos manuales. El csv es la fuente de `archivo`, `recorte` y `revisar_moto`; el xlsx, la de
`texto`, `formato`, `luz` y `estado`. Nada de esto va al repo: tiene patentes de terceros.
Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_NAME = "etiquetas.csv"
XLSX_NAME = "etiquetas.xlsx"
BOXES_NAME = "cajas.csv"
SHEET = "etiquetas"
GUIDE_SHEET = "instrucciones"

FORMATS = ("anterior", "mercosur", "moto", "otro")
LIGHTS = ("dia", "noche")
STATES = ("legible", "ilegible", "fuera_de_alcance")
STATE_LEGIBLE = "legible"
CAR_FORMATS = ("anterior", "mercosur")  # los que lee el servicio: una sola línea
REVIEW_VALUE = "sí"
LABEL_FIELDS = ("texto", "formato", "luz", "estado")
# Validación de Excel de la celda C2 (se extiende a la columna): vacío, o 6 a 7 caracteres, solo letras
# mayúsculas y dígitos (FIND distingue mayúsculas; el texto se rellena con ceros para mirar 7 posiciones).
TEXT_FORMULA = (
    "OR(LEN(C2)=0,AND(LEN(C2)>=6,LEN(C2)<=7,"
    'SUMPRODUCT(--ISNUMBER(FIND(MID(C2&"0000000",{1,2,3,4,5,6,7},1),'
    '"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")))=7))'
)
TEXT_PATTERN = re.compile(r"^[A-Z0-9]{6,7}$")
TEXT_ERROR = (
    "El texto tiene que quedar vacío o ser la patente de 6 o 7 caracteres: letras MAYÚSCULAS y dígitos, sin "
    "espacios ni guiones (por ejemplo AB123CD o ABC123). Si no se lee, dejalo vacío y poné estado = ilegible."
)

# Columnas del xlsx, en este orden: A id, B miniatura, C texto, D formato, E luz, F estado, G revisar_moto.
COLUMNS = ("id", "miniatura", "texto", "formato", "luz", "estado", "revisar_moto")
WIDTHS = {"A": 6, "B": 60, "C": 14, "D": 12, "E": 9, "F": 16, "G": 13}
ROW_HEIGHT_PT = 85
HEADER_HEIGHT_PT = 24
EMU_PER_PIXEL = 9525
THUMB_W, THUMB_H = 420, 110  # píxeles de la miniatura de una fila
CROP_W, PHOTO_W, THUMB_GAP = 300, 110, 4
THUMB_QUALITY = 86
ANCHOR_PADDING = 2  # píxeles entre la miniatura y el borde de la celda

COLOR_HEADER = "1F3864"
COLOR_LOCKED = "EDEDED"
COLOR_LOCKED_TEXT = "595959"
COLOR_MISSING_TEXT = "FFC7CE"  # estado legible y texto vacío
COLOR_REVIEW_MOTO = "FFE699"  # revisar_moto = sí


class LabelsError(ValueError):
    """Las etiquetas no se pueden leer, no son válidas o no se pueden pisar."""


# --- lectura -------------------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _text(value: Any) -> str:
    """Una celda como texto: vacía si no hay nada; sin el `.0` que Excel agrega a un número entero."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_xlsx_rows(path: Path) -> dict[str, dict[str, str]]:
    """`id → {texto, formato, luz, estado}` de la hoja de etiquetas. Los encabezados se buscan por nombre."""
    from openpyxl import load_workbook

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # archivo abierto en Excel, dañado o que no es un xlsx
        raise LabelsError(f"no se pudo abrir {path.name}: {exc}") from exc
    try:
        sheet = workbook[SHEET] if SHEET in workbook.sheetnames else workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = [_text(cell).lower() for cell in next(rows, ())]
        missing = [name for name in ("id", *LABEL_FIELDS) if name not in header]
        if missing:
            raise LabelsError(f"{path.name} no tiene las columnas {', '.join(missing)}")
        index = {name: header.index(name) for name in ("id", *LABEL_FIELDS)}
        found: dict[str, dict[str, str]] = {}
        for row in rows:
            ident = _text(row[index["id"]]) if len(row) > index["id"] else ""
            if not ident:
                continue
            if ident in found:
                raise LabelsError(f"{path.name}: el id {ident} está repetido")
            found[ident] = {
                name: _text(row[index[name]]) if len(row) > index[name] else "" for name in LABEL_FIELDS
            }
        return found
    finally:
        workbook.close()


def read_label_rows(set_dir: Path) -> list[dict[str, str]]:
    """Las filas del set con las etiquetas más recientes: las del xlsx si existe, si no las del csv."""
    rows = read_csv_rows(set_dir / CSV_NAME)
    xlsx = set_dir / XLSX_NAME
    if not xlsx.is_file():
        return rows
    labels = read_xlsx_rows(xlsx)
    ids = {row["id"] for row in rows}
    if ids != set(labels):
        raise LabelsError(
            f"{XLSX_NAME} no tiene los mismos ids que {CSV_NAME} "
            f"(sobran {len(set(labels) - ids)}, faltan {len(ids - set(labels))}): regeneralo con `xlsx`"
        )
    return [{**row, **labels[row["id"]]} for row in rows]


def has_labels(rows: Iterable[Mapping[str, str]]) -> bool:
    """`True` si alguna fila tiene texto, formato o estado (la luz viene precargada y no cuenta)."""
    return any(row.get("texto") or row.get("formato") or row.get("estado") for row in rows)


def labels_loaded(set_dir: Path) -> bool:
    """`True` si el set ya tiene etiquetas cargadas, en el csv o en el xlsx."""
    return has_labels(read_label_rows(set_dir))


def row_problems(row: Mapping[str, str]) -> list[str]:
    """Qué está mal en una fila (lista vacía si está bien)."""
    problems = []
    if row.get("formato") and row["formato"] not in FORMATS:
        problems.append(f"formato '{row['formato']}'")
    if row.get("luz") and row["luz"] not in LIGHTS:
        problems.append(f"luz '{row['luz']}'")
    if row.get("estado") and row["estado"] not in STATES:
        problems.append(f"estado '{row['estado']}'")
    if row.get("texto") and not TEXT_PATTERN.match(row["texto"]):
        problems.append(f"texto '{row['texto']}'")
    return problems


def check_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Avance y problemas de las etiquetas (solo cifras e ids: nunca el texto de las patentes)."""
    invalid = {row["id"]: row_problems(row) for row in rows if row_problems(row)}
    return {
        "filas": len(rows),
        "con_estado": sum(bool(row["estado"]) for row in rows),
        "legibles": sum(row["estado"] == STATE_LEGIBLE for row in rows),
        "ilegibles": sum(row["estado"] == "ilegible" for row in rows),
        "fuera_de_alcance": sum(row["estado"] == "fuera_de_alcance" for row in rows),
        "sin_estado": sum(not row["estado"] for row in rows),
        "legibles_sin_texto": sum(row["estado"] == STATE_LEGIBLE and not row["texto"] for row in rows),
        "con_texto_sin_estado_legible": sum(
            bool(row["texto"]) and row["estado"] != STATE_LEGIBLE for row in rows
        ),
        "filas_con_problemas": len(invalid),
        "ids_con_problemas": sorted(invalid),
    }


def to_csv(set_dir: Path) -> Path | None:
    """Escribe en `etiquetas.csv` lo que hay en el xlsx; antes copia el csv a `etiquetas-previo-N.csv`.

    Devuelve la copia, o `None` si el csv ya estaba al día.
    """
    path = set_dir / CSV_NAME
    current = read_csv_rows(path)
    merged = read_label_rows(set_dir)
    if merged == current:
        return None
    number = 1
    while (set_dir / f"etiquetas-previo-{number}.csv").exists():
        number += 1
    backup = set_dir / f"etiquetas-previo-{number}.csv"
    shutil.copyfile(path, backup)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(current[0]))
        writer.writeheader()
        writer.writerows(merged)
    return backup


# --- repetidos y reposición ----------------------------------------------------------------------------


def is_legible_car(row: Mapping[str, str]) -> bool:
    """Una patente legible de auto (formato anterior o Mercosur) con su texto cargado."""
    return row.get("estado") == STATE_LEGIBLE and row.get("formato") in CAR_FORMATS and bool(row.get("texto"))


def crop_area(set_dir: Path, row: Mapping[str, str]) -> int:
    """Píxeles del recorte de una fila (0 si no se puede leer): a más resolución, mejor para conservar."""
    crop = _decode(set_dir / row["recorte"])
    return 0 if crop is None else int(crop.shape[0] * crop.shape[1])


def repeated_texts(
    rows: Sequence[Mapping[str, str]], area: Callable[[Mapping[str, str]], int]
) -> list[tuple[str, str]]:
    """Una sola lectura por patente repetida: pares `(id que sale, id que se conserva)`.

    Agrupa por el texto de las filas legibles. De cada grupo se conserva la de noche (la luz más escasa),
    después la de recorte más grande y, si empatan, la de menor id.
    """
    groups: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        if row.get("estado") == STATE_LEGIBLE and row.get("texto"):
            groups.setdefault(row["texto"], []).append(row)
    pairs: list[tuple[str, str]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = min(group, key=lambda row: (row.get("luz") != "noche", -area(row), row["id"]))
        pairs.extend((row["id"], keep["id"]) for row in group if row is not keep)
    return sorted(pairs)


def near_texts(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Pares de ids de patentes legibles de auto que difieren en un solo carácter (¿la misma mal escrita?).

    Solo para revisar a ojo: no se saca nada. Nunca devuelve los textos.
    """
    from rapidfuzz.distance import Levenshtein

    cars = [row for row in rows if is_legible_car(row)]
    return [
        (a["id"], b["id"])
        for index, a in enumerate(cars)
        for b in cars[index + 1 :]
        if a["texto"] != b["texto"] and Levenshtein.distance(a["texto"], b["texto"]) == 1
    ]


# --- armado del xlsx -------------------------------------------------------------------------------


def _decode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def thumbnail(crop: np.ndarray, photo: np.ndarray | None, box: Sequence[float] | None) -> np.ndarray:
    """El recorte grande y, a la derecha, la foto entera con la caja marcada, en una sola imagen."""
    canvas = np.full((THUMB_H, THUMB_W, 3), 255, np.uint8)
    fitted = _fit(crop, CROP_W if photo is not None else THUMB_W, THUMB_H)
    canvas[: fitted.shape[0], : fitted.shape[1]] = fitted
    if photo is not None:
        scene = photo.copy()
        if box is not None:
            x, y, w, h = box
            line = max(2, max(scene.shape[:2]) // 150)
            cv2.rectangle(scene, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), line)
        small = _fit(scene, PHOTO_W, THUMB_H)
        left = CROP_W + THUMB_GAP
        canvas[: small.shape[0], left : left + small.shape[1]] = small
    return canvas


def _jpeg(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    if not ok:
        raise OSError("no se pudo codificar la miniatura")
    return bytes(data.tobytes())


def _box(text: str) -> list[float] | None:
    try:
        x, y, w, h = (float(part) for part in text.split())
    except ValueError:
        return None
    return [x, y, w, h]


def _column_pixels(width_chars: float) -> int:
    return int(width_chars * 7 + 5)


def build_xlsx(set_dir: Path, photos_dir: Path | None = None, out: Path | None = None) -> Path:
    """Arma `etiquetas.xlsx` desde `etiquetas.csv`. Si ya existe con etiquetas cargadas, no lo pisa.

    Las celdas editables son `texto`, `formato`, `luz` y `estado`; `id`, `miniatura` y `revisar_moto` quedan
    en gris y bloqueadas (la hoja está protegida sin contraseña). Las etiquetas que `etiquetas.csv` ya tenga
    se precargan.
    """
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image
    from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, TwoCellAnchor
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation

    out = out or set_dir / XLSX_NAME
    rows = read_csv_rows(set_dir / CSV_NAME)
    if out.exists() and has_labels(read_xlsx_rows(out).values()):
        raise LabelsError(f"{out.name} ya tiene etiquetas cargadas: no se pisa")
    boxes = (
        {row["id"]: row for row in read_csv_rows(set_dir / BOXES_NAME)}
        if (set_dir / BOXES_NAME).is_file()
        else {}
    )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET
    last = len(rows) + 1
    sheet.append(list(COLUMNS))
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=COLOR_HEADER, end_color=COLOR_HEADER)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = HEADER_HEIGHT_PT
    for letter, width in WIDTHS.items():
        sheet.column_dimensions[letter].width = width

    locked_fill = PatternFill("solid", start_color=COLOR_LOCKED, end_color=COLOR_LOCKED)
    locked_font = Font(color=COLOR_LOCKED_TEXT)
    editable = Protection(locked=False)
    centered = Alignment(horizontal="center", vertical="center")
    col_px, row_px = _column_pixels(WIDTHS["B"]), int(ROW_HEIGHT_PT * 96 / 72)
    for number, row in enumerate(rows, start=2):
        ident = row["id"]
        sheet.row_dimensions[number].height = ROW_HEIGHT_PT
        values = {
            "id": ident,
            "texto": row.get("texto", ""),
            "formato": row.get("formato", ""),
            "luz": row.get("luz", ""),
            "estado": row.get("estado", ""),
            "revisar_moto": row.get("revisar_moto", ""),
        }
        for position, name in enumerate(COLUMNS, start=1):
            cell = sheet.cell(row=number, column=position, value=values.get(name) or None)
            cell.alignment = centered
            if name in ("id", "miniatura", "revisar_moto"):
                cell.fill, cell.font = locked_fill, locked_font
            else:
                cell.protection = editable
            if name in ("id", "texto"):
                cell.number_format = "@"
            if name == "texto":
                cell.font = Font(name="Consolas", size=14, bold=True)
        crop = _decode(set_dir / row["recorte"])
        if crop is None:
            continue
        photo = None
        box = _box(boxes[ident]["caja"]) if ident in boxes else None
        if photos_dir is not None:
            photo = _decode(photos_dir / row["archivo"])
        image = Image(BytesIO(_jpeg(thumbnail(crop, photo, box))))
        image.width, image.height = THUMB_W, THUMB_H
        pad = ANCHOR_PADDING * EMU_PER_PIXEL
        image.anchor = TwoCellAnchor(
            editAs="twoCell",  # se mueve, se achica y se oculta con la fila al filtrar
            _from=AnchorMarker(col=1, row=number - 1, colOff=pad, rowOff=pad),
            to=AnchorMarker(
                col=1,
                row=number - 1,
                colOff=(col_px - ANCHOR_PADDING) * EMU_PER_PIXEL,
                rowOff=(row_px - ANCHOR_PADDING) * EMU_PER_PIXEL,
            ),
        )
        sheet.add_image(image)

    for letter, name, options in (
        ("D", "formato", FORMATS),
        ("E", "luz", LIGHTS),
        ("F", "estado", STATES),
    ):
        validation = DataValidation(
            type="list",
            formula1='"' + ",".join(options) + '"',
            allow_blank=True,
            showErrorMessage=True,
            errorStyle="stop",
            errorTitle=f"{name} no válido",
            error=f"Elegí un valor de la lista: {', '.join(options)}.",
        )
        validation.add(f"{letter}2:{letter}{last}")
        sheet.add_data_validation(validation)
    text_validation = DataValidation(
        type="custom",
        formula1=TEXT_FORMULA,
        allow_blank=True,
        showErrorMessage=True,
        errorStyle="stop",
        errorTitle="Texto no válido",
        error=TEXT_ERROR,
        showInputMessage=True,
        promptTitle="texto",
        prompt="Vacío, o 6 a 7 caracteres en MAYÚSCULAS, sin espacios ni guiones.",
    )
    text_validation.add(f"C2:C{last}")
    sheet.add_data_validation(text_validation)

    area = f"A2:G{last}"
    sheet.conditional_formatting.add(
        area,
        FormulaRule(
            formula=['AND($F2="legible",LEN($C2)=0)'],
            fill=PatternFill("solid", start_color=COLOR_MISSING_TEXT, end_color=COLOR_MISSING_TEXT),
            stopIfTrue=True,
        ),
    )
    sheet.conditional_formatting.add(
        area,
        FormulaRule(
            formula=[f'$G2="{REVIEW_VALUE}"'],
            fill=PatternFill("solid", start_color=COLOR_REVIEW_MOTO, end_color=COLOR_REVIEW_MOTO),
        ),
    )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:G{last}"
    protection = sheet.protection
    protection.sheet = True  # sin contraseña: Revisar > Desproteger hoja
    protection.objects = True  # las miniaturas no se mueven ni se borran
    protection.scenarios = True
    for allowed in (
        "autoFilter",
        "sort",
        "formatColumns",
        "formatRows",
        "selectLockedCells",
        "selectUnlockedCells",
    ):
        setattr(protection, allowed, False)  # en OOXML, False = el usuario puede hacerlo

    guide = workbook.create_sheet(GUIDE_SHEET)
    for line in (
        "Cómo etiquetar",
        "Completá texto, formato, luz y estado de cada fila. id, miniatura y revisar_moto son solo lectura.",
        "texto: la patente en MAYÚSCULAS, 6 o 7 caracteres, sin espacios ni guiones. Vacío si no se lee.",
        "formato: " + ", ".join(FORMATS) + ".",
        "luz: " + ", ".join(LIGHTS) + " (viene precargada con una estimación: confirmala o corregila).",
        "estado: " + ", ".join(STATES) + ".",
        "Amarillo: revisar_moto = sí; confirmá si es una moto (formato = moto).",
        "Rojo: estado legible con el texto vacío.",
        "Hoja protegida sin contraseña (Revisar > Desproteger hoja). No cambies el orden de las columnas.",
    ):
        guide.append([line])
    guide["A1"].font = Font(bold=True, size=13)
    guide.column_dimensions["A"].width = 120
    workbook.save(out)
    return out


# --- línea de comandos -------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("xlsx", "arma etiquetas.xlsx desde etiquetas.csv (no pisa uno con etiquetas)"),
        ("to-csv", "escribe en etiquetas.csv lo que hay en etiquetas.xlsx (con copia del csv)"),
        ("check", "muestra el avance y los problemas de las etiquetas (solo cifras)"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--set", type=Path, dest="set_dir", help="carpeta del set (ocr-ar-set)")
        if name == "xlsx":
            command.add_argument(
                "--photos", type=Path, help="export COCO con las fotos enteras (por defecto, ../ocr-ar)"
            )
    args = parser.parse_args(argv)
    if args.set_dir is None:
        base = os.environ.get("EVAL_DATA_DIR")
        if base is None:
            parser.error("indicá --set o definí EVAL_DATA_DIR")
        args.set_dir = Path(base) / "ocr-ar-set"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [args.set_dir] + ([args.photos] if getattr(args, "photos", None) else [])
    if any(path.resolve().is_relative_to(REPO_ROOT) for path in paths):
        print("El set y las fotos no pueden estar dentro del repositorio: tienen patentes de terceros.")
        return 1
    try:
        if args.command == "xlsx":
            photos = args.photos or args.set_dir.parent / "ocr-ar"
            out = build_xlsx(args.set_dir, photos if photos.is_dir() else None)
            print(f"{out}: {len(read_csv_rows(args.set_dir / CSV_NAME))} filas")
        elif args.command == "to-csv":
            backup = to_csv(args.set_dir)
            print(
                "etiquetas.csv ya estaba al día"
                if backup is None
                else f"etiquetas.csv actualizado; copia en {backup.name}"
            )
        else:
            report = check_rows(read_label_rows(args.set_dir))
            for key, value in report.items():
                print(f"{key}: {value}")
    except (LabelsError, OSError, ImportError) as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
