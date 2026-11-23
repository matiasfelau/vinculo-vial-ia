FROM python:3.12.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12.12-slim-bookworm

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_DISABLE_TELEMETRY=1 \
    MODELS_DIR=/models \
    MODEL_DETECTOR=rfdetr_openimages_v2 \
    MODEL_SEGMENTER=rfdetr_seg_coco \
    MODEL_EMBEDDER=dinov2_vits14_reid_v2 \
    MODEL_OCR=parseq

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

# Pesos de la cascada por defecto (ENV de arriba): detector, segmentador, re-ID y OCR. Un COPY por archivo:
# si falta uno el build falla, y cada uno es su propia capa. Los experimentales (daños, rebajes, rostros) y
# las versiones no activas no entran. No están en git: salen del release models-vN (ver el README). El
# servicio verifica su sha256 contra el manifest al arrancar. Un montaje en /models (el compose local)
# tapa estas capas y tiene prioridad.
# El directorio se crea aparte: un COPY --chmod lo crearía 0444 y el usuario no root no podría entrar.
RUN mkdir -m 0755 /models
COPY --chmod=0444 models/weights/rfdetr_openimages_v2.onnx /models/rfdetr_openimages_v2.onnx
COPY --chmod=0444 models/weights/rfdetr_seg_coco.onnx /models/rfdetr_seg_coco.onnx
COPY --chmod=0444 models/weights/dinov2_vits14_reid_v2.onnx /models/dinov2_vits14_reid_v2.onnx
COPY --chmod=0444 models/weights/parseq.onnx /models/parseq.onnx

WORKDIR /app
COPY models/manifest.json ./models/manifest.json

USER app

EXPOSE 8080

CMD ["python", "-m", "vvi_ia.main"]
