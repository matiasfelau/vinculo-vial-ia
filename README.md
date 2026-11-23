# vinculo-vial-ia

Perito de evidencia fotográfica de Vínculo Vial: valida cada alerta con una cascada de modelos ONNX en
CPU, emite el veredicto (GREEN o RED con códigos de motivo) y sube las fotos redactadas a S3. Es
stateless: consume de RabbitMQ y no accede a internet en producción.

## CI

`.github/workflows/ci.yml` corre en todo `pull_request` hacia `dev` o `main` (y en `push` a `dev`).
Tiene dos jobs:

| Job | Qué hace | ¿Corta? |
|---|---|---|
| `test` | ruff, mypy (`domain` y `service`), pruebas con cobertura contra un RabbitMQ real y `diff-cover` | Sí |
| `security` | Trivy `fs` (HIGH y CRITICAL con corrección disponible, secretos y configuración) y Semgrep | Trivy sí; Semgrep no (informativo, pasa a gate en v0.14) |

Las acciones de terceros están fijadas por SHA de commit (la versión va en un comentario) y el
workflow declara `permissions: contents: read`. Dependabot actualiza semanalmente `pip`,
`github-actions` y `docker`.

### Cobertura

- **Qué se mide:** `src/vvi_ia`, que es lo que se despliega. `tools/` (evaluación, exportación,
  etiquetado y entrenamiento) corre fuera de Docker, no entra en la imagen y queda **fuera del
  reporte**. Se configura en `[tool.coverage.run]` de `pyproject.toml`. Es la única exclusión.
- **Gate sobre el código nuevo:** `diff-cover` exige 85 % o más de las líneas que el PR agrega o
  cambia respecto de la rama base del PR.
- **Gate sobre el total:** el job falla si la cobertura de `src/vvi_ia` baja de 85 %
  (`--cov-fail-under=85`). Hoy mide 96,5 % de las líneas con las pruebas de integración contra
  RabbitMQ y 95 % sin ellas. Se publica además en el resumen del job y como artefacto `cobertura`
  (`coverage.xml`).

### Seguridad

- **Trivy** lee `uv.lock`, el `Dockerfile` y busca secretos. Las excepciones viven solo en
  `.trivyignore`, cada una con su motivo y fecha de revisión.
- Hoy hay cuatro excepciones (`pymongo` y `strawberry-graphql`), todas del extra `tools`: las fija
  `fiftyone` y no se pueden corregir sin romper las herramientas. La imagen se construye con
  `uv sync --no-dev` y sin extras, y ninguno de esos paquetes está en su cierre de dependencias.
- Para que `dev` no tenga hallazgos, se subieron `brotli`, `brotlicffi`, `certifi`, `dnspython`,
  `filelock`, `nltk`, `soupsieve` y `pillow` (de 12 a 13, ampliando el límite `<13` a `<14`). Todos
  son de `dev` o `tools`; el cierre de runtime no cambió.

### Correr en local

```bash
uv sync --extra dev

# Lint, tipos y pruebas con cobertura
uv run ruff check .
uv run mypy src/vvi_ia/domain src/vvi_ia/service
uv run pytest --cov --cov-report=xml --cov-report=term-missing:skip-covered

# Gate sobre el código nuevo (con los cambios commiteados y origin/dev actualizado)
pip install diff-cover==11.0.2
diff-cover coverage.xml --compare-branch=origin/dev --fail-under=85

# Seguridad
docker run --rm -v "$PWD:/src" aquasec/trivy:latest fs --scanners vuln,secret,misconfig \
  --severity HIGH,CRITICAL --ignore-unfixed --ignorefile /src/.trivyignore /src
```

Las pruebas que necesitan RabbitMQ se omiten si falta `RABBITMQ_TEST_URL`; el CI la define contra un
servicio real.

## Release

Publicar una versión es empujar un tag `vX.Y.Z`. `.github/workflows/publish.yml` lo dispara y tiene dos
jobs:

| Job | Qué hace |
|---|---|
| `test` | Lo mismo que el job `test` de `ci.yml`: ruff, mypy (`domain` y `service`) y pruebas con cobertura (gate de 85 %), incluida la integración contra un RabbitMQ real |
| `publish` | Solo si `test` pasa: baja y verifica los pesos, construye y sube la imagen |

```bash
git switch main && git pull
git tag v1.2.3
git push origin v1.2.3
```

El tag tiene que ser `vX.Y.Z` (el workflow rechaza cualquier otro). Un tag `models-vN` no dispara nada.

### Qué imagen sale

`felaufernandezbottaro/vinculo-vial-ia` en Docker Hub, con dos tags: la versión sin la `v` (`1.2.3`) y
`latest`. Secrets del repo (Settings → Secrets and variables → Actions): `DOCKERHUB_USERNAME` y
`DOCKERHUB_TOKEN` (un access token de Docker Hub con permiso de escritura, no la contraseña).

La imagen lleva los pesos en `/models` y corre sin acceso a internet: no descarga nada al arrancar. El
`Dockerfile` fija `MODELS_DIR=/models` y los `MODEL_*` de la cascada por defecto; cualquier variable de
entorno los pisa, y un volumen montado en `/models` tapa los pesos de la imagen (es lo que hace
`docker-compose.yml`). El resto de la configuración (`RABBITMQ_URL`, bucket, clave HPKE, etc.) se pasa
en runtime.

| Puerto | Modelo | Archivo |
|---|---|---|
| `detector` | `rfdetr_openimages_v2` | `rfdetr_openimages_v2.onnx` |
| `segmenter` | `rfdetr_seg_coco` | `rfdetr_seg_coco.onnx` |
| `embedder` | `dinov2_vits14_reid_v2` | `dinov2_vits14_reid_v2.onnx` |
| `ocr` | `parseq` | `parseq.onnx` |

Quedan afuera los modelos experimentales (daños/CarDD, rebajes y rostros/SCRFD) y las versiones
anteriores no activas (`rfdetr_openimages`, `dinov2_vits14`, `dinov2_vits14_reid`). CarDD no se
redistribuye y SCRFD requiere licencia comercial de InsightFace.

### De dónde salen los pesos

De un GitHub Release de este repo, `models-v1` (los pesos nunca van a git). Antes del build, el job
`publish` baja cada `.onnx` con `gh release download` y verifica su sha256 contra `models/manifest.json`;
si uno no coincide, falla y no se publica nada. El servicio vuelve a verificarlo al arrancar.

Para crear el release la primera vez, con los `.onnx` en `models/weights/`:

```bash
bash scripts/publicar-modelos.sh
```

Verifica los sha256 locales contra el manifest, crea el release `models-v1` y sube los cuatro archivos.
No pisa un release existente. Necesita `gh` autenticado con permiso de escritura sobre el repo.

### Actualizar los pesos

Un release de pesos no se modifica: se crea el siguiente.

1. Si cambia un modelo, actualizar `models/manifest.json` (archivo, `sha256`, versión) y los `.onnx`
   locales. Si cambia **qué** modelos entran, actualizar también la lista `MODELS` de
   `scripts/publicar-modelos.sh` y de `publish.yml`, los `COPY` y los `MODEL_*` del `Dockerfile` y la
   whitelist de `.dockerignore`.
2. Publicar el release nuevo: `bash scripts/publicar-modelos.sh models-v2`.
3. En `.github/workflows/publish.yml`, cambiar `MODELS_TAG: models-v1` por `models-v2`, y mergear.
4. Empujar un tag `vX.Y.Z` nuevo. Los tags anteriores quedan atados al release que usaron.

### Probar la imagen en local

```bash
docker build -t vinculo-vial-ia:local .   # con los pesos en models/weights/
```

El `.dockerignore` deja entrar al contexto solo los cuatro `.onnx` de la cascada.

### Tiempo de arranque y healthcheck

El servicio carga los 4 modelos (verificando su sha256) antes de levantar el servidor de salud y empezar
a consumir. Medido el 23/11/2026 con la imagen en Docker Desktop (Windows), con el rootfs de solo
lectura y sin caché: unos 75 s hasta `/ready` (detector ~30 s, segmentador ~30 s, re-ID ~12 s, OCR
~17 s, y el resto de las dependencias). En un host más lento puede tardar más.

Por eso el `healthcheck` de `docker-compose.yml` usa `start_period: 120s`: mientras dura, un fallo no
cuenta contra los reintentos. El `Dockerfile` no define `HEALTHCHECK`; quien corra la imagen fuera del
compose tiene que usar el mismo margen (`/health` en el puerto `HEALTH_PORT`, 8080 por defecto).
