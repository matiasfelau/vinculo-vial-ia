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
