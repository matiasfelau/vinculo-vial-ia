#!/usr/bin/env bash
# Publica los pesos de la cascada como un GitHub Release de este repo (por defecto, models-v1).
#
# Verifica el sha256 de cada .onnx local contra models/manifest.json, crea el release y sube los
# archivos. Si el release ya existe, o un hash no coincide, corta sin subir nada: un release de pesos
# no se pisa (para cambiar los pesos se crea el siguiente, models-v2, ver el README).
#
# Uso:    bash scripts/publicar-modelos.sh [TAG]
# Entorno: MODELS_DIR (carpeta con los .onnx; por defecto models/weights)
# Requiere: gh (autenticado con permiso de escritura sobre el repo), sha256sum y Python 3.
#
# Solo se suben los modelos de la cascada por defecto. No agregar los experimentales: CarDD no se
# redistribuye (licencia del PIC Lab) y SCRFD requiere licencia comercial de InsightFace.
set -euo pipefail

# Debe coincidir con MODELS de .github/workflows/publish.yml y con los COPY del Dockerfile.
MODELS=(rfdetr_openimages_v2 rfdetr_seg_coco dinov2_vits14_reid_v2 parseq)

TAG="${1:-models-v1}"
if [[ ! "$TAG" =~ ^models-v[0-9]+$ ]]; then
  echo "error: el tag '$TAG' no tiene el formato models-vN" >&2
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MANIFEST="models/manifest.json"
WEIGHTS_DIR="${MODELS_DIR:-models/weights}"

for tool in gh sha256sum; do
  command -v "$tool" >/dev/null || { echo "error: falta $tool" >&2; exit 1; }
done
PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
[[ -n "$PYTHON" ]] || { echo "error: falta Python 3" >&2; exit 1; }

# "<archivo> <sha256>" de un modelo del manifest.
manifest_entry() {
  "$PYTHON" -I -c '
import json, sys
name = sys.argv[2]
with open(sys.argv[1], encoding="utf-8") as f:
    models = json.load(f)["models"]
for m in models:
    if m["name"] == name and m.get("sha256"):
        print(m["file"], m["sha256"])
        break
else:
    sys.exit(1)
' "$MANIFEST" "$1"
}

FILES=()
NOTES="Pesos de la cascada por defecto de vinculo-vial-ia, verificados contra models/manifest.json."$'\n'
for name in "${MODELS[@]}"; do
  entry="$(manifest_entry "$name")" || { echo "error: $name no está en $MANIFEST o no tiene sha256" >&2; exit 1; }
  file="${entry% *}"
  expected="${entry#* }"
  path="$WEIGHTS_DIR/$file"
  [[ -f "$path" ]] || { echo "error: falta $path" >&2; exit 1; }
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "error: el sha256 de $path no coincide con el manifest" >&2
    echo "  esperado: $expected" >&2
    echo "  obtenido: $actual" >&2
    exit 1
  fi
  echo "OK $file $actual"
  FILES+=("$path")
  NOTES+=$'\n'"- \`$file\` (\`$name\`): sha256 \`$expected\`"
done

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "error: el release $TAG ya existe; no se pisa (publicá el siguiente, models-vN+1)" >&2
  exit 1
fi

# --latest=false: este release no es una versión del servicio y no debe quedar como el último.
gh release create "$TAG" "${FILES[@]}" \
  --title "Pesos de la cascada ($TAG)" \
  --notes "$NOTES" \
  --latest=false

echo "Release $TAG creado con ${#FILES[@]} modelos."
