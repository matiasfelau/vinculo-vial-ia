"""16/11/2026: el producto solo usa datos con camino comercial; dos datasets no comerciales salieron."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("wider face", "widerface", "wider_face", "veri-776", "veri776", "reid_veri")
SCANNED = (
    *REPO_ROOT.glob("*.md"),
    *REPO_ROOT.glob("models/*.json"),
    *REPO_ROOT.glob(".env.example"),
    *(REPO_ROOT / "src").rglob("*.py"),
    *(REPO_ROOT / "tools").rglob("*.py"),
    *(REPO_ROOT / "tools").rglob("*.md"),
)


def test_ningun_archivo_nombra_wider_face_ni_veri_776() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}: {word}"
        for path in SCANNED
        for text in [path.read_text(encoding="utf-8").lower()]
        for word in FORBIDDEN
        if word in text
    ]

    assert offenders == []


def test_third_party_separa_los_modulos_experimentales_del_producto() -> None:
    text = (REPO_ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")

    assert "## 1. El producto" in text and "## 2. Módulos experimentales (fuera del producto)" in text
    experimental = text.split("## 2. Módulos experimentales")[1]
    assert all(word in experimental for word in ("CarDD", "SCRFD", "Rebajes"))
    product = text.split("## 2. Módulos experimentales")[0]
    assert "CarDD" not in product and "SCRFD" not in product
