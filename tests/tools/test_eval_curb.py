"""tools/eval/curb.py: decisión sobre las máscaras, fallas por causa y recomendación de RAMP_MARGIN.

Todo es sintético: máscaras rectangulares, un auto "dibujado" por su máscara y modelos falsos.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from vvi_ia.domain.model import CarInstance
from vvi_ia.domain.ramp import DrivewayCheck

REPO_ROOT = Path(__file__).resolve().parents[2]
SHAPE = (60, 100)
WHEEL = 10.0  # diámetro de rueda de las pruebas: el margen por defecto equivale a 0,8 px


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_curb", REPO_ROOT / "tools" / "eval" / "curb.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_curb"] = module
    spec.loader.exec_module(module)
    return module


curb_eval = load()


def box(y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    mask = np.zeros(SHAPE, bool)
    mask[y1:y2, x1:x2] = True
    return mask


CURB = box(40, 44, 2, 98)  # el cordón
CUT = box(40, 46, 30, 60)  # el rebaje: columnas 30 a 59
EMPTY = np.zeros(SHAPE, bool)


def car(x1: int, x2: int) -> np.ndarray:
    return box(46, 58, x1, x2)  # el auto, debajo del cordón


def sample(
    kind: str,
    car_mask: np.ndarray | None,
    gt: tuple[np.ndarray, np.ndarray] = (CURB, CUT),
    name: str = "f.jpg",
) -> Any:
    image = np.full((*SHAPE, 3), 90, np.uint8)
    return curb_eval.Sample(name, kind, image, gt[0], gt[1], car_mask, WHEEL)


# --- decisión -----------------------------------------------------------------------------------------


def test_sin_auto_es_no_car_y_no_se_consulta_al_dominio() -> None:
    assert curb_eval.decide(CURB, CUT, None, WHEEL) == DrivewayCheck.NO_CAR
    assert curb_eval.decide(CURB, CUT, EMPTY, WHEEL) == DrivewayCheck.NO_CAR


def test_un_auto_sobre_el_rebaje_bloquea() -> None:
    assert curb_eval.decide(CURB, CUT, car(35, 85), WHEEL) == DrivewayCheck.BLOCKING


def test_un_auto_justo_en_el_borde_es_marginal() -> None:
    assert curb_eval.decide(CURB, CUT, car(59, 90), WHEEL) == DrivewayCheck.BLOCKING_MARGINAL


def test_un_auto_junto_al_rebaje_no_lo_bloquea() -> None:
    assert curb_eval.decide(CURB, CUT, car(70, 95), WHEEL) == DrivewayCheck.NOT_BLOCKING


def test_sin_rebaje_no_hay_entrada() -> None:
    assert curb_eval.decide(CURB, EMPTY, car(35, 85), WHEEL) == DrivewayCheck.NO_DRIVEWAY


def test_el_margen_cambia_la_decision_en_el_borde() -> None:
    near = car(61, 90)  # 1 px de separación con el rebaje

    assert curb_eval.decide(CURB, CUT, near, WHEEL, margin=0.08) == DrivewayCheck.NOT_BLOCKING
    assert curb_eval.decide(CURB, CUT, near, WHEEL, margin=0.30) == DrivewayCheck.BLOCKING_MARGINAL


# --- máscaras anotadas y calidad -----------------------------------------------------------------------


def test_las_mascaras_anotadas_se_arman_con_los_poligonos_de_cada_clase() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "categories": [{"id": 1, "name": "curb"}, {"id": 2, "name": "curb_cut"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "segmentation": [[2, 40, 97, 40, 97, 43, 2, 43]]},
            {"image_id": 1, "category_id": 2, "segmentation": [[30, 40, 59, 40, 59, 45, 30, 45]]},
            {"image_id": 2, "category_id": 2, "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]]},
        ],
    }

    curb, cut = curb_eval.ground_truth(coco, "a.jpg", SHAPE)

    assert curb.sum() > 0 and cut.sum() > 0 and not (curb & cut).all()
    assert not cut[:10, :10].any()  # el polígono de la otra foto no entra
    assert cut[42, 40] and not cut[42, 70]


def test_iou_y_mascara_buena() -> None:
    assert curb_eval.iou(CUT, CUT) == 1.0
    assert curb_eval.iou(EMPTY, EMPTY) is None
    assert curb_eval.iou(CUT, box(40, 46, 31, 61)) == pytest.approx(29 * 6 / (31 * 6))
    assert curb_eval.is_good(EMPTY, EMPTY)  # las dos vacías: bien
    assert not curb_eval.is_good(EMPTY, CUT)
    assert not curb_eval.is_good(CUT, EMPTY)
    assert not curb_eval.is_good(box(10, 20, 0, 10), CUT)


# --- cada tipo de foto -----------------------------------------------------------------------------------


def test_cada_tipo_del_csv_tiene_su_decision_esperada() -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "curb"))
    import photos

    assert set(curb_eval.EXPECTED) == set(photos.TYPES) - {photos.DISCARDED}
    assert curb_eval.EXPECTED["positivo_claro"] == {DrivewayCheck.BLOCKING}
    assert curb_eval.EXPECTED["positivo_marginal"] == {
        DrivewayCheck.BLOCKING,
        DrivewayCheck.BLOCKING_MARGINAL,
    }
    assert curb_eval.EXPECTED["negativo_auto_junto_rebaje"] == {DrivewayCheck.NOT_BLOCKING}
    assert curb_eval.EXPECTED["negativo_auto_sin_rebaje"] == {DrivewayCheck.NO_DRIVEWAY}
    assert curb_eval.EXPECTED["negativo_parecido"] == {DrivewayCheck.NO_DRIVEWAY, DrivewayCheck.NO_CAR}
    assert curb_eval.EXPECTED["negativo_rebaje_sin_auto"] == {DrivewayCheck.NO_CAR}


def test_un_acierto() -> None:
    s = sample("positivo_claro", car(35, 85))

    result = curb_eval.judge(s, (CURB, CUT), DrivewayCheck.BLOCKING)

    assert result["ok"] and result["causa"] is None and result["mascara_buena"]


def test_decision_mala_con_mascara_buena() -> None:
    s = sample("positivo_claro", car(61, 90))

    result = curb_eval.judge(s, (CURB, CUT), DrivewayCheck.NOT_BLOCKING)

    assert not result["ok"] and result["causa"] == curb_eval.DECISION_BAD


def test_mascara_mala() -> None:
    s = sample("positivo_claro", car(35, 85))

    result = curb_eval.judge(s, (CURB, EMPTY), DrivewayCheck.NO_DRIVEWAY)

    assert not result["ok"] and result["causa"] == curb_eval.MASK_BAD and not result["mascara_buena"]


def test_rebaje_sin_auto_tiene_que_detectar_tambien_el_curb_cut() -> None:
    s = sample("negativo_rebaje_sin_auto", None)

    found = curb_eval.judge(s, (CURB, CUT), DrivewayCheck.NO_CAR)
    missed = curb_eval.judge(s, (CURB, EMPTY), DrivewayCheck.NO_CAR)

    assert found["ok"]
    assert not missed["ok"] and missed["causa"] == curb_eval.MASK_BAD  # NO_CAR, pero sin ver el rebaje


def test_negativo_parecido_acepta_no_driveway_y_no_car() -> None:
    s = sample("negativo_parecido", car(35, 85), gt=(CURB, EMPTY))

    assert curb_eval.judge(s, (CURB, EMPTY), DrivewayCheck.NO_DRIVEWAY)["ok"]
    assert curb_eval.judge(s, (CURB, EMPTY), DrivewayCheck.NO_CAR)["ok"]
    assert not curb_eval.judge(s, (CURB, EMPTY), DrivewayCheck.BLOCKING)["ok"]


# --- métricas ------------------------------------------------------------------------------------------


def judged(rows: list[tuple[str, DrivewayCheck, bool]]) -> list[tuple[Any, dict[str, Any]]]:
    out = []
    for kind, decision, good in rows:
        s = sample(kind, car(35, 85))
        prediction = (CURB, CUT) if good else (EMPTY, EMPTY)
        out.append((s, curb_eval.judge(s, prediction, decision)))
    return out


def test_precision_y_recall_de_blocking() -> None:
    rows = judged(
        [
            ("positivo_claro", DrivewayCheck.BLOCKING, True),  # acierto
            ("positivo_claro", DrivewayCheck.BLOCKING, True),  # acierto
            ("positivo_marginal", DrivewayCheck.BLOCKING_MARGINAL, True),  # no cuenta como BLOCKING
            ("positivo_claro", DrivewayCheck.NOT_BLOCKING, True),  # se perdió
            ("negativo_auto_junto_rebaje", DrivewayCheck.BLOCKING, True),  # falso positivo
        ]
    )

    summary = curb_eval.summarize(rows)

    assert summary["precision_blocking"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["recall_blocking"] == pytest.approx(2 / 4, abs=1e-4)
    assert summary["pct_marginal"] == 0.2 and summary["pct_marginal_en_positivas"] == 0.25


def test_resumen_por_tipo_y_por_causa_de_falla() -> None:
    rows = judged(
        [
            ("positivo_claro", DrivewayCheck.BLOCKING, True),
            ("positivo_claro", DrivewayCheck.NOT_BLOCKING, True),  # decisión mala con máscara buena
            ("positivo_claro", DrivewayCheck.NO_DRIVEWAY, False),  # máscara mala
            ("negativo_auto_sin_rebaje", DrivewayCheck.NO_DRIVEWAY, True),
        ]
    )

    summary = curb_eval.summarize(rows)

    assert summary["por_tipo"]["positivo_claro"] == {
        "fotos": 3,
        "aciertos": 1,
        "decisiones": {"BLOCKING": 1, "NOT_BLOCKING": 1, "NO_DRIVEWAY": 1},
    }
    assert summary["fallas"] == {curb_eval.MASK_BAD: 1, curb_eval.DECISION_BAD: 1}
    assert summary["aciertos"] == 2 and "negativo_parecido" not in summary["por_tipo"]


def test_sin_predicciones_blocking_no_hay_precision() -> None:
    summary = curb_eval.summarize(judged([("positivo_claro", DrivewayCheck.NOT_BLOCKING, True)]))

    assert summary["precision_blocking"] is None and summary["recall_blocking"] == 0.0


# --- RAMP_MARGIN ----------------------------------------------------------------------------------------


def test_el_barrido_encuentra_el_margen_que_acierta_mas() -> None:
    # Un positivo marginal con el auto a 2 px del rebaje: solo se acepta con un margen que lo vea como cercano
    samples = [
        sample("positivo_marginal", car(61, 90)),
        sample("positivo_marginal", car(61, 90), name="g.jpg"),
        sample("positivo_claro", car(35, 85), name="h.jpg"),
    ]

    sweep = curb_eval.margin_sweep(samples)
    best = curb_eval.recommend_margin(sweep)

    rates = {row["margen"]: row["tasa"] for row in sweep}
    assert rates[0.0] == pytest.approx(1 / 3, abs=1e-4)
    assert max(rates.values()) == pytest.approx(1.0)
    assert best is not None and rates[best] == pytest.approx(1.0)


def test_ante_un_empate_se_prefiere_el_margen_mas_cercano_al_actual() -> None:
    sweep = [{"margen": m, "tasa": 1.0} for m in (0.0, 0.06, 0.1, 0.3)]

    assert curb_eval.recommend_margin(sweep, default=0.08) == 0.06
    assert curb_eval.recommend_margin([{"margen": 0.0, "tasa": None}]) is None


# --- reporte y línea de comandos --------------------------------------------------------------------------


def test_el_reporte_compara_los_sistemas_y_dibuja_solo_las_fallas(tmp_path: Path) -> None:
    samples = [
        sample("positivo_claro", car(35, 85), name="uno.jpg"),
        sample("positivo_claro", car(35, 85), name="dos.jpg"),
    ]
    systems = {
        "bueno": lambda s: (s.gt_curb, s.gt_cut),
        "malo": lambda s: (EMPTY, EMPTY) if s.name == "dos.jpg" else (s.gt_curb, s.gt_cut),
        "roto": lambda s: None,
    }

    report, detail = curb_eval.build_report(systems, samples, 0.08, 0.45)
    written = curb_eval.write_failures(detail, tmp_path)

    assert report["sistemas"]["bueno"]["aciertos"] == 2
    assert (
        report["sistemas"]["malo"]["aciertos"] == 1
        and report["sistemas"]["malo"]["fallas"][curb_eval.MASK_BAD] == 1
    )
    assert (
        report["sistemas"]["roto"]["fallas_del_modelo"] == 2 and report["sistemas"]["roto"]["aciertos"] == 0
    )
    assert written == {"bueno": 0, "malo": 1, "roto": 2}
    assert [p.name for p in (tmp_path / "failures" / "malo").iterdir()] == ["dos.jpg"]
    assert "margen" in report and report["margen"]["actual"] == 0.08


def make_dataset(root: Path, rows: list[tuple[str, str]]) -> None:
    """`<root>/curb/` con el split test, fotos.csv y sus anotaciones."""
    curb = root / "curb"
    test = curb / "dataset" / "test"
    test.mkdir(parents=True)
    images, annotations, lines = [], [], []
    for index, (name, kind) in enumerate(rows, start=1):
        cv2.imwrite(str(test / name), np.full((*SHAPE, 3), 90, np.uint8))
        images.append({"id": index, "file_name": name, "width": SHAPE[1], "height": SHAPE[0]})
        annotations.append(
            {
                "id": 2 * index,
                "image_id": index,
                "category_id": 1,
                "segmentation": [[2, 40, 97, 40, 97, 43, 2, 43]],
            }
        )
        if kind != "negativo_auto_sin_rebaje":
            annotations.append(
                {
                    "id": 2 * index + 1,
                    "image_id": index,
                    "category_id": 2,
                    "segmentation": [[30, 40, 59, 40, 59, 45, 30, 45]],
                }
            )
        parecido = "boca_tormenta" if kind == "negativo_parecido" else ""
        lines.append(f"{name},2026-09-28,b,calle{index},{kind},garage,{parecido},dia,seco,ninguna,celular,")
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "curb"}, {"id": 2, "name": "curb_cut"}],
    }
    (test / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    (curb / "fotos.csv").write_text(
        "archivo,fecha,barrio,calle,tipo,rebaje,parecido,iluminacion,clima,oclusion,dispositivo,notas\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def fake_models() -> Any:
    big = np.zeros(SHAPE, bool)
    big[46:58, 35:85] = True
    return SimpleNamespace(
        cars=SimpleNamespace(cars=lambda img: [CarInstance(big, 0.9)]),
        curb=SimpleNamespace(curb=lambda img: (CURB, CUT)),
        vehicle=SimpleNamespace(detect=lambda img: []),
        versions=lambda: {"segmenter": "seg@abc", "curb": "curb@abc"},
    )


def test_main_de_punta_a_punta_sin_sam3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    make_dataset(root, [("a.jpg", "positivo_claro"), ("b.jpg", "negativo_auto_sin_rebaje")])
    monkeypatch.setattr(curb_eval, "build_models", lambda *a, **k: fake_models())

    code = curb_eval.main(
        [
            "--data-dir",
            str(root),
            "--out-dir",
            str(tmp_path / "out"),
            "--models-dir",
            str(tmp_path),
            "--no-sam3",
        ]
    )

    assert code == 0
    report = json.loads((tmp_path / "out" / "curb_report.json").read_text(encoding="utf-8"))
    assert set(report["sistemas"]) == {"entrenado", "oraculo"} and report["modelos"]["curb"] == "curb@abc"
    assert report["sistemas"]["oraculo"]["aciertos"] == 2  # con las máscaras anotadas acierta las dos
    shown = capsys.readouterr().out
    assert "RAMP_MARGIN" in shown and "a.jpg" not in shown and "b.jpg" not in shown


def test_main_con_sam3_inyectado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    make_dataset(root, [("a.jpg", "positivo_claro")])
    monkeypatch.setattr(curb_eval, "build_models", lambda *a, **k: fake_models())

    code = curb_eval.main(
        ["--data-dir", str(root), "--out-dir", str(tmp_path / "out"), "--models-dir", str(tmp_path)],
        sam3_factory=lambda args: lambda s: (s.gt_curb, s.gt_cut),
    )

    report = json.loads((tmp_path / "out" / "curb_report.json").read_text(encoding="utf-8"))
    assert code == 0 and list(report["sistemas"]) == ["sam3", "entrenado", "oraculo"]


def test_main_se_niega_a_trabajar_dentro_del_repo(capsys: pytest.CaptureFixture[str]) -> None:
    code = curb_eval.main(
        [
            "--data-dir",
            str(REPO_ROOT / "x"),
            "--out-dir",
            str(REPO_ROOT / "y"),
            "--models-dir",
            str(REPO_ROOT),
        ]
    )

    assert code == 1 and "repositorio" in capsys.readouterr().err


def test_main_sin_split_de_prueba(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = curb_eval.main(
        ["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--models-dir", str(tmp_path)]
    )

    assert code == 1 and "Corré antes prepare.py" in capsys.readouterr().err


def test_main_exige_las_rutas(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EVAL_DATA_DIR", "EVAL_OUT", "MODELS_DIR"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit):
        curb_eval.parse_args([])


def test_los_valores_por_defecto_coinciden_con_la_config() -> None:
    from vvi_ia.config import Settings

    assert curb_eval.DEFAULT_MARGIN == Settings.model_fields["ramp_margin"].default
    assert curb_eval.DEFAULT_SCALE_FALLBACK == Settings.model_fields["ramp_scale_fallback"].default


def test_main_sin_modelo_entrenado_evalua_sam3_y_el_oraculo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    make_dataset(root, [("a.jpg", "positivo_claro")])
    selections: list[Any] = []

    def build(selection: Any, *args: Any, **kwargs: Any) -> Any:
        selections.append(dict(selection))
        models = fake_models()
        models.versions = lambda: {"segmenter": "seg@abc"}  # no hay modelo de rebajes que versionar
        return models

    monkeypatch.setattr(curb_eval, "build_models", build)

    code = curb_eval.main(
        [
            "--data-dir",
            str(root),
            "--out-dir",
            str(tmp_path / "out"),
            "--models-dir",
            str(tmp_path),
            "--curb-model",
            "none",
        ],
        sam3_factory=lambda args: lambda s: (s.gt_curb, s.gt_cut),
    )

    report = json.loads((tmp_path / "out" / "curb_report.json").read_text(encoding="utf-8"))
    assert code == 0 and list(report["sistemas"]) == ["sam3", "oraculo"] and report["modelos"]["curb"] is None
    assert selections == [{"segmenter": "rfdetr_seg_coco"}]  # el modelo de rebajes ni se intenta cargar
