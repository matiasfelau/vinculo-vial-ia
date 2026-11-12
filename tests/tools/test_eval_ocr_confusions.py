"""tools/eval/ocr_confusions.py: diferencias, variantes V0, V1 y V2, lista "dejar una afuera" y métricas."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script() -> ModuleType:
    tools = str(REPO_ROOT / "tools" / "eval")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "eval_ocr_confusions", REPO_ROOT / "tools" / "eval" / "ocr_confusions.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_ocr_confusions"] = module
    spec.loader.exec_module(module)
    return module


oc = load_script()


def reading(label: str, text: str, probs: list[float] | None = None, in_scope: bool = True) -> object:
    return oc.Reading(label, text, tuple(probs or [0.95] * len(text)), in_scope)


def test_la_normalizacion_por_posicion_no_deja_diferencias() -> None:
    assert oc.differences("AB012CD", reading("AB012CD", "ABO12CD")) == []


def test_una_diferencia_real_lleva_la_confianza_del_caracter_leido() -> None:
    probs = [0.9, 0.9, 0.9, 0.9, 0.55, 0.9, 0.9]

    diffs = oc.differences("AB123CD", reading("AB123CD", "AB129CD", probs))

    assert [(d.reported, d.read, d.prob) for d in diffs] == [("3", "9", 0.55)]
    assert diffs[0].pair == ("3", "9")


def test_con_largos_distintos_se_alinea_y_el_que_falta_no_tiene_pareja() -> None:
    diffs = oc.differences("AB123CD", reading("AB123CD", "AB12CD"))

    assert len(diffs) == 1 and diffs[0].read is None and diffs[0].pair is None


def test_v0_pide_dos_diferencias_y_lectura_segura() -> None:
    two = reading("AB123CD", "AB129CE")
    one = reading("AB123CD", "AB129CD")
    weak = reading("AB123CD", "AB129CE", [0.5] * 7)
    none = frozenset()

    assert oc.v0(oc.differences("AB123CD", two), two, none, 0.78, 0.0)
    assert not oc.v0(oc.differences("AB123CD", one), one, none, 0.78, 0.0)
    assert not oc.v0(oc.differences("AB123CD", weak), weak, none, 0.78, 0.0)


def test_v1_no_cuenta_las_confusiones_frecuentes() -> None:
    r = reading("AB123CD", "AB129CE")  # 3/9 y D/E
    confusions = frozenset({("3", "9")})

    diffs = oc.differences("AB123CD", r)

    assert oc.v0(diffs, r, frozenset(), 0.78, 0.0)
    assert not oc.v1(diffs, r, confusions, 0.78, 0.0)  # queda una sola diferencia: duda


def test_v2_con_una_diferencia_segura_y_fuera_de_la_lista_es_mismatch() -> None:
    r = reading("AB123CD", "AB129CD")

    diffs = oc.differences("AB123CD", r)

    assert oc.v2(diffs, r, frozenset(), 0.0, 0.78)
    assert not oc.v2(diffs, r, frozenset({("3", "9")}), 0.0, 0.78)  # confusión frecuente: duda


def test_v2_una_diferencia_poco_segura_es_duda() -> None:
    probs = [0.95, 0.95, 0.95, 0.4, 0.95, 0.95, 0.95]
    r = reading("AB123CD", "AB193CD", probs)  # 2 → 9 leído con 0,4 (la lectura entera queda poco segura)

    assert not oc.v2(oc.differences("AB123CD", r), r, frozenset(), 0.0, 0.78)
    assert oc.v2(oc.differences("AB123CD", r), r, frozenset(), 0.0, 0.30)


def test_la_lista_dejar_una_afuera_no_aprende_de_la_propia_lectura() -> None:
    # Con MIN_PAIR_COUNT = 2, cada lectura necesita que OTRAS dos tengan la pareja
    three = [reading(f"AB{n}23CD", f"AB{n}29CD") for n in (1, 4, 5)]
    two = three[:2]

    assert all(("3", "9") not in lst for lst in oc.leave_one_out(two))
    assert all(("3", "9") in lst for lst in oc.leave_one_out(three))


def test_un_recorte_roto_no_aporta_parejas() -> None:
    assert oc.error_pairs(reading("AB123CD", "XYZ9876")) == {}
    assert oc.error_pairs(reading("AB123CD", "ZZZZZZZ")) == {}  # más de tres diferencias


def test_evaluate_cuenta_s5_y_u2() -> None:
    readings = [reading("AB123CD", "AB123CD"), reading("EF456GH", "EF456GH"), reading("IJ789KL", "IJ789KL")]
    lists = [frozenset()] * 3

    result = oc.evaluate(readings, lists, oc.v0, 0.78, 0.0)

    assert result["S5_pares"] == 6 and result["S5_detectadas"] == 6 and result["S5"] == 1.0
    assert result["U2_marcadas"] == 0 and result["U2_lecturas"] == 3
