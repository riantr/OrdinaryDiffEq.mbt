#!/usr/bin/env python3
"""Extract the Verner (Vern7 / Vern9) tableaus from OrdinaryDiffEq.jl into
MoonBit ``const`` declarations.

Three tables per method, all from::

    OrdinaryDiffEq.jl/lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl

  * ``VernNTableau``              — stage abscissae ``c``, the strictly
    lower-triangular ``a`` weights, the solution weights ``b`` and the
    embedded ``btilde`` error weights,
  * ``VernNExtraStages``          — the ``a``/``c`` of the extra stages the
    dense interpolant needs (``k11..k16`` for Vern7, ``k17..k26`` for Vern9),
  * ``VernNInterpolationCoefficients`` — the interpolant polynomials ``r``.

Each table is defined twice: once constrained to ``CompiledFloats`` (the
Float64 runtime dispatch) and once generic with high-precision literals. This
tool reads the ``CompiledFloats`` variant, matching what Float64 Julia
actually executes.

Invariants checked before anything is written:

  * every stage row of ``a`` sums to its abscissa ``c``,
  * the solution weights sum to 1 (constants are integrated exactly),
  * ``btilde`` is non-zero (an embedded pair that estimates nothing is a bug),
  * the interpolant reproduces the step endpoints: ``b_i(1)`` equals the
    ``b_i`` used by the step's solution formula for every stage the
    interpolant uses, and ``b_1'(0) = 1``, ``b_i'(0) = 0`` for ``i >= 2``.

Usage::

    python tools/extract_verner.py --method vern7 [--check]
    python tools/extract_verner.py --method vern9 [--check]

``--check`` re-extracts and diffs against the existing file; it never writes.
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERNER_SRC = REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" / "OrdinaryDiffEqVerner" / "src" / "verner_tableaus.jl"

INVARIANT_TOL = 1.0e-9

# `name = convert(T, value)` / `convert(T2, ...)`. Commented-out lines start
# with `#` and never match.
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*convert\(\s*T2?\s*,\s*(.+?)\s*\)\s*$")
RATIONAL_RE = re.compile(r"^(-?\d+)\s*//\s*(-?\d+)$")


class Method:
    """Per-method extraction recipe: naming, stage layout, invariants."""

    def __init__(self, key, moon_prefix, stages, tableau_name, extra_name,
                 interp_name, out_path, stage_groups, b_names, btilde_names,
                 interp_spec, extra_spec):
        self.key = key
        self.moon_prefix = moon_prefix
        self.stages = stages                  # total stages incl. extra ones
        self.tableau_name = tableau_name
        self.extra_name = extra_name
        self.interp_name = interp_name
        self.out_path = out_path
        # stage_groups: {stage_number: [coefficient names in that a row]}
        self.stage_groups = stage_groups
        self.b_names = b_names
        self.btilde_names = btilde_names
        # interp_spec: list of (k_slot_1based, stage_number)
        self.interp_spec = interp_spec
        # extra_spec: list of (k_slot_1based, stage_number, [a coefficient names])
        self.extra_spec = extra_spec


def _struct_fields(text: str, struct_name: str) -> list:
    """Field names of `struct <struct_name>{...}`, in declaration order."""
    m = re.search(rf"^struct {struct_name}\{{[^\n]*\n.*?^end$", text, re.M | re.S)
    if not m:
        raise SystemExit(f"error: struct {struct_name} not found")
    return re.findall(r"^\s*([A-Za-z_]\w*)::", m.group(0), re.M)


def _extra_stage_spec(text: str, struct_name: str, first_slot: int):
    """`(k_slot, stage, [a names])` per extra stage, read off the struct.

    The rows are not uniform (Vern9's stage 24 omits a2422, stage 26 stops at
    a2621, ...), so the field list is the source of truth. Each stage starts at
    its `c{nn}` field and is followed by that stage's `a` names; the `k` slot it
    lands in is offset from the smallest stage number in the struct.
    """
    groups = []
    for name in _struct_fields(text, struct_name):
        if re.fullmatch(r"c\d+", name):
            groups.append([int(name[1:]), []])
        elif groups:
            groups[-1][1].append(name)
    if not groups:
        raise SystemExit(f"error: no extra stages found in {struct_name}")
    first = groups[0][0]
    return [(first_slot + stage - first, stage, names) for stage, names in groups]


def _vern7_recipe(text):
    # perform_step! (verner_rk_perform_step.jl:255) computes k1..k10 and stores
    # them as integrator.k[1..10]; the extra stages fill k[11..16].
    groups = {
        2: ["a021"],
        3: ["a031", "a032"],
        4: ["a041", "a043"],
        5: ["a051", "a053", "a054"],
        6: ["a061", "a063", "a064", "a065"],
        7: ["a071", "a073", "a074", "a075", "a076"],
        8: ["a081", "a083", "a084", "a085", "a086", "a087"],
        9: ["a091", "a093", "a094", "a095", "a096", "a097", "a098"],
        10: ["a101", "a103", "a104", "a105", "a106", "a107"],
    }
    # interpolants.jl `vern7pre0`: k[1], k[4..9], k[11..16].
    interp_spec = [(1, 1)] + [(s, s) for s in range(4, 10)] + [(s, s) for s in range(11, 17)]
    return Method(
        key="vern7", moon_prefix="Vern7", stages=16,
        tableau_name="Vern7Tableau", extra_name="Vern7ExtraStages",
        interp_name="Vern7InterpolationCoefficients",
        out_path=REPO_ROOT / "src" / "src" / "alg_vern7_coeffs.mbt",
        stage_groups=groups,
        b_names=["b1", "b4", "b5", "b6", "b7", "b8", "b9"],
        btilde_names=["btilde1", "btilde4", "btilde5", "btilde6", "btilde7",
                      "btilde8", "btilde9", "btilde10"],
        interp_spec=interp_spec,
        extra_spec=_extra_stage_spec(text, "Vern7ExtraStages", 11),
    )


def _vern9_recipe(text):
    # perform_step! (verner_rk_perform_step.jl:1021) computes k1..k16 but only
    # k1 and k8..k16 are kept: integrator.k[1]=k1, k[2..10]=k8..k16. The extra
    # stages fill k[11..20].
    groups = {
        2: ["a0201"],
        3: ["a0301", "a0302"],
        4: ["a0401", "a0403"],
        5: ["a0501", "a0503", "a0504"],
        6: ["a0601", "a0604", "a0605"],
        7: ["a0701", "a0704", "a0705", "a0706"],
        8: ["a0801", "a0806", "a0807"],
        9: ["a0901", "a0906", "a0907", "a0908"],
        10: ["a1001", "a1006", "a1007", "a1008", "a1009"],
        11: ["a1101", "a1106", "a1107", "a1108", "a1109", "a1110"],
        12: ["a1201", "a1206", "a1207", "a1208", "a1209", "a1210", "a1211"],
        13: ["a1301", "a1306", "a1307", "a1308", "a1309", "a1310", "a1311", "a1312"],
        14: ["a1401", "a1406", "a1407", "a1408", "a1409", "a1410", "a1411",
             "a1412", "a1413"],
        # k15 and k16 are the pair evaluated at `t + dt` from two different
        # rows (g15 / g16), which is what makes the embedded estimate free.
        # Neither carries a `c` field: their abscissa is implied.
        15: ["a1501", "a1506", "a1507", "a1508", "a1509", "a1510", "a1511",
             "a1512", "a1513", "a1514"],
        16: ["a1601", "a1606", "a1607", "a1608", "a1609", "a1610", "a1611",
             "a1612", "a1613"],
    }
    # integrator.k[1] = k1 and integrator.k[2..10] = k8..k16; k10 (= k16) only
    # feeds the error estimate. The extra stages then occupy k[11..20].
    # The interpolant pairs k slot -> the stage whose polynomial weights it
    # uses (interpolants.jl `vern9pre0`), skipping the k[10] slot.
    interp_spec = [(1, 1)] + [(slot, slot + 6) for slot in range(2, 10)] + \
                  [(slot, slot + 6) for slot in range(11, 21)]
    return Method(
        key="vern9", moon_prefix="Vern9", stages=20,
        tableau_name="Vern9Tableau", extra_name="Vern9ExtraStages",
        interp_name="Vern9InterpolationCoefficients",
        out_path=REPO_ROOT / "src" / "src" / "alg_vern9_coeffs.mbt",
        stage_groups=groups,
        b_names=["b1", "b8", "b9", "b10", "b11", "b12", "b13", "b14", "b15"],
        btilde_names=["btilde1", "btilde8", "btilde9", "btilde10", "btilde11",
                      "btilde12", "btilde13", "btilde14", "btilde15", "btilde16"],
        interp_spec=interp_spec,
        extra_spec=_extra_stage_spec(text, "Vern9ExtraStages", 11),
    )


RECIPES = {"vern7": _vern7_recipe, "vern9": _vern9_recipe}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def split_functions(text: str):
    """Yield ``(name, signature, body, body_start_line)`` per function.

    Handles both ``@fold function Name(...)`` and plain ``function Name(...)``.
    A body ends at the first line that is exactly ``end`` at column 0.
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i]
        starts = stripped.startswith("@fold function") or stripped.startswith("function ")
        if not starts:
            i += 1
            continue
        name = stripped.split("function", 1)[1].split("(", 1)[0].strip()
        # Consume the signature (continuation lines are indented).
        j = i + 1
        while j < n and lines[j] != "end" and not ASSIGN_RE.match(lines[j]):
            j += 1
        signature = "\n".join(lines[i:j])
        k = j
        while k < n and lines[k] != "end":
            k += 1
        body = "\n".join(lines[j:k])
        yield name, signature, body, j + 1
        i = k + 1


def parse_assignments(body: str, body_start_line: int) -> dict:
    """``{name: (value, line_number)}`` for every ``convert`` assignment."""
    out = {}
    for offset, line in enumerate(body.split("\n")):
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2).strip()
        rm = RATIONAL_RE.match(raw)
        value = int(rm.group(1)) / int(rm.group(2)) if rm else float(raw)
        out[name] = (value, body_start_line + offset)
    return out


def pick(text: str, name: str) -> dict:
    """The ``CompiledFloats`` method of `name`, as an assignment dict."""
    for fname, sig, body, start in split_functions(text):
        if fname != name:
            continue
        if "CompiledFloats" in sig:
            return parse_assignments(body, start)
    raise SystemExit(f"error: no CompiledFloats method found for {name}")


def fmt(value: float) -> str:
    """Shortest round-trip decimal, never in exponent notation."""
    return format(Decimal(repr(value)), "f")


def moon_name(prefix: str, julia_name: str) -> str:
    return prefix + julia_name[0].upper() + julia_name[1:]


# ---------------------------------------------------------------------------
# Interpolation weight evaluation (for the invariant checks)
# ---------------------------------------------------------------------------

# How many polynomial coefficients each interpolant stage carries. Vern7's `r01`
# block is r011..r017 (7 terms) and each later stage drops two leading powers of
# Theta, leaving 6 (r{stage}2..r{stage}7); Vern9 has 9 and 8. Stage numbers are
# zero-padded to two digits in the Julia field names.
INTERP_DEGREE = {"vern7": (7, 7), "vern9": (9, 9)}

# Vern7 names the abscissa of stage s `c{s}`; Vern9 names it `c{s-1}` (its
# stage 2 sits at t + c1*dt). Only the stages that carry a `c` field need this.
C_OFFSET = {"vern7": 0, "vern9": 1}


def c_field(recipe: Method, stage: int) -> str:
    return f"c{stage - C_OFFSET[recipe.key]}"


def interp_names(recipe: Method, stage: int) -> list:
    """Julia field names of stage `stage`'s interpolant polynomial."""
    p_deg, q_deg = INTERP_DEGREE[recipe.key]
    if stage == 1:
        return [f"r01{i}" for i in range(1, p_deg + 1)]
    return [f"r{stage:02d}{i}" for i in range(2, q_deg + 1)]


def interp_poly(coeffs: dict, stage: int, x: float, recipe: Method) -> float:
    """b_i(x) for the interpolant: `x * P(x)` for stage 1, `x^2 * P(x)` else.

    Each polynomial is evaluated as a plain power series in x with the
    constant term first, which is exactly Julia's `@evalpoly`.
    """
    names = interp_names(recipe, stage)
    lead = 1 if stage == 1 else 2
    acc = 0.0
    xpow = 1.0
    for nm in names:
        acc += coeffs[nm][0] * xpow
        xpow *= x
    return (x ** lead) * acc


def interp_poly_prime(coeffs: dict, stage: int, x: float, recipe: Method) -> float:
    """d/dx b_i(x), by central differences (invariant checks only)."""
    h = 1.0e-4
    return (interp_poly(coeffs, stage, x + h, recipe)
            - interp_poly(coeffs, stage, x - h, recipe)) / (2 * h)


# ---------------------------------------------------------------------------
# Extraction + invariants
# ---------------------------------------------------------------------------

def extract(recipe: Method, text: str) -> dict:
    tab = pick(text, recipe.tableau_name)
    extra = pick(text, recipe.extra_name)
    interp = pick(text, recipe.interp_name)

    coef = {"tab": tab, "extra": extra, "interp": interp}

    # Every referenced name must exist. Stages evaluated at `t + dt` carry no
    # `c` field in Julia (the abscissa is implied), so only stages that do have
    # one are requested here.
    wanted = []
    for stage, names in recipe.stage_groups.items():
        wanted += names
        if c_field(recipe, stage) in tab:
            wanted.append(c_field(recipe, stage))
    wanted += recipe.b_names + recipe.btilde_names
    for _slot, stage, names in recipe.extra_spec:
        wanted.append(f"c{stage}")
        wanted += names
    for _slot, stage in recipe.interp_spec:
        wanted += interp_names(recipe, stage)
    missing = sorted({n for n in wanted if n not in tab and n not in extra and n not in interp})
    if missing:
        raise SystemExit(f"error: {recipe.key}: missing coefficients: {missing}")
    return coef


def check_invariants(recipe: Method, coef: dict) -> list:
    tab, extra, interp = coef["tab"], coef["extra"], coef["interp"]
    val = lambda d, n: d[n][0]
    problems = []

    # Row sums == abscissae. The final stage of each group is evaluated at
    # `t + dt` and Julia stores no `c` for it, so its abscissa is 1.
    for stage, names in sorted(recipe.stage_groups.items()):
        total = sum(val(tab, n) for n in names)
        field = c_field(recipe, stage)
        c = val(tab, field) if field in tab else 1.0
        if abs(total - c) > INVARIANT_TOL:
            problems.append(f"row {stage} sums to {total!r}, {field} = {c!r}")

    # Solution weights integrate constants exactly.
    b_total = sum(val(tab, n) for n in recipe.b_names)
    if abs(b_total - 1.0) > INVARIANT_TOL:
        problems.append(f"sum(b) = {b_total!r}, not 1")

    # The embedded pair must actually estimate something.
    bt_total = sum(abs(val(tab, n)) for n in recipe.btilde_names)
    if bt_total < INVARIANT_TOL:
        problems.append("btilde is all zero")

    # Extra stage rows must sum to their abscissae too.
    for _slot, stage, names in recipe.extra_spec:
        total = sum(val(extra, n) for n in names)
        c = val(extra, f"c{stage}")
        if abs(total - c) > INVARIANT_TOL:
            problems.append(f"extra row {stage} sums to {total!r}, c{stage} = {c!r}")

    # The interpolant must reproduce the step endpoints: b_i(1) equals the
    # solution weight of stage i (0 for stages the step does not use, which is
    # how Vern7 handles k10).
    b_map = {name: val(tab, name) for name in recipe.b_names}
    for _slot, stage in recipe.interp_spec:
        got = interp_poly(interp, stage, 1.0, recipe)
        want = b_map.get(f"b{stage}", 0.0)
        if abs(got - want) > INVARIANT_TOL:
            problems.append(f"interp b_{stage}(1) = {got!r} != b{stage} = {want!r}")

    # And its derivative must start at the first stage.
    d1 = interp_poly_prime(interp, 1, 0.0, recipe)
    if abs(d1 - 1.0) > 1.0e-5:
        problems.append(f"interp b_1'(0) = {d1!r}, expected 1")
    for _slot, stage in recipe.interp_spec:
        if stage == 1:
            continue
        d = interp_poly_prime(interp, stage, 0.0, recipe)
        if abs(d) > 1.0e-5:
            problems.append(f"interp b_{stage}'(0) = {d!r}, expected 0")
    return problems


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit(recipe: Method, coef: dict, src: Path) -> str:
    p = recipe.moon_prefix
    tab, extra, interp = coef["tab"], coef["extra"], coef["interp"]
    lines = []
    add = lines.append
    add(f"// {p} tableau, extra-stage and dense-interpolant coefficients.")
    add("//")
    add("// GENERATED by tools/extract_verner.py from")
    add(f"//   {src.name}")
    add("// (the `CompiledFloats` methods, i.e. the Float64 dispatch Julia uses")
    add("// at runtime; the generic `big\"...\"` methods carry the same values to")
    add("// higher precision). Do not edit by hand; re-run the tool instead.")

    def emit_const(name: str, payload: tuple, julia_name: str | None = None) -> None:
        value, line = payload
        jn = julia_name or name
        add("///|")
        add(f"const {moon_name(p, name)} : Double = {fmt(value)} // verner_tableaus.jl:{line} ({jn})")

    def section(title: str, *notes: str) -> None:
        add("")
        add(f"// --- {title} " + "-" * max(0, 66 - len(title)))
        for note in notes:
            add(f"// {note}")

    def stage_group(stage: int, names: list, fields: dict) -> None:
        add("")
        add(f"// stage {stage}")
        for name in names:
            emit_const(name, fields[name])

    section("Stage abscissae c",
            "The final stage of each group is evaluated at t + dt and carries no c.")
    for stage in sorted(recipe.stage_groups):
        field = c_field(recipe, stage)
        if field in tab:
            add("")
            emit_const(field, tab[field])

    section("Stage coefficients a (one group per stage)")
    for stage in sorted(recipe.stage_groups):
        stage_group(stage, recipe.stage_groups[stage], tab)

    section("Solution weights b")
    for name in recipe.b_names:
        add("")
        emit_const(name, tab[name])

    section("Embedded lower-order weights btilde (error estimate, M2)")
    for name in recipe.btilde_names:
        add("")
        emit_const(name, tab[name])

    section("Extra stages for the dense interpolant",
            "Computed on demand by `fill_dense_stages`, never counted in nf.")
    for _slot, stage, names in recipe.extra_spec:
        add("")
        add(f"// stage {stage} (k slot {_slot})")
        emit_const(f"c{stage}", extra[f"c{stage}"])
        for name in names:
            emit_const(name, extra[name])

    section("Dense interpolant weights r",
            "b_1(theta) = theta * P(r01x);  b_i(theta) = theta^2 * P(rix).",
            "Stages the interpolant does not use have no r block.")
    for stage in sorted({stage for _slot, stage in recipe.interp_spec}):
        add("")
        add(f"// stage {stage}")
        for name in interp_names(recipe, stage):
            emit_const(name, interp[name])
    add("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=sorted(RECIPES), required=True)
    ap.add_argument("--julia-src", type=Path, default=VERNER_SRC)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not args.julia_src.exists():
        print(f"error: Julia source not found: {args.julia_src}", file=sys.stderr)
        return 2

    text = args.julia_src.read_text(encoding="utf-8")
    recipe = RECIPES[args.method](text)
    out = args.out or recipe.out_path
    coef = extract(recipe, text)

    problems = check_invariants(recipe, coef)
    for prob in problems:
        print(f"invariant violated: {prob}", file=sys.stderr)
    if problems:
        return 1

    content = emit(recipe, coef, args.julia_src)
    if args.check:
        if not out.exists():
            print(f"error: {out} does not exist (run without --check)", file=sys.stderr)
            return 2
        if out.read_text(encoding="utf-8") != content:
            print(f"error: {out} is stale; re-run tools/extract_verner.py", file=sys.stderr)
            return 1
        print(f"ok: {out} matches {args.julia_src.name}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    n = len(coef["tab"]) + len(coef["extra"]) + len(coef["interp"])
    print(f"wrote {out} ({n} coefficients, invariants hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())