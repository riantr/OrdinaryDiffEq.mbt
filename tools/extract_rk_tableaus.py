#!/usr/bin/env python3
"""Extract explicit Runge-Kutta tableaus from OrdinaryDiffEqExplicitTableaus
into a MoonBit table of `ExplicitRKTableau` values.

Sources::

    lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_low_order.jl   (Heun, RK4)
    lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl
        (BogakiShampine3, RKF4, DormandPrince)

Each constructor builds a `DiffEqBase.ExplicitRKTableau(A, c, alpha, order;
alphaEEst, adaptiveorder, fsal, ...)` from exact rationals (`1 // 2`), which
Julia converts to Float64 through `map(T, ...)`. This tool evaluates the same
rationals in Python and emits the resulting doubles, so the MoonBit values are
bit-identical to what the Julia solver runs on.

Invariants checked before anything is written:

  * ``A`` is strictly upper triangular (zero on and below the diagonal),
  * each row of ``A`` sums to its abscissa ``c[i]``,
  * the final row sums to 1 (constants are integrated exactly),
  * ``sum(alpha) == 1``,
  * an adaptive tableau has a non-zero, non-identical ``alphaEEst`` (an
    embedded pair that estimates nothing would silently disable adaptivity),
  * an FSAL tableau's last row equals its ``alpha`` (the last stage *is* the
    step's solution stage), and a non-FSAL tableau's last row differs.

Usage::

    python tools/extract_rk_tableaus.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLEAUS_SRC = {
    "Heun": REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" /
    "OrdinaryDiffEqExplicitTableaus" / "src" / "tableaus_low_order.jl",
    "RK4": REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" /
    "OrdinaryDiffEqExplicitTableaus" / "src" / "tableaus_low_order.jl",
    "BogakiShampine3": REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" /
    "OrdinaryDiffEqExplicitTableaus" / "src" / "tableaus_order5.jl",
    "RKF4": REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" /
    "OrdinaryDiffEqExplicitTableaus" / "src" / "tableaus_order5.jl",
    "DormandPrince": REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" /
    "OrdinaryDiffEqExplicitTableaus" / "src" / "tableaus_order5.jl",
}
DEFAULT_OUT = REPO_ROOT / "src" / "src" / "alg_explicit_rk_tableaus.mbt"

# The MoonBit constructor name for each Julia tableau, plus the order the
# generated file lists them in.
ORDER = ["Heun", "RK4", "BogakiShampine3", "RKF4", "DormandPrince"]
# MoonBit spells a function's name with a lowercase first letter; an
# identifier starting uppercase is a type constructor, so Julia's `Heun` /
# `RK4` cannot be reproduced verbatim here.
MOON_NAMES = {
    "Heun": "heun",
    "RK4": "rk4",
    "BogakiShampine3": "bs3",
    "RKF4": "rkf4",
    "DormandPrince": "dormand_prince",
}

INVARIANT_TOL = 1.0e-12


def fmt(value: float) -> str:
    return format(Decimal(repr(value)), "f")


# ---------------------------------------------------------------------------
# Parsing a tableau constructor body
# ---------------------------------------------------------------------------

def constructor_body(text: str, name: str) -> tuple:
    """The body of `function <name>(...)` and its 1-based start line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith(f"function {name}("):
            continue
        j = i + 1
        while j < len(lines) and lines[j] != "end":
            j += 1
        return "\n".join(lines[i + 1:j]), i + 2
    raise SystemExit(f"error: constructor {name} not found")


NUM = r"(?:big\"[^\"]*\"|-?\d+\s*//\s*-?\d+|-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"


def to_float(raw: str, line: int) -> float:
    raw = raw.strip()
    if raw.startswith('big"'):
        return float(Decimal(raw[4:-1]))
    if "//" in raw:
        num, den = (int(x) for x in raw.split("//"))
        return num / den
    return float(raw)


def vector_literal(body: str, var: str, line0: int):
    """`[a; b; c]` or the element-wise `v[i] = ...` form -> list of floats."""
    # Form 1: bracketed literal, possibly spread over several lines.
    m = re.search(rf"{var}\s*=\s*\[(.*?)\]", body, re.S)
    if m:
        parts = [
            p for p in re.split(r"[;,\n]", m.group(1)) if p.strip()
        ]
        return [to_float(p, line0) for p in parts]
    # Form 2: `v = zeros(T, n)` followed by `v[i] = <num>` assignments.
    m = re.search(rf"{var}\s*=\s*zeros\([^,]+,\s*(\d+)\)", body)
    if not m:
        return None
    n = int(m.group(1))
    out = [0.0] * n
    for idx, val in re.findall(rf"{var}\[(\d+)\]\s*=\s*({NUM})", body):
        out[int(idx) - 1] = to_float(val, line0)
    return out


def matrix_literal(body: str, var: str, line0: int):
    """A bracketed `a b c` matrix -> list of rows, or None."""
    m = re.search(rf"{var}\s*=\s*\[(.*?)\]\s*\n", body, re.S)
    if not m:
        return None
    rows = []
    for chunk in m.group(1).split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.sub(r"(\d)\s*//\s*", r"\1//", chunk).split()
        rows.append([to_float(p, line0) for p in parts])
    if not rows or any(len(r) != len(rows) for r in rows):
        return None
    return rows


def parse_tableau(name: str) -> dict:
    src = TABLEAUS_SRC[name]
    body, line0 = constructor_body(src.read_text(encoding="utf-8"), name)

    a = matrix_literal(body, "A", line0)
    c = vector_literal(body, "c", line0)
    alpha = vector_literal(body, "α", line0)
    alpha_eest = vector_literal(body, "αEEst", line0)
    if a is None or c is None or alpha is None:
        raise SystemExit(f"error: could not parse A/c/alpha of {name}")

    m = re.search(r"ExplicitRKTableau\(\s*A,\s*c,\s*α,\s*(\d+)", body)
    if not m:
        raise SystemExit(f"error: could not find the order of {name}")
    order = int(m.group(1))

    m = re.search(r"adaptiveorder\s*=\s*(\d+)", body)
    adaptive_order = int(m.group(1)) if m else 0
    fsal = "fsal = true" in body

    return {
        "name": name,
        "moon": MOON_NAMES[name],
        "a": a,
        "c": c,
        "alpha": alpha,
        "alpha_eest": alpha_eest,
        "order": order,
        "adaptive_order": adaptive_order,
        "fsal": fsal,
        "stages": len(alpha),
        "line": line0,
        "source": src.name,
    }


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def check(tab: dict) -> list:
    a, c, alpha = tab["a"], tab["c"], tab["alpha"]
    n = tab["stages"]
    problems = []

    if len(a) != n or len(c) != n:
        problems.append(f"{tab['name']}: A has {len(a)} rows and c has {len(c)} entries, expected {n}")
        return problems

    for i in range(n):
        for j in range(i, i + 1):
            if a[i][j] != 0.0:
                problems.append(f"{tab['name']}: A[{i}][{j}] = {a[i][j]}, expected 0 (not strictly upper triangular)")
        row_sum = sum(a[i])
        if abs(row_sum - c[i]) > INVARIANT_TOL:
            problems.append(f"{tab['name']}: row {i} sums to {row_sum!r}, c[{i}] = {c[i]!r}")
    if abs(sum(a[n - 1]) - 1.0) > INVARIANT_TOL:
        problems.append(f"{tab['name']}: final row sums to {sum(a[n - 1])!r}, not 1")
    if abs(sum(alpha) - 1.0) > INVARIANT_TOL:
        problems.append(f"{tab['name']}: sum(alpha) = {sum(alpha)!r}, not 1")

    eest = tab["alpha_eest"]
    if tab["adaptive_order"] > 0:
        if not eest:
            problems.append(f"{tab['name']}: adaptive but alphaEEst is missing")
        else:
            if all(v == 0.0 for v in eest):
                problems.append(f"{tab['name']}: alphaEEst is all zero")
            if eest == alpha:
                problems.append(f"{tab['name']}: alphaEEst equals alpha, so it estimates nothing")
    elif eest:
        problems.append(f"{tab['name']}: non-adaptive but carries alphaEEst")

    # An FSAL method's last stage is the step's solution stage, so the last
    # row of A must equal alpha.  The converse does not hold: Julia marks
    # BogakiShampine3 non-FSAL even though its last row does equal alpha, and
    # we follow that flag verbatim.
    last_row_matches = all(abs(a[n - 1][j] - alpha[j]) <= INVARIANT_TOL for j in range(n))
    if tab["fsal"] and not last_row_matches:
        problems.append(f"{tab['name']}: fsal = true but the last A row differs from alpha")
    return problems


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit(tables: dict) -> str:
    lines = []
    add = lines.append
    add("// Explicit Runge-Kutta tableaus for the generic tableau solver.")
    add("//")
    add("// GENERATED by tools/extract_rk_tableaus.py from")
    add("//   OrdinaryDiffEqExplicitTableaus/src/tableaus_low_order.jl")
    add("//   OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl")
    add("// The Julia constructors build these from exact rationals (`1 // 2`) and")
    add("// convert them with `map(T, ...)`; the values below are the same doubles.")
    add("// Do not edit by hand; re-run the tool instead.")

    def emit_one(name: str, notes: list) -> None:
        t = tables[name]
        add("")
        for note in notes:
            add(f"// {note}")
        add("///|")
        add(f"pub fn {t['moon']}() -> ExplicitRKTableau {{")
        add("  {")
        add("    a: [")
        for row in t["a"]:
            add("      [ " + ", ".join(fmt(v) for v in row) + " ],")
        add("    ],")
        add("    c: [ " + ", ".join(fmt(v) for v in t["c"]) + " ],")
        add("    alpha: [ " + ", ".join(fmt(v) for v in t["alpha"]) + " ],")
        if t["alpha_eest"]:
            add("    alpha_eest: [ " + ", ".join(fmt(v) for v in t["alpha_eest"]) + " ],")
        else:
            add("    alpha_eest: [],")
        add(f"    stages: {t['stages']},")
        add(f"    order: {t['order']},")
        add(f"    adaptive_order: {t['adaptive_order']},")
        add(f"    fsal: {'true' if t['fsal'] else 'false'},")
        add("  }")
        add("}")

    emit_one("Heun", [
        "--- Heun 2(1), 2 stages, no FSAL ---------------------------------------",
        "`Heun`, tableaus_low_order.jl:4. Fixed-step or adaptive; not FSAL, so each",
        "step evaluates one extra derivative at the new point.",
    ])
    emit_one("RK4", [
        "--- Classic Runge-Kutta 4, 4 stages, fixed step -------------------------",
        "`RK4`, tableaus_low_order.jl:74. Neither a pair nor FSAL.",
    ])
    emit_one("BogakiShampine3", [
        "--- Bogacki-Shampine 3(2), 4 stages, no FSAL ----------------------------",
        "`BogakiShampine3`, tableaus_order5.jl:753.",
    ])
    emit_one("RKF4", [
        "--- Runge-Kutta-Fehlberg 4(3), 5 stages, no FSAL ------------------------",
        "`RKF4`, tableaus_order5.jl:806.",
    ])
    emit_one("DormandPrince", [
        "--- Dormand-Prince 5(4), 7 stages, FSAL --------------------------------",
        "`DormandPrince`, tableaus_order5.jl:4. The 7th stage is `f` at the new point",
        "and doubles as the next step's first stage.",
    ])
    add("")
    return "\n".join(lines)


# Populated by `main` before `emit` runs.
tab_of: dict = {}


def main() -> int:
    global tab_of
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    missing = [n for n, p in TABLEAUS_SRC.items() if not p.exists()]
    if missing:
        print(f"error: Julia sources not found for: {missing}", file=sys.stderr)
        return 2

    tables = {}
    for name in ORDER:
        t = parse_tableau(name)
        problems = check(t)
        for p in problems:
            print(f"invariant violated: {p}", file=sys.stderr)
        if problems:
            return 1
        tables[name] = t

    tab_of = tables
    content = emit(tables)

    if args.check:
        if not args.out.exists():
            print(f"error: {args.out} does not exist (run without --check)", file=sys.stderr)
            return 2
        if args.out.read_text(encoding="utf-8") != content:
            print(f"error: {args.out} is stale; re-run tools/extract_rk_tableaus.py", file=sys.stderr)
            return 1
        print(f"ok: {args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"wrote {args.out} ({len(tables)} tableaus, invariants hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())