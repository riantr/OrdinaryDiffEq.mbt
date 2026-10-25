#!/usr/bin/env python3
"""Extract explicit Runge-Kutta tableaus from the OrdinaryDiffEq.jl monorepo
into MoonBit ``const`` declarations.

Currently targets Tsit5 (Tsitouras 5(4): 7 stages, FSAL, 4th-order "free"
interpolant) from::

    OrdinaryDiffEq.jl/lib/OrdinaryDiffEqTsit5/src/tsit_tableaus.jl

For every tableau the Julia file defines two ``@fold`` methods: one constrained
to ``CompiledFloats`` (Float32/Float64) and one generic with high-precision
``big"..."`` literals. At Float64 runtime Julia dispatches to the
``CompiledFloats`` method, so that is the variant extracted here; the two
variants agree to within 1 ulp.

The extracted coefficients are checked against tableau invariants before the
file is written:

  * each ``a`` row sums to its stage abscissa ``c`` (row-sum condition),
  * the final ``a`` row sums to 1 (the method integrates constants exactly),
  * the interpolant weights satisfy ``b_i(1) = a_{7,i}`` and ``b_7(1) = 0``,
    so the polynomial reproduces the step endpoint,
  * ``b_1'(0) = 1`` and ``b_i'(0) = 0`` for ``i >= 2``, so the interpolant
    derivative at the left endpoint is the first stage.

Usage::

    python tools/extract_tableaus.py [--julia-src PATH] [--out PATH] [--check]

``--check`` re-extracts and diffs against the existing file (CI-friendly);
it never writes.
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JULIA_SRC = REPO_ROOT.parent / "OrdinaryDiffEq.jl" / "lib" / "OrdinaryDiffEqTsit5" / "src" / "tsit_tableaus.jl"
DEFAULT_OUT = REPO_ROOT / "src" / "src" / "alg_tsit5_coeffs.mbt"

# `name = convert(T, value)` — also matches `convert(T2, ...)`. Commented-out
# assignments (the unused 5th-order `b` weights) start with `#` and never match.
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*convert\(\s*T2?\s*,\s*(.+?)\s*\)\s*$")
RATIONAL_RE = re.compile(r"^(-?\d+)\s*//\s*(-?\d+)$")

# Tolerance for the invariant checks. The Julia literals are independently
# rounded, so invariants hold to rounding error rather than exactly.
INVARIANT_TOL = 1.0e-9


def split_fold_functions(text: str):
    """Yield ``(name, signature, body, body_start_line)`` per ``@fold function``.

    The signature runs from the ``@fold function`` keyword to the first
    assignment (or the terminating ``end``); the body runs to the first line
    that is exactly ``end`` at column 0. ``body_start_line`` is the 1-based
    line number of the body's first line, used for provenance comments.
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].startswith("@fold function"):
            i += 1
            continue
        j = i + 1
        while j < n and lines[j] != "end" and not ASSIGN_RE.match(lines[j]):
            j += 1
        signature = "\n".join(lines[i:j])
        k = j
        while k < n and lines[k] != "end":
            k += 1
        body = "\n".join(lines[j:k])
        name = lines[i].split("function", 1)[1].split("(", 1)[0].strip()
        yield name, signature, body, j + 1  # 1-based line of the body start
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
        if rm:
            value = int(rm.group(1)) / int(rm.group(2))
        else:
            value = float(raw)
        out[name] = (value, body_start_line + offset)
    return out


def pick_variant(functions: dict, want: str) -> dict:
    """Select the ``CompiledFloats`` (Float64) method for tableau ``want``.

    ``functions`` maps a name to the list of its ``(signature, body,
    start_line)`` payloads — the Julia file defines two ``@fold`` methods per
    tableau, and they share a name.
    """
    for _sig, body, start in functions.get(want, []):
        if "CompiledFloats" in _sig:
            return parse_assignments(body, start)
    raise SystemExit(f"error: no CompiledFloats method found for {want}")


def fmt(value: float) -> str:
    """Shortest round-trip decimal, never in exponent notation."""
    return format(Decimal(repr(value)), "f")


def moonbit_name(julia_name: str) -> str:
    """`a21` -> `Tsit5A21` (MoonBit constants are upper-camel values)."""
    return "Tsit5" + julia_name[0].upper() + julia_name[1:]


# ---------------------------------------------------------------------------
# Tableau invariants
# ---------------------------------------------------------------------------

def stage_rows(a: dict):
    """The six ``a`` rows as lists, indexed by stage 2..7."""
    return [
        ["a21"],
        ["a31", "a32"],
        ["a41", "a42", "a43"],
        ["a51", "a52", "a53", "a54"],
        ["a61", "a62", "a63", "a64", "a65"],
        ["a71", "a72", "a73", "a74", "a75", "a76"],
    ]


def b_at_one(r: dict, stage: int) -> float:
    """Interpolant weight b_i(1)."""
    if stage == 1:
        return r["r11"] + r["r12"] + r["r13"] + r["r14"]
    i = stage
    return r[f"r{i}2"] + r[f"r{i}3"] + r[f"r{i}4"]


def b_prime_at_zero(r: dict, stage: int) -> float:
    """Interpolant weight derivative b_i'(0)."""
    if stage == 1:
        return r["r11"]
    return 0.0  # b_i has an explicit Θ² factor for i >= 2


def check_invariants(coef: dict) -> list:
    """Return a list of human-readable failures (empty when all hold)."""
    def val(group: str, name: str) -> float:
        return coef[group][name][0]

    problems = []
    rows = stage_rows(coef["a"])
    for idx, row in enumerate(rows):
        stage = idx + 2
        abscissa = val("c", f"c{stage - 1}")
        total = sum(val("a", name) for name in row)
        if abs(total - abscissa) > INVARIANT_TOL:
            problems.append(
                f"row sum a{stage}* = {total!r} != c{stage - 1} = {abscissa!r}"
            )
    last_row_total = sum(val("a", name) for name in rows[-1])
    if abs(last_row_total - 1.0) > INVARIANT_TOL:
        problems.append(f"final row sums to {last_row_total!r}, not 1")
    r = {k: v[0] for k, v in coef["r"].items()}
    a7 = [val("a", name) for name in rows[-1]] + [0.0]  # b_7(1) = 0
    for stage in range(1, 8):
        got, want = b_at_one(r, stage), a7[stage - 1]
        if abs(got - want) > INVARIANT_TOL:
            problems.append(f"b_{stage}(1) = {got!r} != a7{stage} = {want!r}")
    if abs(r["r11"] - 1.0) > INVARIANT_TOL:
        problems.append(f"r11 = {r['r11']!r}, expected 1")
    for stage in range(2, 8):
        if abs(b_prime_at_zero(r, stage)) > INVARIANT_TOL:
            problems.append(f"b_{stage}'(0) != 0")
    return problems


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit(coef: dict, src: Path) -> str:
    """Render the MoonBit constants in `moon fmt`'s canonical shape (doc
    marker before every `const`, blank line between items), so that
    regenerating the file is idempotent under `moon fmt`."""
    lines = []
    add = lines.append
    add("// Tsit5 tableau and dense-interpolant coefficients.")
    add("//")
    add("// GENERATED by tools/extract_tableaus.py from")
    add(f"//   {src.name}")
    add("// (the `CompiledFloats` methods, i.e. the Float64 dispatch Julia uses")
    add("// at runtime; the `big\"...\"` generic methods carry the same values to")
    add("// higher precision). Do not edit by hand; re-run the tool instead.")

    def emit_const(name: str, value_line: tuple) -> None:
        value, line = value_line
        add("")
        add("///|")
        add(f"const {moonbit_name(name)} : Double = {fmt(value)} // tsit_tableaus.jl:{line}")

    add("")
    add("// --- Stage abscissae c ---------------------------------------------")
    for i in range(1, 7):
        emit_const(f"c{i}", coef["c"][f"c{i}"])
    add("")
    add("// --- Stage coefficients a (rows indexed by stage) -------------------")
    for idx, row in enumerate(stage_rows(coef["a"])):
        add(f"// stage {idx + 2}")
        for name in row:
            emit_const(name, coef["a"][name])
    add("")
    add("// --- Embedded 4th-order weights btilde (error estimate, M2) ---------")
    for i in range(1, 8):
        emit_const(f"btilde{i}", coef["btilde"][f"btilde{i}"])
    add("")
    add("// --- Dense interpolant weights r (b_i(theta) polynomials) ----------")
    add("// b_1(theta) = theta*(r11 + r12*theta + r13*theta^2 + r14*theta^3);")
    add("// b_i(theta) = theta^2*(ri2 + ri3*theta + ri4*theta^2) for i = 2..7.")
    for name, payload in coef["r"].items():
        emit_const(name, payload)
    add("")
    return "\n".join(lines)


def extract(src: Path) -> dict:
    text = src.read_text(encoding="utf-8")
    functions = {}
    for name, sig, body, start in split_fold_functions(text):
        functions.setdefault(name, []).append((sig, body, start))
    cache = pick_variant(functions, "Tsit5ConstantCacheActual")
    interp = pick_variant(functions, "Tsit5Interp")
    coef = {
        "c": {k: v for k, v in cache.items() if re.fullmatch(r"c[1-6]", k)},
        "a": {k: v for k, v in cache.items() if re.fullmatch(r"a[1-7][1-7]", k)},
        "btilde": {k: v for k, v in cache.items() if k.startswith("btilde")},
        "r": {k: v for k, v in interp.items() if re.fullmatch(r"r[1-7][1-4]", k)},
    }
    expected = {
        "c": [f"c{i}" for i in range(1, 7)],
        "a": [f"a{s}{j}" for s, cols in enumerate(
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
            start=2) for j in cols],
        "btilde": [f"btilde{i}" for i in range(1, 8)],
        "r": ([f"r1{j}" for j in range(1, 5)] +
              [f"r{i}{j}" for i in range(2, 8) for j in range(2, 5)]),
    }
    for group, names in expected.items():
        missing = [n for n in names if n not in coef[group]]
        if missing:
            raise SystemExit(f"error: missing coefficients in {group}: {missing}")
    return coef


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--julia-src", type=Path, default=DEFAULT_JULIA_SRC,
                    help="path to tsit_tableaus.jl in the Julia monorepo")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="MoonBit file to generate")
    ap.add_argument("--check", action="store_true",
                    help="diff against the existing file instead of writing")
    args = ap.parse_args()

    if not args.julia_src.exists():
        print(f"error: Julia source not found: {args.julia_src}", file=sys.stderr)
        return 2

    coef = extract(args.julia_src)
    problems = check_invariants(coef)
    for p in problems:
        print(f"invariant violated: {p}", file=sys.stderr)
    if problems:
        return 1

    content = emit(coef, args.julia_src)
    if args.check:
        if not args.out.exists():
            print(f"error: {args.out} does not exist (run without --check)", file=sys.stderr)
            return 2
        current = args.out.read_text(encoding="utf-8")
        if current != content:
            print(f"error: {args.out} is stale; re-run tools/extract_tableaus.py", file=sys.stderr)
            return 1
        print(f"ok: {args.out} matches {args.julia_src.name}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    n = sum(len(g) for g in coef.values())
    print(f"wrote {args.out} ({n} coefficients, invariants hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
