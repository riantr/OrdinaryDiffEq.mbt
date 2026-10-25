#!/usr/bin/env python3
"""Quality-metrics gate for the MoonBit ODE port.

Wraps `qa_analysis.analyze()` and gates the codebase on a handful of
cheap-to-compute quality metrics, so complexity hotspots, unchecked array
access, duplication creep and architectural invariants cannot silently regress
between CI runs. Replaces the one-off manual measurement in QA_Deep_Report.md.

Gated metrics (defaults calibrated on the 2026-10-25 baseline + headroom):

  * max decision points in a single source file   (baseline 102, cap 120)
  * total unsafe_get/unsafe_set call sites        (baseline 1051, cap 1600)
  * duplicate-code rate (%)                       (baseline 10.9, cap 20.0)
  * number of `test` blocks                       (baseline 98, floor 90)
  * architectural invariants, all must be 0:
    extern, panic, abort, raise/error

Nothing here is a taste judgement: the caps are set well above today's values
so ordinary work passes, and they only trip on step changes (a new 1000-line
solver module, a copy-paste explosion, deleted tests, or the introduction of
an FFI boundary / panic into a library that deliberately has none).

Usage:
    python3 qa/metrics_gate.py [--json-out qa/metrics.json]
Exit codes: 0 = pass, 1 = gate violation, 2 = analysis failed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qa_analysis import analyze
except Exception as exc:  # pragma: no cover - import guard
    print(f"metrics_gate: cannot import qa_analysis: {exc}", file=sys.stderr)
    sys.exit(2)


def build_summary(rows):
    lines = ["## Quality metrics gate", ""]
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['value']} | {r['threshold']} | {r['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Quality metrics gate")
    ap.add_argument("--max-decisions", type=int, default=120,
                    help="max decision points in one source file (default 120)")
    ap.add_argument("--max-unsafe", type=int, default=1600,
                    help="max unsafe_get/unsafe_set sites (default 1600)")
    ap.add_argument("--max-dup-rate", type=float, default=20.0,
                    help="max duplicate-code rate %%(default 20.0)")
    ap.add_argument("--min-test-blocks", type=int, default=90,
                    help="min number of test blocks (default 90)")
    ap.add_argument("--max-extern", type=int, default=0,
                    help="max extern declarations (default 0)")
    ap.add_argument("--max-panic", type=int, default=0,
                    help="max panic( call sites (default 0)")
    ap.add_argument("--max-abort", type=int, default=0,
                    help="max abort( call sites (default 0)")
    ap.add_argument("--max-raise", type=int, default=0,
                    help="max raise(/error( call sites (default 0)")
    ap.add_argument("--json-out", default=None,
                    help="write the computed metrics as JSON here")
    args = ap.parse_args()

    try:
        data = analyze()
    except Exception as exc:
        print(f"metrics_gate: analysis failed: {exc}", file=sys.stderr)
        sys.exit(2)

    metrics = data["metrics"]
    sast = data["sast"]["counts"]
    dup = data["duplicates"]
    blocks = data["tests"]["total_test_blocks"]

    hotspots = sorted(metrics, key=lambda r: -r["decisions"])
    worst = hotspots[0] if hotspots else None
    max_decisions = worst["decisions"] if worst else 0
    max_decisions_file = worst["file"] if worst else "-"
    unsafe = sast.get("unsafe_get/set", 0)
    extern = sast.get("extern", 0)
    panic = sast.get("panic", 0)
    abort = sast.get("abort", 0)
    raise_ = sast.get("raise/error", 0)
    dup_rate = dup.get("rate_pct", 0.0)

    rows = []
    violations = []

    def check(metric, value, limit, comparator, threshold_text, display=None):
        ok = value <= limit if comparator == "le" else value >= limit
        rows.append({"metric": metric,
                     "value": value if display is None else display,
                     "threshold": threshold_text, "status": "PASS" if ok else "FAIL"})
        if not ok:
            show = value if display is None else display
            violations.append(f"{metric}: {show} {threshold_text}")

    check("Max decision points / file", max_decisions, args.max_decisions, "le",
          f"<= {args.max_decisions}", display=f"{max_decisions} ({max_decisions_file})")
    check("unsafe_get / unsafe_set sites", unsafe, args.max_unsafe, "le",
          f"<= {args.max_unsafe}")
    check("Duplicate-code rate", dup_rate, args.max_dup_rate, "le",
          f"<= {args.max_dup_rate}%", display=f"{dup_rate}%")
    check("Test blocks", blocks, args.min_test_blocks, "ge",
          f">= {args.min_test_blocks}")
    for name, val, limit in (("extern declarations", extern, args.max_extern),
                             ("panic() sites", panic, args.max_panic),
                             ("abort() sites", abort, args.max_abort),
                             ("raise()/error() sites", raise_, args.max_raise)):
        check(name, val, limit, "le", f"<= {limit}")

    summary = build_summary(rows)
    # hotspot table: actionable context even when the gate passes
    top = ["", "### Complexity hotspots", "",
           "| File | Decisions | Code lines |", "|---|---|---|"]
    for r in hotspots[:5]:
        top.append(f"| {r['file']} | {r['decisions']} | {r['code_lines']} |")
    summary += "\n".join(top) + "\n"

    print(summary)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as fh:
                fh.write(summary + "\n")
        except OSError as exc:
            print(f"metrics_gate: could not write step summary: {exc}",
                  file=sys.stderr)

    if args.json_out:
        payload = {
            "max_decisions": max_decisions,
            "max_decisions_file": max_decisions_file,
            "unsafe_get_set": unsafe,
            "extern": extern,
            "panic": panic,
            "abort": abort,
            "raise_error": raise_,
            "duplication_rate_pct": dup_rate,
            "test_blocks": blocks,
            "code_lines": sum(r["code_lines"] for r in metrics),
            "src_files": len(metrics),
            "violations": violations,
        }
        try:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"\nJSON metrics -> {args.json_out}")
        except OSError as exc:
            print(f"metrics_gate: could not write JSON: {exc}", file=sys.stderr)

    if violations:
        print("\nQuality metrics gate FAILED:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)

    print("\nQuality metrics gate PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
