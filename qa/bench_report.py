#!/usr/bin/env python3
"""MoonBit benchmark reporter + sanity gate for CI.

MoonBit prints benchmark results like::

    [riantr/ordinarydiffeq] bench bench_test.mbt:55 ("bench: tsit5 decay") ok
    time (mean ± σ)         range (min … max)
     471.14 ns ±  49.29 ns   408.78 ns … 578.35 ns  in 10 × 100000 runs

This script:
1. runs ``moon bench --target native --release`` (or reads pre-captured
   output via ``--input``),
2. parses each benchmark's mean time and unit, normalizing to nanoseconds,
3. writes a Markdown summary to ``$GITHUB_STEP_SUMMARY`` (visible in PRs)
   and a JSON artifact (for manual trend tracking across runs),
4. exits non-zero when a *sanity* budget is violated.

The budget is intentionally generous (default 50 ms/bench): its job is to
catch catastrophic regressions (e.g. a 100x slowdown, an accidental
``O(n^2)`` blow-up, or a near-hang), NOT to act as a precision regression
gate. Absolute timings vary across runners, so a tight machine-specific
threshold would cause false failures. Real regression trending is done
out-of-band by downloading the JSON artifact and comparing runs.

Pure stdlib; no third-party dependencies.
"""
import re
import os
import sys
import json
import glob
import argparse
import subprocess

# [riantr/ordinarydiffeq] bench bench_test.mbt:55 ("bench: tsit5 decay") ok
BENCH_LINE_RE = re.compile(
    r'bench\s+\S+?:(\d+)\s+\("([^"]+)"\)\s+ok'
)
#   471.14 ns ±  49.29 ns   408.78 ns … 578.35 ns  in 10 × 100000 runs
TIME_LINE_RE = re.compile(
    r'^\s*([\d.]+)\s*(ns|µs|us|ms|s)\s*±'
)

UNIT_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "µs": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}


def parse_bench_output(text):
    """Return a list of dicts: {name, line, mean_ns, unit, raw}."""
    results = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = BENCH_LINE_RE.search(line)
        if not m:
            continue
        line_no = int(m.group(1))
        name = m.group(2)
        # the following line(s) hold the time; find the next TIME_LINE match
        for j in range(i + 1, min(i + 4, len(lines))):
            tm = TIME_LINE_RE.search(lines[j])
            if tm:
                val = float(tm.group(1))
                unit = tm.group(2)
                results.append({
                    "name": name,
                    "line": line_no,
                    "mean_ns": val * UNIT_TO_NS[unit],
                    "unit": unit,
                    "raw": lines[j].strip(),
                })
                break
    return results


def run_bench(bench_dir, bench_cmd):
    out = subprocess.run(bench_cmd, cwd=bench_dir,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        print(out.stdout)
        print(f"ERROR: `{' '.join(bench_cmd)}` exited {out.returncode}")
        sys.exit(2)
    return out.stdout + "\n" + out.stderr


def fmt_ns(ns):
    if ns >= 1_000_000.0:
        return f"{ns / 1_000_000.0:.3f} ms"
    if ns >= 1_000.0:
        return f"{ns / 1_000.0:.3f} µs"
    return f"{ns:.2f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="src/src",
                    help="dir to run moon bench from (default src/src)")
    ap.add_argument("--bench-cmd", default=None,
                    help="override command (default "
                         "'moon bench --target native --release')")
    ap.add_argument("--input", default=None,
                    help="read pre-captured bench output from file "
                         "instead of running moon bench")
    ap.add_argument("--json-out", default="qa/bench_results.json")
    ap.add_argument("--budget-ms", type=float, default=50.0,
                    help="per-bench sanity ceiling in ms (default 50)")
    args = ap.parse_args()

    budget_ns = args.budget_ms * 1_000_000.0

    if args.input:
        text = open(args.input, encoding="utf-8", errors="replace").read()
    else:
        cmd = args.bench_cmd.split() if args.bench_cmd \
            else ["moon", "bench", "--target", "native", "--release"]
        text = run_bench(args.bench_dir, cmd)

    results = parse_bench_output(text)
    if not results:
        print("ERROR: no benchmark results parsed from output")
        sys.exit(2)

    results.sort(key=lambda r: r["mean_ns"], reverse=True)

    lines = [
        "## MoonBit Benchmarks (native, release)",
        "",
        f"- **Benchmarks: {len(results)}** (sanity budget: "
        f"{args.budget_ms:.0f} ms/bench)",
        "",
        "| Benchmark | Mean | Raw |",
        "|---|---|---|",
    ]
    failures = []
    for r in results:
        flag = ""
        if r["mean_ns"] > budget_ns:
            failures.append(
                f"{r['name']}: {fmt_ns(r['mean_ns'])} exceeds "
                f"{args.budget_ms:.0f} ms budget")
            flag = " ⛔"
        lines.append(
            f"| {r['name']} | {fmt_ns(r['mean_ns'])} | `{r['raw']}` |{flag}|")
    text_md = "\n".join(lines)

    gs = os.environ.get("GITHUB_STEP_SUMMARY")
    if gs:
        with open(gs, "a", encoding="utf-8") as fh:
            fh.write(text_md + "\n")
    print(text_md)

    payload = {
        "target": "native",
        "mode": "release",
        "budget_ms": args.budget_ms,
        "benchmarks": results,
    }
    try:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nJSON results -> {args.json_out}")
    except OSError as e:
        print(f"\n(warning) could not write JSON to {args.json_out}: {e}")

    if failures:
        print("\nBENCHMARK SANITY GATE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nBENCHMARK SANITY GATE PASSED ✅")


if __name__ == "__main__":
    main()
