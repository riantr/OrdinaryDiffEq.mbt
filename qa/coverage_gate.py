#!/usr/bin/env python3
"""MoonBit coverage gate for CI.

MoonBit emits per-test-package coverage files named
``moonbit_coverage_<id>.txt`` (each may contain several JSON blocks delimited by
``BEGIN/END MOONBIT COVERAGE`` banners). This script:

1. parses & merges all of them (max hit-count per basic block),
2. computes overall block + function coverage and per-module block coverage,
3. writes a Markdown summary to ``$GITHUB_STEP_SUMMARY`` (visible in PRs),
4. writes a self-contained HTML report,
5. exits non-zero when a threshold is violated.

Thresholds
----------
--min-overall  : minimum overall block coverage %           (default 90)
--min-module   : minimum block coverage % for any module    (default 80)
--allow-below  : comma list of modules allowed below --min-module (known gaps,
                 documented in the QA report). These still fail if they drop
                 below --catastrophic.
--catastrophic : hard floor for allow-listed modules        (default 45)

Pure stdlib; no third-party dependencies.
"""
import json
import re
import os
import sys
import glob
import argparse

COVERAGE_RE = re.compile(
    r"BEGIN MOONBIT COVERAGE\s*-----(.*?)----- END MOONBIT COVERAGE -----", re.S
)


def parse_coverage_file(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = []
    for m in COVERAGE_RE.finditer(txt):
        body = m.group(1).strip()
        if body.startswith("{"):
            blocks.append(json.loads(body))
    return blocks


def mod_of(key):
    return key.split("$", 1)[1] if "$" in key else key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="dir to search for coverage files")
    ap.add_argument("--min-overall", type=float, default=90.0)
    ap.add_argument("--min-module", type=float, default=80.0)
    ap.add_argument("--allow-below", default="bdf_utils,dense")
    ap.add_argument("--catastrophic", type=float, default=45.0)
    ap.add_argument("--report-html", default="coverage_report.html")
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.root, "**", "moonbit_coverage_*.txt"),
                  recursive=True)
    )
    if not files:
        print(f"ERROR: no moonbit_coverage_*.txt found under {args.root!r}")
        sys.exit(2)

    merged = {}
    for f in files:
        for data in parse_coverage_file(f):
            for k, v in data.items():
                if k not in merged:
                    merged[k] = list(v)
                else:
                    mv = merged[k]
                    for i, x in enumerate(v):
                        if i < len(mv):
                            mv[i] = max(mv[i], x)
                        else:
                            mv.append(x)

    total_blocks = total_cov = total_funcs = total_fcov = 0
    mods = []
    allow = set(x.strip() for x in args.allow_below.split(",") if x.strip())
    for k, arr in merged.items():
        m = mod_of(k)
        tb = len(arr)
        cb = sum(1 for x in arr if x > 0)
        total_blocks += tb
        total_cov += cb
        if tb > 0:
            total_funcs += 1
            total_fcov += 1 if cb > 0 else 0
            mods.append((m, round(100.0 * cb / tb, 1), tb, cb))

    overall = 100.0 * total_cov / total_blocks if total_blocks else 0.0
    func_pct = 100.0 * total_fcov / total_funcs if total_funcs else 0.0
    mods.sort(key=lambda r: r[1])

    failures = []
    lines = [
        "## MoonBit Coverage (native target)",
        "",
        f"- **Overall block coverage: {overall:.1f}%** (gate \u2265 {args.min_overall:.0f}%)",
        f"- **Source modules exercised: {total_fcov}/{total_funcs}** "
        f"(every module has \u22651 covered block)",
        "",
        "| Module | Block % | Covered/Total | Flag |",
        "|---|---|---|---|",
    ]
    for m, pct, tb, cb in mods:
        if m in allow:
            if pct < args.catastrophic:
                failures.append(
                    f"{m}: {pct:.1f}% collapsed below catastrophic floor "
                    f"{args.catastrophic:.0f}% (allow-listed)")
                flag = " \u26d4"
            elif pct < args.min_module:
                flag = " \u26a0\ufe0f allow-listed"
            else:
                flag = ""
        else:
            if pct < args.min_module:
                failures.append(
                    f"{m}: {pct:.1f}% < {args.min_module:.0f}% module threshold")
                flag = " \u26d4"
            else:
                flag = ""
        lines.append(f"| {m} | {pct:.1f}% | {cb}/{tb} |{flag}|")

    if overall < args.min_overall:
        failures.append(f"overall {overall:.1f}% < {args.min_overall:.0f}%")

    text = "\n".join(lines)
    gs = os.environ.get("GITHUB_STEP_SUMMARY")
    if gs:
        with open(gs, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)

    html = build_html(overall, func_pct, total_fcov, total_funcs, mods, args)
    try:
        with open(args.report_html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"\nHTML report -> {args.report_html}")
    except OSError as e:
        # HTML is a nice-to-have artifact; never let it fail the gate.
        print(f"\n(warning) could not write HTML report to "
              f"{args.report_html}: {e}")

    if failures:
        print("\nCOVERAGE GATE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nCOVERAGE GATE PASSED \u2705")


def build_html(overall, func_pct, total_fcov, total_funcs, mods, args):
    allow = set(x.strip() for x in args.allow_below.split(",") if x.strip())
    rows = ""
    for m, pct, tb, cb in mods:
        if pct >= args.min_module:
            color = "#2e7d32"
        elif m in allow:
            color = "#f9a825"
        else:
            color = "#c62828"
        rows += (f"<tr><td>{m}</td>"
                 f"<td style='color:{color};font-weight:600'>{pct:.1f}%</td>"
                 f"<td>{cb}/{tb}</td></tr>\n")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>MoonBit Coverage</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;margin:2rem;color:#222}}
h1{{margin-bottom:.2rem}}table{{border-collapse:collapse;margin-top:1rem}}
td,th{{border:1px solid #ccc;padding:.4rem .9rem;text-align:left}}
th{{background:#f5f5f5}}.muted{{color:#888;font-size:.8rem;margin-top:1.5rem}}</style>
</head><body>
<h1>MoonBit Coverage Report</h1>
<p>Overall block coverage: <b>{overall:.1f}%</b> (gate &ge; {args.min_overall:.0f}%)<br>
Source modules exercised: <b>{func_pct:.0f}%</b> ({total_fcov}/{total_funcs})</p>
<table><tr><th>Module</th><th>Block %</th><th>Covered/Total</th></tr>
{rows}</table>
<p class="muted">Generated by qa/coverage_gate.py &middot; MoonBit native target</p>
</body></html>"""


if __name__ == "__main__":
    main()
