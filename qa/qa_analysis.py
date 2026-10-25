#!/usr/bin/env python3
"""Deep QA analysis for the SciML MoonBit port.

Importable module + CLI. `analyze()` returns a dict with:
  - coverage (merged across test packages, per module + overall)
  - test inventory
  - SAST signal scan (comments stripped)
  - LOC / complexity metrics (comments stripped)
  - duplicate-code detection (shingling), plus global duplication rate

Run as a script to print the whole blob as JSON:

    python3 qa/qa_analysis.py
"""
import json, re, os, glob, sys, hashlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))   # .../OrdinaryDiffEq.mbt/qa
PORT_REPO = os.path.dirname(HERE)                    # .../OrdinaryDiffEq.mbt
ROOT = os.path.dirname(PORT_REPO)                    # .../SciML (workspace)
PORT = os.path.join(PORT_REPO, "src", "src")

SHINGLE = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _comment_start(line):
    """Index where a line comment begins, ignoring `//` inside string literals.

    Without this the SAST scan counts words like "error" appearing in `///`
    doc comments as real calls (control.mbt:114 was exactly such a false
    positive), overstating the signal.
    """
    i = 0
    in_str = False
    n = len(line)
    while i < n - 1:
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
        elif c == "/" and line[i + 1] == "/" and not in_str:
            return i
        i += 1
    return None


def strip_comments(code):
    """Drop `//` line comments so metrics measure real code, not prose."""
    out = []
    for line in code.splitlines():
        cut = _comment_start(line)
        if cut is not None:
            line = line[:cut]
        out.append(line)
    return "\n".join(out)


def parse_coverage_file(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    # file may contain multiple JSON blocks delimited by BEGIN/END banners
    blocks = []
    for m in re.finditer(
        r"BEGIN MOONBIT COVERAGE\s*-----(.*?)----- END MOONBIT COVERAGE -----",
        txt, re.S):
        body = m.group(1).strip()
        if body.startswith("{"):
            blocks.append(json.loads(body))
    return blocks


def module_of(key):
    # key like riantr/ordinarydiffeq$alg_tsit5  -> module name after $
    return key.split("$", 1)[1] if "$" in key else key


def complexity_proxy(code):
    # crude: count branching/loop keywords as decision points
    return len(re.findall(r"\b(if|match|while|for|loop|guard|try|else)\b", code))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze():
    # ---- 1. coverage ----
    cov_files = glob.glob(os.path.join(PORT_REPO, "src", "_build",
                                       "moonbit_coverage_*.txt"))
    merged = {}
    for f in cov_files:
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

    mod_stats = defaultdict(lambda: [0, 0, 0, 0])  # blocks, cov, funcs, funcs_cov
    total_blocks = covered_blocks = func_total = func_covered = 0
    for key, arr in merged.items():
        mod = module_of(key)
        tb = len(arr)
        cb = sum(1 for x in arr if x > 0)
        total_blocks += tb
        covered_blocks += cb
        func_total += 1
        fc = 1 if cb > 0 else 0
        func_covered += fc
        s = mod_stats[mod]
        s[0] += tb; s[1] += cb; s[2] += 1; s[3] += fc

    coverage = {
        "overall_block_pct": round(100.0 * covered_blocks / total_blocks, 1) if total_blocks else None,
        "overall_func_pct": round(100.0 * func_covered / func_total, 1) if func_total else None,
        "total_blocks": total_blocks,
        "covered_blocks": covered_blocks,
        "total_funcs": func_total,
        "covered_funcs": func_covered,
        "modules": sorted([
            {"module": m, "block_pct": round(100.0 * s[1] / s[0], 1) if s[0] else None,
             "blocks": s[0], "covered_blocks": s[1],
             "func_pct": round(100.0 * s[3] / s[2], 1) if s[2] else None,
             "funcs": s[2], "covered_funcs": s[3]}
            for m, s in mod_stats.items()
        ], key=lambda r: (r["block_pct"] if r["block_pct"] is not None else 999)),
    }

    # ---- 2. test inventory ----
    test_files = sorted(glob.glob(os.path.join(PORT, "*_test.mbt")))
    test_info = []
    for tf in test_files:
        src = open(tf, encoding="utf-8", errors="replace").read()
        n_blocks = len(re.findall(r"\btest\s+(?:\"[^\"]*\"\s*)?\{", src))
        test_info.append({"file": os.path.basename(tf), "test_blocks": n_blocks,
                          "loc": src.count("\n") + 1})

    # ---- 3. SAST signal scan (non-test sources, comments stripped) ----
    src_files = sorted([f for f in glob.glob(os.path.join(PORT, "*.mbt"))
                        if not f.endswith("_test.mbt")])
    sast_patterns = {
        "unsafe": r"\bunsafe\b",
        "unsafe_get/set": r"\.unsafe_get\(|\.unsafe_set\(",
        "extern": r"\bextern\b",
        "panic": r"\bpanic\s*\(",
        "abort": r"\babort\s*!?\s*\(",
        "raise/error": r"\braise\s*\(|\berror\s*\(",
        "TODO/FIXME": r"\b(TODO|FIXME|XXX|HACK)\b",
        "unwrap": r"\.unwrap\(|\.expect\(",
        "inspect!": r"\binspect\s*!\s*\(",
    }
    sast = {k: 0 for k in sast_patterns}
    sast_hits = defaultdict(list)
    for sf in src_files:
        code = strip_comments(open(sf, encoding="utf-8", errors="replace").read())
        for k, pat in sast_patterns.items():
            c = len(re.findall(pat, code))
            sast[k] += c
            if c:
                sast_hits[k].append(os.path.basename(sf))

    # ---- 4. LOC / complexity (comments stripped) ----
    metrics = []
    for sf in src_files:
        raw = open(sf, encoding="utf-8", errors="replace").read()
        code = strip_comments(raw)
        lines = code.splitlines()
        code_lines = [l for l in lines if l.strip()]
        metrics.append({
            "file": os.path.basename(sf),
            "loc": raw.count("\n") + 1,
            "code_lines": len(code_lines),
            "blank": sum(1 for l in lines if not l.strip()),
            "comments": sum(1 for l in raw.splitlines() if l.strip().startswith("//")),
            "decisions": complexity_proxy(code),
            "max_line_len": max((len(l) for l in raw.splitlines()), default=0),
        })
    metrics.sort(key=lambda r: r["code_lines"], reverse=True)

    # ---- 5. duplicate code (normalized N-line shingles) ----
    def norm_line(l):
        return re.sub(r"\s+", " ", l.strip())

    hashes = defaultdict(list)  # hash -> list of (file, start_line)
    total_positions = 0
    for sf in src_files + test_files:
        code = open(sf, encoding="utf-8", errors="replace").read()
        lines = [norm_line(l) for l in code.splitlines() if norm_line(l)]
        for i in range(len(lines) - SHINGLE + 1):
            sh = lines[i:i + SHINGLE]
            h = hashlib.md5("|".join(sh).encode("utf-8")).hexdigest()
            hashes[h].append((os.path.basename(sf), i + 1))
            total_positions += 1

    dup_clusters = []
    duplicated_positions = 0
    for h, occ in hashes.items():
        if len(occ) > 1:
            duplicated_positions += len(occ)
            dup_clusters.append({"count": len(occ),
                                 "files": sorted(set(f for f, _ in occ)),
                                 "sample_line": occ[0][1]})
    dup_clusters.sort(key=lambda r: r["count"], reverse=True)
    dup_rate = round(100.0 * duplicated_positions / total_positions, 1) if total_positions else 0.0

    return {
        "coverage": coverage,
        "tests": {"files": test_info, "n_test_files": len(test_info),
                  "total_test_blocks": sum(t["test_blocks"] for t in test_info)},
        "sast": {"counts": sast, "hits": {k: v for k, v in sast_hits.items()}},
        "metrics": metrics,
        "duplicates": {"shingle_size": SHINGLE, "clusters": dup_clusters[:25],
                       "n_clusters": len(dup_clusters),
                       "total_positions": total_positions,
                       "duplicated_positions": duplicated_positions,
                       "rate_pct": dup_rate},
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
