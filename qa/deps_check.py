#!/usr/bin/env python3
"""MoonBit dependency-pin guard (supply-chain reproducibility check).

NOTE: MoonBit has **no** ``moon-lock.json``. Dependencies are pinned directly in
``moon.mod`` using an ``@version`` suffix, e.g.::

    import { "riantr/moonbit_image@0.3.7" }

Toolchain-bundled packages (``moonbitlang/core/*``) carry no version and are
exempt. This guard fails the build if any *external* registry dependency lacks an
exact ``x.y.z`` semver pin, so a dependency can never silently float.

It also prints the resolved dependency tree (``moon tree``) for artifact upload.
"""
import os
import re
import sys
import glob
import argparse
import subprocess

IMPORT_BLOCK_RE = re.compile(r"import\s*\{([^}]*)\}", re.S)
STRING_RE = re.compile(r'"([^"]+)"')
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
EXCLUDE_DIRS = ("_build", ".mooncakes")


def check_file(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    block = IMPORT_BLOCK_RE.search(txt)
    if not block:
        return []  # no imports -> nothing to check
    problems = []
    for m in STRING_RE.finditer(block.group(1)):
        dep = m.group(1)
        if dep.startswith("moonbitlang/core"):
            continue  # toolchain-bundled, intentionally version-less
        if "/" not in dep:
            continue
        if "@" not in dep:
            problems.append(f"{path}: {dep} has NO version pin")
        else:
            ver = dep.rsplit("@", 1)[1]
            if not SEMVER_RE.fullmatch(ver):
                problems.append(
                    f"{path}: {dep} pin '{ver}' is not an exact semver (x.y.z)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    mods = glob.glob(os.path.join(args.root, "**", "moon.mod"), recursive=True)
    mods = [f for f in mods
            if all(d not in f.split(os.sep) for d in EXCLUDE_DIRS)]
    all_problems = []
    module_root = None
    for f in mods:
        all_problems += check_file(f)
        if module_root is None:
            module_root = os.path.dirname(f)

    if all_problems:
        print("DEPENDENCY PIN CHECK FAILED:")
        for p in all_problems:
            print("  -", p)
        sys.exit(1)

    print(f"Dependency pin check passed: {len(mods)} moon.mod scanned, "
          f"all external deps exactly pinned (x.y.z).")

    if module_root:
        try:
            # moon tree has no -C; run from the module root.
            out = subprocess.run(["moon", "tree"], cwd=module_root,
                                 capture_output=True, text=True)
            print("\nResolved dependency tree:\n" + out.stdout)
            if out.returncode != 0 and out.stderr:
                print(out.stderr)
        except FileNotFoundError:
            print("(moon not on PATH; skipping `moon tree` output)")


if __name__ == "__main__":
    main()
