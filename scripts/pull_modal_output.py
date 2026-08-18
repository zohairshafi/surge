#!/usr/bin/env python3
"""Pull a SURGE output directory from the shared Modal volume, excluding graphs/.

The per-graph bundles in ``graphs/`` are the dominant volume payload (edge
index + target adjacency per reconstruction level per lake) and are not needed
for downstream analysis, so they are skipped by default.

Usage:
    python scripts/pull_modal_output.py --path 10k_v1.0 [--dest output/10k_v1.0]
    python scripts/pull_modal_output.py --path 25k_v1.0 --dest output/25k_v1.0
    python scripts/pull_modal_output.py --path 10k_v1.0 --include-graphs   # opt back in

Requires the Modal CLI on PATH.
"""
import argparse
import os
import subprocess
import sys

VOLUME = "rol_output"
EXCLUDED = {"graphs"}


def _run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def list_top_level(remote_path):
    out = subprocess.run(
        ["modal", "volume", "ls", VOLUME, f"/{remote_path}"],
        check=True, capture_output=True, text=True,
    ).stdout
    items = []
    for line in out.splitlines():
        # Lines look like "<path>/<name>" (dirs) or "<path>/<name>"
        if remote_path in line:
            name = line.strip().split("/")[-1]
            if name:
                items.append(name)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="10k_v1.0",
                    help="Remote path inside the volume (default: 10k_v1.0)")
    ap.add_argument("--dest", default=None,
                    help="Local destination dir (default: output/<path>)")
    ap.add_argument("--include-graphs", action="store_true",
                    help="Also pull the graphs/ subdir")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be pulled without downloading")
    args = ap.parse_args()

    dest = args.dest or f"output/{args.path}"
    items = list_top_level(args.path)
    if not items:
        print(f"No top-level items found under /{args.path}")
        sys.exit(1)

    pulls = [i for i in items if args.include_graphs or i not in EXCLUDED]
    skipped = [i for i in items if not args.include_graphs and i in EXCLUDED]
    print(f"Found {len(items)} top-level item(s) under /{args.path}:")
    print(f"  PULLING: {', '.join(sorted(pulls))}")
    if skipped:
        print(f"  SKIPPING: {', '.join(sorted(skipped))}")

    if args.dry_run:
        return

    os.makedirs(dest, exist_ok=True)
    for item in sorted(pulls):
        _run(["modal", "volume", "get", VOLUME,
              f"/{args.path}/{item}", dest, "--force"])
    print(f"Done. Local copy at {dest}")


if __name__ == "__main__":
    main()
