#!/usr/bin/env python3
"""
Build the tutorial notebooks from the plain-Python sources in ``tools/sources``.

The ``.py`` files are the source of truth, in jupytext's ``percent`` format.
Keeping it that way means readable diffs, no output-blob churn in git, and CI
that can execute the notebooks as ordinary code. The ``.ipynb`` files in
``Notebooks/`` are generated, executed, and committed with outputs preserved so
that every figure renders on GitHub without anyone installing a thing.

    python tools/build_notebooks.py            # build + execute all
    python tools/build_notebooks.py 00 06      # only these
    python tools/build_notebooks.py --no-exec  # build only
"""

import argparse
import pathlib
import re
import sys
import time

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "sources"
OUT = ROOT / "Notebooks"

CELL_RE = re.compile(r"^# %%(.*)$", re.MULTILINE)


def parse_percent(text):
    """Split a jupytext ``percent``-format file into (kind, source) cells."""
    parts = CELL_RE.split(text)
    if parts[0].strip():
        raise ValueError("content before the first '# %%' marker")
    cells = []
    for header, body in zip(parts[1::2], parts[2::2]):
        kind = "markdown" if "[markdown]" in header else "code"
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in body.strip("\n").splitlines()
            )
        cells.append((kind, body.strip("\n")))
    return cells


def build(path):
    cells = []
    for kind, body in parse_percent(path.read_text()):
        if not body.strip():
            continue
        cells.append(new_markdown_cell(body) if kind == "markdown"
                     else new_code_cell(body))
    nb = new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.metadata.language_info = {"name": "python"}
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="notebook number prefixes to build")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    sources = sorted(SRC.glob("*.py"))
    if args.which:
        sources = [s for s in sources
                   if any(s.name.startswith(w) for w in args.which)]
    if not sources:
        print("no matching sources", file=sys.stderr)
        return 1

    failures = []
    for src in sources:
        nb = build(src)
        dest = OUT / (src.stem + ".ipynb")

        if not args.no_exec:
            from nbclient import NotebookClient

            print(f"executing {src.stem} ...", flush=True)
            t0 = time.time()
            # Execute with the NOTEBOOK'S directory as cwd, not the repo root.
            # That is what nbmake does in CI and what a student gets when they
            # open Notebooks/T01 in JupyterLab, so every notebook's
            # `sys.path.insert(0, "..")` and every relative data path means the
            # same thing everywhere. Building from the repo root instead let
            # those paths be wrong-but-harmless here and broken elsewhere.
            client = NotebookClient(nb, timeout=args.timeout,
                                    kernel_name="python3",
                                    resources={"metadata": {"path": str(OUT)}})
            try:
                client.execute()
            except Exception as exc:                      # noqa: BLE001
                failures.append((src.stem, str(exc).splitlines()[0][:160]))
                print(f"  FAILED after {time.time() - t0:.1f}s")
                nbformat.write(nb, dest)
                continue
            print(f"  ok in {time.time() - t0:.1f}s")

        nbformat.write(nb, dest)
        print(f"  wrote {dest.relative_to(ROOT)}")

    if failures:
        print("\nFAILURES:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
