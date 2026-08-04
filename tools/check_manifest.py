#!/usr/bin/env python3
"""Assert the README's notebook table matches what is actually on disk.

EarthByte's GPlately-pyGMT README claims 72 notebooks while its directory holds
80 — exactly the drift a two-line CI check prevents.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
notebooks = sorted(p.stem for p in (ROOT / "Notebooks").glob("T*.ipynb"))
sources = sorted(p.stem for p in (ROOT / "tools" / "sources").glob("T*.py"))
readme = (ROOT / "README.md").read_text()

problems = []

missing = set(sources) - set(notebooks)
if missing:
    problems.append(f"sources with no built notebook: {sorted(missing)}")
orphans = set(notebooks) - set(sources)
if orphans:
    problems.append(f"notebooks with no source: {sorted(orphans)}")

# A table row is a *claim* that a notebook exists — unless it is explicitly
# marked "planned", in which case it is a roadmap entry. Distinguishing the two
# is the whole point: the check exists to stop the README claiming notebooks
# that are not there (EarthByte's GPlately README says 72 while its directory
# holds 80), not to stop us writing down what we intend to build.
rows = re.findall(r"^\|\s*(\d\d)\s*\|(.*)$", readme, re.MULTILINE)
listed = {num for num, rest in rows if "planned" not in rest.lower()}
planned = {num for num, rest in rows if "planned" in rest.lower()}
actual = {n[1:3] for n in notebooks}
if listed != actual:
    problems.append(f"README lists {sorted(listed)} but Notebooks/ holds {sorted(actual)}")

if problems:
    print("MANIFEST DRIFT:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"manifest OK: {len(notebooks)} notebooks built, README agrees"
      f"{f'; {len(planned)} more planned' if planned else ''}")
