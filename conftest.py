"""Make the repository root importable when tests are run from a clone.

Without this, a bare `pytest tests` fails at collection with
`ModuleNotFoundError: No module named 'geodynkit'`, because pytest only adds
the *rootdir* to sys.path under the legacy "prepend" import mode when a test
package has no __init__.py — and not at all for a src-less layout like this
one. Running `python -m pytest` happens to work, since that form puts the
current directory on sys.path, which is exactly the kind of difference that
passes locally and fails in CI.

CI additionally does `pip install -e . --no-deps`, which is the real fix. This
file is here so that someone who clones the repo and types `pytest` gets a
working test run without having to install anything first.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
