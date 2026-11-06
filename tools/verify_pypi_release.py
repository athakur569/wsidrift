"""Check that a PyPI release of wsidrift was built from the commit its tag names (README, Versions).

Downloads the release's source archive from PyPI, extracts it to a temporary directory, and compares its `src/` and
`pyproject.toml` with the tagged commit file for file (`git archive`). Prints each difference; exits 1 if any.
Run from a clone with tags:

    python tools/verify_pypi_release.py 0.1.1        # compares with tag v0.1.1
"""

import filecmp
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def diff_trees(a: Path, b: Path, rel: str = "") -> list[str]:
    cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
    out = [f"only in PyPI: {rel}{n}" for n in cmp.left_only if not n.endswith(".egg-info")]
    out += [f"only in tag: {rel}{n}" for n in cmp.right_only]
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    out += [f"differs: {rel}{n}" for n in mismatch + errors]
    for d in cmp.common_dirs:
        out += diff_trees(a / d, b / d, f"{rel}{d}/")
    return out


def main(version: str) -> int:
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/wsidrift/{version}/json"))
    sdist = next(f for f in meta["urls"] if f["packagetype"] == "sdist")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sdist["url"]).read())) as tf:
            tf.extractall(tmp / "pypi", filter="data")
        pypi = next((tmp / "pypi").iterdir())
        archive = subprocess.run(["git", "archive", f"v{version}", "src", "pyproject.toml"],
                                 check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(tmp / "tag", filter="data")
        problems = diff_trees(pypi / "src", tmp / "tag" / "src")
        if not filecmp.cmp(pypi / "pyproject.toml", tmp / "tag" / "pyproject.toml", shallow=False):
            problems.append("differs: pyproject.toml")
    print(f"PyPI wsidrift {version} ({sdist['upload_time']}) against tag v{version}: "
          + ("identical" if not problems else f"{len(problems)} differences"))
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
