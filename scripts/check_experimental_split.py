#!/usr/bin/env python3
"""Check a release dist directory: flashinfer-python and flashinfer-experimental.

release-build.yml runs this on the built artifacts before uploading them. It
fails unless:

* there is exactly one flashinfer-python wheel, one flashinfer-python sdist and
  one flashinfer-experimental wheel, all of the same version;
* the two wheels share no file (they install into one package directory, so a
  shared file would be overwritten and then deleted by an uninstall);
* flashinfer-python ships no experimental package except the parts stable code
  imports without flashinfer-experimental, and the sdist matches it;
* flashinfer-experimental pins ``flashinfer-python==<version>`` and
  flashinfer-python offers it as the ``experimental`` extra;
* with ``--max-file-bytes N``, every artifact is smaller than N bytes (the
  project's PyPI per-file limit; sizes are always reported).

Usage: check_experimental_split.py DIST_DIR [--max-file-bytes N]
"""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

# Keep in sync with _EXPERIMENTAL_TREE_RE / _EXPERIMENTAL_CORE_RE in
# build_backend.py (tests/cpu/infra/test_check_experimental_split.py checks it).
EXPERIMENTAL_TREE = re.compile(
    r"^(flashinfer/experimental/|flashinfer/data/csrc/experimental/|b12x/)"
)
EXPERIMENTAL_CORE = re.compile(
    r"^flashinfer/experimental/"
    r"(__init__\.py$|cake/|cake_nvfp4_per_token/(__init__|support)\.py$)"
)


def _one(dist: Path, pattern: str) -> Path:
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {dist}, found {found}")
    return found[0]


def _wheel(path: Path) -> tuple[set[str], dict, str]:
    with zipfile.ZipFile(path) as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        dist_info = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(dist_info) != 1:
            raise SystemExit(f"{path.name}: expected one .dist-info, got {dist_info}")
        (info,) = dist_info
        metadata = Parser().parsestr(zf.read(f"{info}/METADATA").decode("utf-8"))
        top_level = zf.read(f"{info}/top_level.txt").decode("utf-8")
    payload = {n for n in names if not n.startswith(f"{info}/")}
    return payload, metadata, top_level


def check(dist: Path, max_file_bytes: int | None = None) -> list[str]:
    errors = []
    core_whl = _one(dist, "flashinfer_python-*.whl")
    core_sdist = _one(dist, "flashinfer_python-*.tar.gz")
    exp_whl = _one(dist, "flashinfer_experimental-*.whl")

    core, core_meta, core_top = _wheel(core_whl)
    exp, exp_meta, exp_top = _wheel(exp_whl)
    version = core_meta["Version"]
    if exp_meta["Name"] != "flashinfer-experimental":
        errors.append(f"{exp_whl.name}: Name is {exp_meta['Name']!r}")
    if exp_meta["Version"] != version:
        errors.append(
            f"version mismatch: flashinfer-python {version}, "
            f"flashinfer-experimental {exp_meta['Version']}"
        )
    if f"flashinfer_python-{version}.tar.gz" != core_sdist.name:
        errors.append(f"sdist {core_sdist.name} does not match version {version}")
    pins = exp_meta.get_all("Requires-Dist") or []
    if pins != [f"flashinfer-python=={version}"]:
        errors.append(f"flashinfer-experimental Requires-Dist is {pins}")
    extras = core_meta.get_all("Requires-Dist") or []
    if 'flashinfer-experimental; extra == "experimental"' not in extras:
        errors.append("flashinfer-python lacks the 'experimental' extra")

    shared = core & exp
    if shared:
        errors.append(f"{len(shared)} files in both wheels, e.g. {sorted(shared)[:5]}")
    leaked = sorted(
        n for n in core if EXPERIMENTAL_TREE.match(n) and not EXPERIMENTAL_CORE.match(n)
    )
    if leaked:
        errors.append(f"flashinfer-python ships experimental files: {leaked[:5]}")
    stray = sorted(
        n for n in exp if not EXPERIMENTAL_TREE.match(n) or EXPERIMENTAL_CORE.match(n)
    )
    if stray:
        errors.append(
            f"flashinfer-experimental ships non-experimental files: {stray[:5]}"
        )
    if "flashinfer/experimental/_installed.py" not in exp:
        errors.append("flashinfer-experimental lacks its presence marker _installed.py")
    if "b12x" in core_top.split():
        errors.append("flashinfer-python top_level.txt still lists b12x")
    if sorted(exp_top.split()) != ["b12x", "flashinfer"]:
        errors.append(f"flashinfer-experimental top_level.txt is {exp_top.split()}")

    with tarfile.open(core_sdist) as tf:
        sdist_files = [
            m.name.split("/", 1)[1]
            for m in tf.getmembers()
            if m.isfile() and "/" in m.name
        ]
    in_sdist = sorted(
        n
        for n in sdist_files
        if n.startswith("csrc/experimental/")
        or (EXPERIMENTAL_TREE.match(n) and not EXPERIMENTAL_CORE.match(n))
    )
    if in_sdist:
        errors.append(
            f"flashinfer-python sdist ships experimental files: {in_sdist[:5]}"
        )

    for path in (core_whl, core_sdist, exp_whl):
        size = path.stat().st_size
        print(f"{path.name}: {size / 1e6:.1f} MB")
        if max_file_bytes is not None and size >= max_file_bytes:
            errors.append(
                f"{path.name} is {size:,} bytes, not below {max_file_bytes:,}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=None)
    args = parser.parse_args()
    errors = check(args.dist, args.max_file_bytes)
    if errors:
        raise SystemExit(
            "flashinfer-python / flashinfer-experimental split check failed:\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    print("flashinfer-python / flashinfer-experimental split is consistent.")


if __name__ == "__main__":
    main()
