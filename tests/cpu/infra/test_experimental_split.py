"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The flashinfer-python / flashinfer-experimental split (build_backend.py,
"Wheel post-processing").

Release builds ship most of ``flashinfer/experimental/`` in the separate
flashinfer-experimental distribution. These tests pin the contract that makes
that safe for users who install flashinfer-python alone:

* the split rules of build_backend.py and scripts/check_experimental_split.py
  agree, and every file they keep in flashinfer-python exists;
* in a flashinfer-python-only layout (built here from symlinks to the checkout,
  so no wheel build is needed) ``import flashinfer`` and the stable modules
  that reach into the experimental tree still import, and an experimental
  module raises ``ModuleNotFoundError`` that names the package to install;
* in the full layout (this checkout) nothing changes.

CPU-only; no GPU or JIT compilation required.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))  # build_backend imports build_utils
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_backend():
    return _load("flashinfer_build_backend_split", REPO_ROOT / "build_backend.py")


@pytest.fixture(scope="module")
def split_check():
    return _load(
        "check_experimental_split",
        REPO_ROOT / "scripts" / "check_experimental_split.py",
    )


def _tracked(*paths: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return out.stdout.split()


@pytest.mark.cpu_only
def test_release_check_uses_the_build_rules(build_backend, split_check):
    assert (
        split_check.EXPERIMENTAL_TREE.pattern
        == build_backend._EXPERIMENTAL_TREE_RE.pattern
    )
    assert (
        split_check.EXPERIMENTAL_CORE.pattern
        == build_backend._EXPERIMENTAL_CORE_RE.pattern
    )


@pytest.mark.cpu_only
def test_kept_parts_exist_and_the_marker_is_split_out(build_backend):
    tracked = _tracked("flashinfer/experimental")
    kept = [n for n in tracked if build_backend._EXPERIMENTAL_CORE_RE.match(n)]
    for name in (
        "flashinfer/experimental/__init__.py",
        "flashinfer/experimental/cake_nvfp4_per_token/__init__.py",
        "flashinfer/experimental/cake_nvfp4_per_token/support.py",
    ):
        assert name in kept, name
    assert any(n.startswith("flashinfer/experimental/cake/") for n in kept)
    # The presence marker must ship in flashinfer-experimental only, or a
    # flashinfer-python-only install would claim to have the packages.
    assert "flashinfer/experimental/_installed.py" in tracked
    assert build_backend._is_split_experimental_entry(
        "flashinfer/experimental/_installed.py"
    )
    # Everything else in the tree goes to flashinfer-experimental.
    split = [n for n in tracked if build_backend._is_split_experimental_entry(n)]
    assert len(split) + len(kept) == len(tracked)


@pytest.mark.cpu_only
def test_kept_modules_import_nothing_split_out(build_backend):
    """The parts flashinfer-python keeps must not need flashinfer-experimental."""
    import ast

    bad = []
    for name in _tracked("flashinfer/experimental"):
        if not name.endswith(".py") or not build_backend._EXPERIMENTAL_CORE_RE.match(
            name
        ):
            continue
        package = name[: -len(".py")].split("/")[:-1]
        tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                targets = [module] + [f"{module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                continue
            for target in targets:
                path = target.replace(".", "/")
                if not path.startswith("flashinfer/experimental/"):
                    continue
                candidates = (f"{path}.py", f"{path}/__init__.py")
                if not any((REPO_ROOT / c).is_file() for c in candidates):
                    continue  # an attribute, not a module
                if any(
                    build_backend._is_split_experimental_entry(c)
                    for c in candidates
                    if (REPO_ROOT / c).is_file()
                ):
                    bad.append(f"{name}:{node.lineno}: {target}")
    assert not bad, "kept modules import split-out modules:\n" + "\n".join(bad)


def _core_only_tree(tmp_path: Path, build_backend) -> Path:
    """A flashinfer package that has only what flashinfer-python ships.

    Every top-level entry of the checkout's package is symlinked except
    ``experimental/``, which is rebuilt from the files the split keeps.
    """
    root = tmp_path / "site"
    package = root / "flashinfer"
    package.mkdir(parents=True)
    source = REPO_ROOT / "flashinfer"
    for entry in source.iterdir():
        if entry.name in ("experimental", "__pycache__"):
            continue
        (package / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    for name in _tracked("flashinfer/experimental"):
        if build_backend._is_split_experimental_entry(name):
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(REPO_ROOT / name)
    return root


_PROBE = r"""
import importlib, json, sys, warnings
warnings.simplefilter("ignore")
import flashinfer
import flashinfer.experimental as fe

report = {"installed": fe.experimental_package_installed()}
report["loaded"] = sorted(m for m in sys.modules if m.startswith("flashinfer.experimental"))
assert flashinfer.__file__.startswith(sys.argv[1]), flashinfer.__file__

import flashinfer.moe_ep as moe_ep
from flashinfer.moe_ep.core.kernel.registry import _MEGA_KERNEL_REGISTRY
report["w8a8_registered"] = "sm103_mxfp8_mxfp8_bf16_cake" in _MEGA_KERNEL_REGISTRY
report["w8a8_in_all"] = "Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig" in moe_ep.__all__
star = {}
exec("from flashinfer.moe_ep import *", star)
from flashinfer.gemm import mm_fp4
report["mm_fp4_experimental"] = sorted(mm_fp4.experimental_backends)
import flashinfer.cake_gqa_det_train
from flashinfer.experimental.cake import cake_gqa_det_train
report["errors"] = {}
for name in ("flashinfer.experimental.qwen_3_5_block",
             "flashinfer.experimental.cake_nvfp4_per_token.cake_backend",
             "flashinfer.mega_moe_v3"):
    try:
        importlib.import_module(name)
        report["errors"][name] = None
    except ModuleNotFoundError as error:
        report["errors"][name] = [error.name, str(error)]
try:
    moe_ep.Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig
    report["lazy_attr"] = None
except ModuleNotFoundError as error:
    report["lazy_attr"] = str(error)
print(json.dumps(report))
"""


def _probe(root: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(root), FLASHINFER_DISABLE_VERSION_CHECK="1")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(root)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stderr}", pytrace=False)
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SANCTIONED = [
    "flashinfer.experimental",
    "flashinfer.experimental.cake_nvfp4_per_token",
    "flashinfer.experimental.cake_nvfp4_per_token.support",
]


@pytest.mark.cpu_only
def test_flashinfer_python_alone(tmp_path, build_backend):
    report = _probe(_core_only_tree(tmp_path, build_backend))
    assert report["installed"] is False
    assert report["loaded"] == _SANCTIONED
    # stable modules import; the experimental backend is just not registered
    assert report["w8a8_registered"] is False
    assert report["w8a8_in_all"] is False
    assert report["mm_fp4_experimental"] == ["cake"]
    for name, error in report["errors"].items():
        assert error is not None, f"{name} imported without flashinfer-experimental"
        assert error[0].startswith("flashinfer.experimental."), error
        assert "pip install" in error[1] and "flashinfer-experimental" in error[1]
    assert "flashinfer-experimental" in report["lazy_attr"]


@pytest.mark.cpu_only
def test_full_checkout(build_backend):
    report = _probe(REPO_ROOT)
    assert report["installed"] is True
    assert report["loaded"] == _SANCTIONED
    assert report["w8a8_registered"] is True
    assert report["w8a8_in_all"] is True
    assert report["errors"] == dict.fromkeys(report["errors"])
    assert report["lazy_attr"] is None


@pytest.mark.cpu_only
def test_missing_module_message_names_the_matching_version(monkeypatch):
    import flashinfer.experimental as fe
    import flashinfer.version

    monkeypatch.setattr(flashinfer.version, "__version__", "1.2.3")
    message = fe.missing_experimental_package_message("flashinfer.experimental.x")
    assert '"flashinfer-experimental==1.2.3"' in message
    assert '"flashinfer-python[experimental]==1.2.3"' in message
    monkeypatch.setattr(flashinfer.version, "__version__", "0.0.0+unknown")
    message = fe.missing_experimental_package_message("flashinfer.experimental.x")
    assert '"flashinfer-experimental"' in message


@pytest.mark.cpu_only
def test_finder_defers_when_the_packages_are_installed(monkeypatch):
    import flashinfer.experimental as fe

    finder = fe._MissingExperimentalPackageFinder()
    monkeypatch.setattr(fe, "_installed", True)
    assert finder.find_spec("flashinfer.experimental.nope") is None
    with pytest.raises(ModuleNotFoundError, match=r"No module named"):
        importlib.import_module("flashinfer.experimental.does_not_exist")
    monkeypatch.setattr(fe, "_installed", False)
    assert finder.find_spec("flashinfer.other") is None
    assert finder.find_spec(fe._MARKER) is None
    with pytest.raises(ModuleNotFoundError, match="flashinfer-experimental"):
        finder.find_spec("flashinfer.experimental.nope")
    # exactly one finder, however often the namespace is re-executed
    assert (
        sum(
            type(f).__name__ == "_MissingExperimentalPackageFinder"
            for f in sys.meta_path
        )
        == 1
    )


@pytest.mark.cpu_only
def test_version_mismatch_warns(monkeypatch):
    import importlib.metadata

    import flashinfer.experimental as fe
    import flashinfer.version

    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setattr(flashinfer.version, "__version__", "1.2.3")
    versions = {"flashinfer-experimental": "1.2.2"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    with pytest.warns(UserWarning, match=re.escape("flashinfer-experimental 1.2.2")):
        fe._check_experimental_distribution_version()
    versions["flashinfer-experimental"] = "1.2.3"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fe._check_experimental_distribution_version()
