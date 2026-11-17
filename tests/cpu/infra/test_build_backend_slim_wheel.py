"""The wheel post-processing of build_backend: level-9 re-deflate, dropping the
vendored 3rdparty Python trees (flashinfer-ai/flashinfer#7977), and the
flashinfer-python / flashinfer-experimental split of the experimental tree:
the release-only drop from the flashinfer-python wheel and sdist, and the
extraction of the flashinfer-experimental wheel from the full wheel."""

import base64
import csv
import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import tarfile

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def build_backend():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(
        "flashinfer_build_backend", _ROOT / "build_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIST_INFO = "flashinfer_python-0.0.0.dist-info"
FILES = {
    "flashinfer/__init__.py": b"print('hi')\n",
    "flashinfer/experimental/__init__.py": b"",
    "flashinfer/experimental/_installed.py": b"",
    "flashinfer/experimental/cake/__init__.py": b"",
    "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu": b"// cake\n" * 50,
    "flashinfer/experimental/cake_nvfp4_per_token/__init__.py": b"",
    "flashinfer/experimental/cake_nvfp4_per_token/support.py": b"# checker\n",
    "flashinfer/experimental/cake_nvfp4_per_token/cake_backend.py": b"# kernels\n",
    "flashinfer/experimental/foo/__init__.py": b"",
    "flashinfer/experimental/foo/build.json": b'{"programs": []}\n',
    "flashinfer/experimental/foo/csrc/kernel.cu": b"__global__ void k() {}\n" * 200,
    "flashinfer/experimental/foo/csrc/sub/kernel2.cu": b"// generated\n" * 50,
    "flashinfer/data/csrc/experimental/bar/generated/gemm.cu": b"// deepgemm\n" * 50,
    "flashinfer/data/csrc/norm/rmsnorm.cu": b"// stable kernel\n" * 50,
    "flashinfer/data/include/flashinfer/utils.cuh": b"#pragma once\n",
    "flashinfer/data/cutlass/include/cutlass/cutlass.h": b"#pragma once\n",
    "flashinfer/data/cutlass/tools/util/include/cutlass/util/host_tensor.h": b"#pragma once\n",
    "flashinfer/data/cutlass/python/cutlass_library/generator.py": b"def gen(): pass\n",
    "flashinfer/data/cutlass/examples/00_basic_gemm/basic_gemm.cu": b"// example\n",
    "flashinfer/data/cccl/cub/cub/cub.cuh": b"#pragma once\n",
    "flashinfer/data/cccl/libcudacxx/include/cuda/std/version": b"// version\n",
    "flashinfer/data/cccl/thrust/thrust/version.h": b"#pragma once\n",
    "flashinfer/data/cccl/python/cuda_cccl/setup.py": b"# setup\n",
    "flashinfer/data/spdlog/include/spdlog/spdlog.h": b"#pragma once\n",
    f"{DIST_INFO}/METADATA": b"Metadata-Version: 2.1\nName: flashinfer-python\nVersion: 0.0.0\n",
    "b12x/__init__.py": b"# alias\n",
    f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    f"{DIST_INFO}/top_level.txt": b"_flashinfer_filelock\nb12x\nflashinfer\n",
    f"{DIST_INFO}/licenses/LICENSE": b"Apache-2.0\n",
}

# Ship in flashinfer-experimental when the split is on.
EXPERIMENTAL = {
    "flashinfer/experimental/_installed.py",
    "flashinfer/experimental/cake_nvfp4_per_token/cake_backend.py",
    "flashinfer/experimental/foo/__init__.py",
    "flashinfer/experimental/foo/build.json",
    "flashinfer/experimental/foo/csrc/kernel.cu",
    "flashinfer/experimental/foo/csrc/sub/kernel2.cu",
    "flashinfer/data/csrc/experimental/bar/generated/gemm.cu",
    "b12x/__init__.py",
}


def _record_hash(data: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def _write_wheel(path: Path, files: dict[str, bytes] = FILES) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        rows = []
        for name, data in files.items():
            zf.writestr(name, data)
            rows.append((name, _record_hash(data), str(len(data))))
        rows.append((f"{DIST_INFO}/RECORD", "", ""))
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        zf.writestr(f"{DIST_INFO}/RECORD", text.getvalue())


def _entries(path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i for i in zf.infolist()}


def _check_record(path: Path) -> set[str]:
    """RECORD lists exactly the entries, with matching hashes and sizes."""
    with zipfile.ZipFile(path) as zf:
        all_names = [i.filename for i in zf.infolist()]
        assert len(all_names) == len(set(all_names)), "duplicate zip entries"
        names = set(all_names)
        record = zf.read(f"{DIST_INFO}/RECORD").decode()
        listed = {}
        for row in csv.reader(io.StringIO(record)):
            listed[row[0]] = (row[1], row[2])
        assert set(listed) == names
        for name, (digest, size) in listed.items():
            if name == f"{DIST_INFO}/RECORD":
                assert digest == "" and size == ""
                continue
            data = zf.read(name)
            assert digest == _record_hash(data), name
            assert int(size) == len(data), name
        # RECORD is the last entry, as the wheel tools write it.
        assert zf.infolist()[-1].filename == f"{DIST_INFO}/RECORD"
    return names


def _check_record_of(path: Path, dist_info: str) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist()]
        assert len(names) == len(set(names)), "duplicate zip entries"
        rows = list(csv.reader(io.StringIO(zf.read(f"{dist_info}/RECORD").decode())))
        listed = {row[0]: (row[1], row[2]) for row in rows}
        assert set(listed) == set(names)
        for name, (digest, size) in listed.items():
            if name == f"{dist_info}/RECORD":
                assert digest == "" and size == ""
                continue
            data = zf.read(name)
            assert digest == _record_hash(data), name
            assert int(size) == len(data), name
        assert zf.infolist()[-1].filename == f"{dist_info}/RECORD"
    return set(names)


def test_default_keeps_experimental_tree_and_drops_vendored_python(
    tmp_path, monkeypatch, build_backend
):
    monkeypatch.delenv(build_backend._SPLIT_EXPERIMENTAL_FLAG, raising=False)
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    stats = build_backend._slim_wheel(wheel)
    names = _check_record(wheel)
    # vendored 3rdparty outside the header trees is gone
    assert "flashinfer/data/cutlass/python/cutlass_library/generator.py" not in names
    assert "flashinfer/data/cutlass/examples/00_basic_gemm/basic_gemm.cu" not in names
    assert "flashinfer/data/cccl/python/cuda_cccl/setup.py" not in names
    # header trees stay
    for kept in (
        "flashinfer/data/cutlass/include/cutlass/cutlass.h",
        "flashinfer/data/cutlass/tools/util/include/cutlass/util/host_tensor.h",
        "flashinfer/data/cccl/cub/cub/cub.cuh",
        "flashinfer/data/cccl/libcudacxx/include/cuda/std/version",
        "flashinfer/data/cccl/thrust/thrust/version.h",
        "flashinfer/data/spdlog/include/spdlog/spdlog.h",
    ):
        assert kept in names
    # without the release switch the whole experimental tree stays (source installs)
    assert names >= EXPERIMENTAL
    assert stats["dropped_entries"] == 3
    with zipfile.ZipFile(wheel) as zf:
        assert b"b12x" in zf.read(f"{DIST_INFO}/top_level.txt")
    # every entry is deflated and the content is unchanged
    for name, info in _entries(wheel).items():
        assert info.compress_type == zipfile.ZIP_DEFLATED, name
    with zipfile.ZipFile(wheel) as zf:
        assert (
            zf.read("flashinfer/data/csrc/norm/rmsnorm.cu")
            == FILES["flashinfer/data/csrc/norm/rmsnorm.cu"]
        )


def test_release_switch_splits_the_experimental_tree_out(
    tmp_path, monkeypatch, build_backend
):
    monkeypatch.setenv(build_backend._SPLIT_EXPERIMENTAL_FLAG, "1")
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    stats = build_backend._slim_wheel(wheel)
    names = _check_record(wheel)
    assert not EXPERIMENTAL & names
    # what stable code needs without flashinfer-experimental stays
    for kept in (
        "flashinfer/experimental/__init__.py",
        "flashinfer/experimental/cake/__init__.py",
        "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu",
        "flashinfer/experimental/cake_nvfp4_per_token/__init__.py",
        "flashinfer/experimental/cake_nvfp4_per_token/support.py",
        "flashinfer/data/csrc/norm/rmsnorm.cu",
        "flashinfer/data/include/flashinfer/utils.cuh",
    ):
        assert kept in names
    # the b12x alias package moved, so flashinfer-python no longer declares it
    with zipfile.ZipFile(wheel) as zf:
        top_level = zf.read(f"{DIST_INFO}/top_level.txt").decode().split()
    assert top_level == ["_flashinfer_filelock", "flashinfer"]
    assert stats["dropped_entries"] == 3 + len(EXPERIMENTAL)


def test_slimming_is_idempotent(tmp_path, monkeypatch, build_backend):
    monkeypatch.setenv(build_backend._SPLIT_EXPERIMENTAL_FLAG, "1")
    wheel = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    build_backend._slim_wheel(wheel)
    first = wheel.read_bytes()
    second_stats = build_backend._slim_wheel(wheel)
    assert second_stats["dropped_entries"] == 0
    with zipfile.ZipFile(io.BytesIO(first)) as zf:
        first_names = set(zf.namelist())
    assert set(_entries(wheel)) == first_names
    _check_record(wheel)


def test_experimental_wheel_complements_the_split_wheel(
    tmp_path, monkeypatch, build_backend
):
    full = tmp_path / "full" / "flashinfer_python-0.0.0-py3-none-any.whl"
    full.parent.mkdir()
    _write_wheel(full)
    out = build_backend._extract_experimental_wheel(full, tmp_path / "out")
    assert out.name == "flashinfer_experimental-0.0.0-py3-none-any.whl"
    info = "flashinfer_experimental-0.0.0.dist-info"
    exp_names = _check_record_of(out, info)
    payload = {n for n in exp_names if not n.startswith(f"{info}/")}
    assert payload == EXPERIMENTAL

    monkeypatch.setenv(build_backend._SPLIT_EXPERIMENTAL_FLAG, "1")
    build_backend._slim_wheel(full)
    core = {n for n in _check_record(full) if not n.startswith(f"{DIST_INFO}/")}
    # the two wheels partition the files of the full wheel (minus vendored trees)
    assert not core & payload
    expected = {
        n
        for n in FILES
        if not n.startswith(f"{DIST_INFO}/")
        and not build_backend._wheel_entry_dropped(n, False)
    }
    assert core | payload == expected
    with zipfile.ZipFile(out) as zf:
        for name in payload:
            assert zf.read(name) == FILES[name], name
        metadata = zf.read(f"{info}/METADATA").decode()
        assert zf.read(f"{info}/top_level.txt").decode().split() == [
            "b12x",
            "flashinfer",
        ]
        assert zf.read(f"{info}/licenses/LICENSE") == b"Apache-2.0\n"
    assert "Name: flashinfer-experimental\n" in metadata
    assert "Version: 0.0.0\n" in metadata
    # both distributions install into one package directory: same build only
    assert "Requires-Dist: flashinfer-python==0.0.0\n" in metadata
    assert "License-File: LICENSE\n" in metadata


def test_experimental_wheel_requires_experimental_files(tmp_path, build_backend):
    files = {
        n: d
        for n, d in FILES.items()
        if not build_backend._is_split_experimental_entry(n)
    }
    full = tmp_path / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(full, files)
    with pytest.raises(RuntimeError, match="no experimental files"):
        build_backend._extract_experimental_wheel(full, tmp_path / "out")


def _write_sdist(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"flashinfer_python-0.0.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_sdist_split(tmp_path, build_backend):
    sources = "\n".join(
        [
            "csrc/experimental/bar/gemm.cu",
            "csrc/norm/rmsnorm.cu",
            "flashinfer/experimental/__init__.py",
            "flashinfer/experimental/foo/__init__.py",
            "flashinfer/experimental/b12x/_compat/b12x/__init__.py",
            "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu",
        ]
    )
    files = {
        "pyproject.toml": b"[project]\n",
        "csrc/experimental/bar/gemm.cu": b"// exp\n",
        "csrc/norm/rmsnorm.cu": b"// stable\n",
        "flashinfer/__init__.py": b"",
        "flashinfer/experimental/__init__.py": b"",
        "flashinfer/experimental/_installed.py": b"",
        "flashinfer/experimental/foo/__init__.py": b"",
        "flashinfer/experimental/b12x/_compat/b12x/__init__.py": b"",
        "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu": b"// cake\n",
        "flashinfer/experimental/cake_nvfp4_per_token/support.py": b"",
        "flashinfer_python.egg-info/SOURCES.txt": sources.encode(),
        "flashinfer_python.egg-info/top_level.txt": b"b12x\nflashinfer\n",
    }
    sdist = tmp_path / "flashinfer_python-0.0.0.tar.gz"
    _write_sdist(sdist, files)
    stats = build_backend._split_sdist(sdist)
    with tarfile.open(sdist) as tf:
        members = {m.name.split("/", 1)[1]: m for m in tf.getmembers()}
        assert set(members) == {
            "pyproject.toml",
            "csrc/norm/rmsnorm.cu",
            "flashinfer/__init__.py",
            "flashinfer/experimental/__init__.py",
            "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu",
            "flashinfer/experimental/cake_nvfp4_per_token/support.py",
            "flashinfer_python.egg-info/SOURCES.txt",
            "flashinfer_python.egg-info/top_level.txt",
        }
        listed = (
            tf.extractfile(members["flashinfer_python.egg-info/SOURCES.txt"])
            .read()
            .decode()
            .split()
        )
        top_level = (
            tf.extractfile(members["flashinfer_python.egg-info/top_level.txt"])
            .read()
            .decode()
            .split()
        )
    assert listed == [
        "csrc/norm/rmsnorm.cu",
        "flashinfer/experimental/__init__.py",
        "flashinfer/experimental/cake/cake_gqa_det_train/csrc/k.cu",
    ]
    assert top_level == ["flashinfer"]
    assert stats["dropped_entries"] == 4


def test_entry_rules(build_backend):
    dropped = build_backend._wheel_entry_dropped
    assert dropped("flashinfer/data/cutlass/python/x.py", False)
    assert not dropped("flashinfer/data/cutlass/include/cutlass/x.h", False)
    # without the split nothing experimental is dropped
    for name in (
        "flashinfer/experimental/foo/csrc/k.cu",
        "flashinfer/experimental/foo/api.py",
        "flashinfer/data/csrc/experimental/bar/k.cu",
        "b12x/__init__.py",
    ):
        assert not dropped(name, False), name
        assert dropped(name, True), name
    # the experimental tree goes as a whole, Python modules and manifests included
    assert dropped("flashinfer/experimental/foo/build.json", True)
    assert dropped("flashinfer/experimental/hc_sum_sm100/csrc/binding.cu", True)
    assert dropped("flashinfer/experimental/_installed.py", True)
    assert dropped("flashinfer/experimental/cake_nvfp4_per_token/cake_jit.py", True)
    # stable code keeps what it needs
    for name in (
        "flashinfer/experimental/__init__.py",
        "flashinfer/experimental/cake_nvfp4_per_token/__init__.py",
        "flashinfer/experimental/cake_nvfp4_per_token/support.py",
        "flashinfer/experimental/cake/__init__.py",
        "flashinfer/experimental/cake/cake_causal_conv1d_det_train/csrc/k.cu",
        "flashinfer/experimental/cake/cake_kda_nondet_train/build.json",
        "flashinfer/data/csrc/norm/k.cu",
        "flashinfer/cake_gdp_backend/csrc/k.cu",
        "flashinfer/experimentalish.py",
        "b12x_like/x.py",
    ):
        assert not dropped(name, True), name
    # the sdist spells the shared generated sources and the alias package by
    # their repository paths
    split = build_backend._sdist_entry_split
    assert split("csrc/experimental/bar/k.cu")
    assert split("flashinfer/experimental/b12x/_compat/b12x/__init__.py")
    assert not split("csrc/norm/k.cu")
    assert not split("flashinfer/experimental/cake/x/csrc/k.cu")
