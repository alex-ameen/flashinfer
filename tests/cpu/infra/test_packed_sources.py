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

Packed JIT sources: build_backend packs flashinfer/data/csrc/** of a wheel into
flashinfer/data/csrc.tar.xz (FLASHINFER_PACK_CSRC=1), and
flashinfer/jit/_packed_sources.py unpacks it once into the cache.

CPU-only; loads both modules in isolation (no flashinfer import, no GPU).
"""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import lzma
import multiprocessing
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
DIST_INFO = "flashinfer_python-1.2.3.dist-info"


def _load(name, path):
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_backend():
    return _load("flashinfer_build_backend_pack", _ROOT / "build_backend.py")


@pytest.fixture(scope="module")
def packed():
    return _load(
        "flashinfer_packed_sources", _ROOT / "flashinfer" / "jit" / "_packed_sources.py"
    )


SOURCES = {
    "flashinfer/data/csrc/norm/rmsnorm.cu": b"// rmsnorm\n" * 300,
    "flashinfer/data/csrc/gen/a_kernel.cu": b"// generated kernel A\n" * 400,
    "flashinfer/data/csrc/gen/b_kernel.cu": b"// generated kernel A\n" * 399
    + b"// B\n",
    "flashinfer/data/csrc/tool.jinja": b"#!/bin/sh\n",
}
OTHER = {
    "flashinfer/__init__.py": b"",
    "flashinfer/data/include/flashinfer/utils.cuh": b"#pragma once\n",
    "flashinfer/data/csrcfoo/not_packed.txt": b"x\n",
}


def _record_hash(data):
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def _write_wheel(path, files):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        rows = []
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 2, 3, 4, 6))
            mode = 0o755 if name.endswith(".jinja") else 0o644
            info.external_attr = (0o100000 | mode) << 16
            zf.writestr(info, data)
            rows.append((name, _record_hash(data), str(len(data))))
        rows.append((f"{DIST_INFO}/RECORD", "", ""))
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        zf.writestr(f"{DIST_INFO}/RECORD", text.getvalue())


def _build(tmp_path, monkeypatch, build_backend, pack):
    if pack:
        monkeypatch.setenv(build_backend._PACK_CSRC_FLAG, "1")
    else:
        monkeypatch.delenv(build_backend._PACK_CSRC_FLAG, raising=False)
    monkeypatch.delenv(build_backend._EXPERIMENTAL_SOURCES_FLAG, raising=False)
    wheel = tmp_path / "flashinfer_python-1.2.3-py3-none-any.whl"
    files = {
        **SOURCES,
        **OTHER,
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    _write_wheel(wheel, files)
    stats = build_backend._slim_wheel(wheel)
    return wheel, stats


def _check_record(path):
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist()]
        assert len(names) == len(set(names))
        record = zf.read(f"{DIST_INFO}/RECORD").decode()
        listed = {row[0]: row[1:] for row in csv.reader(io.StringIO(record))}
        assert set(listed) == set(names)
        for name, (digest, size) in listed.items():
            if name.endswith("RECORD"):
                continue
            data = zf.read(name)
            assert digest == _record_hash(data) and int(size) == len(data), name
    return set(names)


@pytest.mark.cpu_only
def test_default_keeps_plain_sources(tmp_path, monkeypatch, build_backend):
    wheel, stats = _build(tmp_path, monkeypatch, build_backend, pack=False)
    names = _check_record(wheel)
    assert set(SOURCES) <= names
    assert build_backend._PACKED_CSRC_ARCHIVE not in names
    assert stats["packed_entries"] == 0


@pytest.mark.cpu_only
def test_pack_moves_every_source_into_one_archive(tmp_path, monkeypatch, build_backend):
    wheel, stats = _build(tmp_path, monkeypatch, build_backend, pack=True)
    names = _check_record(wheel)
    assert not set(SOURCES) & names
    # neighbours that only share the prefix text stay
    assert set(OTHER) <= names
    assert stats["packed_entries"] == len(SOURCES)
    with zipfile.ZipFile(wheel) as zf:
        archive = zf.read(build_backend._PACKED_CSRC_ARCHIVE)
        manifest = json.loads(zf.read(build_backend._PACKED_CSRC_MANIFEST))
    assert manifest == {
        "format": 1,
        "sha256": hashlib.sha256(archive).hexdigest(),
        "files": len(SOURCES),
        "bytes": sum(len(d) for d in SOURCES.values()),
        "version": "1.2.3",
    }
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == sorted(
            n[len("flashinfer/data/csrc/") :] for n in SOURCES
        )
        for member in members:
            original = SOURCES[f"flashinfer/data/csrc/{member.name}"]
            assert tar.extractfile(member).read() == original
            assert member.mtime == 0 and member.uid == 0 and member.uname == ""
            expected = 0o755 if member.name.endswith(".jinja") else 0o644
            assert member.mode == expected, member.name


@pytest.mark.cpu_only
def test_pack_is_deterministic_and_idempotent(tmp_path, monkeypatch, build_backend):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, _ = _build(tmp_path / "a", monkeypatch, build_backend, pack=True)
    second, _ = _build(tmp_path / "b", monkeypatch, build_backend, pack=True)
    assert first.read_bytes() == second.read_bytes()
    # re-running the post-processing on a packed wheel changes nothing
    before = first.read_bytes()
    stats = build_backend._slim_wheel(first)
    assert stats["packed_entries"] == 0
    assert first.read_bytes() == before
    _check_record(first)


def _make_archive(tmp_path, packed, files, members=None, name="pkg"):
    buf = io.BytesIO()
    with (
        lzma.open(buf, "wb") as xz,
        tarfile.open(fileobj=xz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for path, data in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o755 if path.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
        for info in members or ():
            tar.addfile(info)
    archive = tmp_path / f"{name}.tar.xz"
    archive.write_bytes(buf.getvalue())
    manifest = tmp_path / f"{name}.json"
    manifest.write_text(
        json.dumps(
            {
                "format": packed.MANIFEST_FORMAT,
                "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
                "files": len(files) + sum(m.isfile() for m in members or ()),
                "version": "1.2.3",
            }
        )
    )
    return archive, manifest


FILES = {"norm/rmsnorm.cu": b"// a\n", "gen/k.cuh": b"// b\n" * 10, "run.sh": b"#!\n"}


@pytest.mark.cpu_only
def test_unpack_round_trip_and_reuse(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    cache = tmp_path / "cache"
    target = packed.ensure_unpacked(archive, manifest, cache)
    assert target.parent == cache and target.name.startswith("1.2.3-")
    for name, data in FILES.items():
        assert (target / name).read_bytes() == data
    assert (target / "run.sh").stat().st_mode & 0o111
    assert not (target / "norm/rmsnorm.cu").stat().st_mode & 0o111
    # second call: no re-extraction (a modified file is left alone)
    (target / "norm/rmsnorm.cu").write_bytes(b"touched")
    assert packed.ensure_unpacked(archive, manifest, cache) == target
    assert (target / "norm/rmsnorm.cu").read_bytes() == b"touched"
    # only the target, its marker and the lock remain
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        [target.name, f"{target.name}.complete", f"{target.name}.lock"]
    )


@pytest.mark.cpu_only
def test_unpack_rejects_corrupt_archive(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    data = bytearray(archive.read_bytes())
    data[-20] ^= 0xFF
    archive.write_bytes(bytes(data))
    with pytest.raises(packed.PackedSourcesError, match="corrupt"):
        packed.ensure_unpacked(archive, manifest, tmp_path / "cache")
    assert not any((tmp_path / "cache").glob("*.complete"))


@pytest.mark.cpu_only
def test_unpack_rejects_file_count_mismatch(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    meta = json.loads(manifest.read_text())
    meta["files"] += 1
    manifest.write_text(json.dumps(meta))
    with pytest.raises(packed.PackedSourcesError, match="holds 3 files"):
        packed.ensure_unpacked(archive, manifest, tmp_path / "cache")
    cache = tmp_path / "cache"
    assert sorted(p.suffix for p in cache.iterdir()) == [".lock"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member",
    [
        "../escape.cu",
        "/abs.cu",
        "a/../../escape.cu",
        "link",
    ],
)
def test_unpack_refuses_unsafe_members(tmp_path, packed, member):
    info = tarfile.TarInfo(member)
    if member == "link":
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
    archive, manifest = _make_archive(tmp_path, packed, FILES, members=[info])
    with pytest.raises(packed.PackedSourcesError, match="refusing archive member"):
        packed.ensure_unpacked(archive, manifest, tmp_path / "cache")
    assert not (tmp_path / "escape.cu").exists()


@pytest.mark.cpu_only
def test_unpack_rejects_bad_manifest(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    manifest.write_text(json.dumps({"format": 99}))
    with pytest.raises(packed.PackedSourcesError, match="unsupported manifest"):
        packed.ensure_unpacked(archive, manifest, tmp_path / "cache")
    manifest.write_text("{not json")
    with pytest.raises(packed.PackedSourcesError, match="cannot read"):
        packed.ensure_unpacked(archive, manifest, tmp_path / "cache")


@pytest.mark.cpu_only
def test_unpack_replaces_an_interrupted_extraction(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    cache = tmp_path / "cache"
    name = packed.unpacked_dir_name(json.loads(manifest.read_text()))
    (cache / name / "norm").mkdir(parents=True)
    (cache / name / "norm" / "rmsnorm.cu").write_bytes(b"half-written")
    target = packed.ensure_unpacked(archive, manifest, cache)
    assert (target / "norm/rmsnorm.cu").read_bytes() == FILES["norm/rmsnorm.cu"]


def _worker(args):
    module_path, archive, manifest, cache = args
    spec = importlib.util.spec_from_file_location("ps_worker", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = module.ensure_unpacked(Path(archive), Path(manifest), Path(cache))
    return str(target), (target / "gen/k.cuh").read_bytes()


@pytest.mark.cpu_only
def test_concurrent_unpack(tmp_path, packed):
    archive, manifest = _make_archive(tmp_path, packed, FILES)
    cache = tmp_path / "cache"
    args = (
        str(_ROOT / "flashinfer" / "jit" / "_packed_sources.py"),
        str(archive),
        str(manifest),
        str(cache),
    )
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        results = pool.map(_worker, [args] * 8)
    assert len({target for target, _ in results}) == 1
    assert all(data == FILES["gen/k.cuh"] for _, data in results)
    assert len(list(cache.glob("*.complete"))) == 1
    assert not [p for p in cache.iterdir() if p.name.startswith(".")]
