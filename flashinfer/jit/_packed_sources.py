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

Unpacking of the packed JIT source tree of release wheels.

Release wheels built with ``FLASHINFER_PACK_CSRC=1`` (see build_backend.py,
"Packed JIT sources") ship ``flashinfer/data/csrc/**`` as one solid xz archive,
``flashinfer/data/csrc.tar.xz``, plus a manifest, ``flashinfer/data/csrc.json``.
A wheel compresses every file on its own, so the thousands of near-identical
generated kernel sources compress about ten times better as one stream.

``FLASHINFER_CSRC_DIR`` (flashinfer/jit/env.py) resolves to the unpacked tree on
first access: :func:`ensure_unpacked` extracts the archive once into the
FlashInfer cache directory, in a directory keyed by the archive digest, and
every later access (in any process) reuses it. Wheels built without the flag,
source checkouts and editable installs keep a real ``data/csrc`` directory and
never reach this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import shutil
import tarfile
import tempfile

MANIFEST_FORMAT = 1


class PackedSourcesError(RuntimeError):
    """The packed source archive is missing, corrupt or cannot be unpacked."""


def _read_manifest(manifest_path: pathlib.Path) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PackedSourcesError(f"cannot read {manifest_path}: {error}") from error
    if manifest.get("format") != MANIFEST_FORMAT:
        raise PackedSourcesError(
            f"{manifest_path}: unsupported manifest format {manifest.get('format')!r}"
        )
    for key in ("sha256", "version", "files"):
        if key not in manifest:
            raise PackedSourcesError(f"{manifest_path}: missing {key!r}")
    return manifest


def _validated_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Regular files and directories with relative, non-escaping names only."""
    members = archive.getmembers()
    for member in members:
        name = member.name
        parts = pathlib.PurePosixPath(name).parts
        if (
            not name
            or name.startswith("/")
            or ".." in parts
            or not (member.isfile() or member.isdir())
        ):
            raise PackedSourcesError(f"refusing archive member {name!r}")
    return members


def unpacked_dir_name(manifest: dict) -> str:
    # The version keeps directories readable; the digest makes them exact.
    version = str(manifest["version"]).replace(os.sep, "_")
    return f"{version}-{manifest['sha256'][:16]}"


def ensure_unpacked(
    archive_path: pathlib.Path,
    manifest_path: pathlib.Path,
    cache_root: pathlib.Path,
) -> pathlib.Path:
    """Return the directory holding the unpacked archive, extracting it once.

    Safe under concurrency: extraction happens under a file lock into a
    temporary sibling directory that is renamed into place, and the directory
    counts as ready only once its ``.complete`` marker (written after the
    rename) exists, so readers never see a partial tree.
    """
    manifest = _read_manifest(manifest_path)
    name = unpacked_dir_name(manifest)
    target = cache_root / name
    marker = cache_root / f"{name}.complete"
    if marker.is_file():
        return target

    from filelock import FileLock

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PackedSourcesError(
            f"cannot create {cache_root} to unpack FlashInfer's JIT sources "
            f"(set FLASHINFER_WORKSPACE_BASE to a writable directory): {error}"
        ) from error
    with FileLock(str(cache_root / f"{name}.lock"), thread_local=False):
        if marker.is_file():
            return target
        try:
            data = archive_path.read_bytes()
        except OSError as error:
            raise PackedSourcesError(f"cannot read {archive_path}: {error}") from error
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest["sha256"]:
            raise PackedSourcesError(
                f"{archive_path} is corrupt: sha256 {digest} does not match "
                f"{manifest['sha256']} in {manifest_path}; reinstall flashinfer-python"
            )
        staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{name}.", dir=cache_root))
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
                members = _validated_members(archive)
                files = sum(member.isfile() for member in members)
                if files != manifest["files"]:
                    raise PackedSourcesError(
                        f"{archive_path} holds {files} files, the manifest "
                        f"lists {manifest['files']}"
                    )
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(staging, members=members, filter="data")
                else:  # Python < 3.10.12: members were validated above
                    archive.extractall(staging, members=members)
            if target.exists():
                # Left behind by an interrupted extraction (no marker yet).
                shutil.rmtree(target)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        marker.write_text(manifest["sha256"] + "\n", encoding="utf-8")
    return target
