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
"""

# FlashInfer experimental namespace.
#
# Code under this package implements experimental backends and
# backend-specific logic: support checks, heuristics, routing, compilation,
# caching, and kernels that are not yet ready for stable support. Public
# experimental APIs live in core, marked with @flashinfer_experimental_api;
# only thin entry points in core may hand off to this package.
#
# Everything here provides no compatibility guarantees and may change or be
# removed without deprecation. Using it is an explicit opt-in: calling an
# @flashinfer_experimental_api function, or naming an experimental backend
# with backend="<name>". Only *automatic* selection (backend="auto") is gated,
# behind FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1. Importing this module
# is always allowed (so tooling and docs can introspect it).
#
# See README.md in this directory for the full policy, and AGENTS.md for
# agent-facing contribution rules.

#
# Packaging: release builds split this tree across two distributions.
# ``flashinfer-python`` keeps this module, the Cake training backends under
# ``cake/`` (they back stable facades) and the import-light
# ``cake_nvfp4_per_token`` support checker; every other experimental package
# ships in ``flashinfer-experimental`` (``pip install flashinfer-experimental``
# or ``pip install "flashinfer-python[experimental]"``). Without it, importing
# one of those packages raises ``ModuleNotFoundError`` naming the distribution
# to install. Source checkouts contain the whole tree. See README.md here.

import contextlib
import importlib.util
import os
import sys
import warnings

from ..api_logging import (
    ExperimentalWarning,
    experimental_auto_backends_allowed,
    flashinfer_experimental_api,
    require_experimental_auto_backends,
    warn_experimental_backend_once,
)
from ..utils import experimental_backend

EXPERIMENTAL_DISTRIBUTION = "flashinfer-experimental"

# Shipped by flashinfer-experimental (and present in every source checkout).
_MARKER = __name__ + "._installed"
_installed = None


def experimental_package_installed() -> bool:
    """Whether the experimental packages are available in this environment.

    True for a source checkout and when ``flashinfer-experimental`` is installed
    next to ``flashinfer-python``; False for ``flashinfer-python`` alone.
    """
    global _installed
    if _installed is None:
        try:
            _installed = importlib.util.find_spec(_MARKER) is not None
        except ImportError:
            _installed = False
    return _installed


def missing_experimental_package_message(feature: str) -> str:
    """The error text for ``feature`` when ``flashinfer-experimental`` is missing."""
    from ..version import __version__

    pin = "" if __version__ == "0.0.0+unknown" else f"=={__version__}"
    return (
        f"{feature} is an experimental FlashInfer feature that ships in the "
        f"separate '{EXPERIMENTAL_DISTRIBUTION}' package, which is not installed. "
        f'Install the version matching flashinfer-python: pip install "'
        f'{EXPERIMENTAL_DISTRIBUTION}{pin}" (or pip install '
        f'"flashinfer-python[experimental]{pin}").'
    )


class _MissingExperimentalPackageFinder:
    """Fallback finder that explains a missing ``flashinfer-experimental``.

    It is appended to ``sys.meta_path``, so it runs only after every regular
    finder failed to locate a ``flashinfer.experimental.*`` module. When the
    experimental distribution is absent it raises ``ModuleNotFoundError`` with
    install instructions; otherwise it returns ``None`` and the usual "No module
    named ..." error is reported unchanged.
    """

    _prefix = __name__ + "."

    def find_spec(self, fullname, path=None, target=None):
        if (
            not fullname.startswith(self._prefix)
            or fullname == _MARKER
            or experimental_package_installed()
        ):
            return None
        raise ModuleNotFoundError(
            missing_experimental_package_message(fullname), name=fullname
        )


if not any(
    type(finder).__name__ == _MissingExperimentalPackageFinder.__name__
    for finder in sys.meta_path
):
    sys.meta_path.append(_MissingExperimentalPackageFinder())


def _check_experimental_distribution_version() -> None:
    """Warn when ``flashinfer-experimental`` and ``flashinfer-python`` differ.

    Both install into one package directory, so a mismatch (possible with
    ``pip install --no-deps``) mixes files of two builds.
    """
    if os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"):
        return
    import importlib.metadata

    from ..version import __version__

    try:
        installed = importlib.metadata.version(EXPERIMENTAL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return
    if __version__ != "0.0.0+unknown" and installed != __version__:
        warnings.warn(
            f"{EXPERIMENTAL_DISTRIBUTION} {installed} does not match "
            f"flashinfer-python {__version__}; experimental features may fail. "
            f'Run pip install "{EXPERIMENTAL_DISTRIBUTION}=={__version__}", or set '
            f"FLASHINFER_DISABLE_VERSION_CHECK=1 to silence this warning.",
            stacklevel=3,
        )


# A metadata problem must never break `import flashinfer`.
with contextlib.suppress(Exception):
    _check_experimental_distribution_version()

__all__ = [
    "EXPERIMENTAL_DISTRIBUTION",
    "ExperimentalWarning",
    "experimental_auto_backends_allowed",
    "experimental_backend",
    "experimental_package_installed",
    "flashinfer_experimental_api",
    "missing_experimental_package_message",
    "require_experimental_auto_backends",
    "warn_experimental_backend_once",
]
