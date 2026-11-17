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

PEP 517 backend of the flashinfer-experimental wheel.

Delegates to ``build_experimental_wheel`` of the repository's build backend,
which builds the full flashinfer-python wheel and extracts the experimental
files from it. Only ``build_wheel`` is provided: the distribution is built from
a repository checkout and published as a wheel (there is no sdist).
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_backend():
    module = sys.modules.get("_flashinfer_repo_build_backend")
    if module is not None:
        return module
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))  # for build_utils
    spec = importlib.util.spec_from_file_location(
        "_flashinfer_repo_build_backend", _REPO_ROOT / "build_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _repo_backend().build_experimental_wheel(wheel_directory, config_settings)
