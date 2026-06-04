from __future__ import annotations

import hashlib
import platform
import shutil
from pathlib import Path

_EXTENSION = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def extension_source_path() -> Path:
    return _repo_root() / "src" / "structure_faithful_gnn" / "pruning" / "csrc" / "relshift_incremental_ext.cpp"


def supports_native_incremental_extension() -> bool:
    return platform.system().lower() == "linux"


def require_incremental_extension(*, force_rebuild: bool = False, verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None and not force_rebuild:
        return _EXTENSION
    if not supports_native_incremental_extension():
        raise RuntimeError(
            "RelShift native incremental extension is only supported on Linux/Colab in this phase. "
            "Windows is intentionally out of scope."
        )

    from torch.utils.cpp_extension import load

    source_path = extension_source_path()
    if not source_path.exists():
        raise RuntimeError(f"Native extension source file not found: {source_path}")

    source_hash = hashlib.sha1(source_path.read_bytes()).hexdigest()[:12]
    module_name = f"relshift_incremental_ext_{source_hash}"
    build_root = _repo_root() / "tmp" / "relshift_incremental_ext"
    build_dir = build_root / module_name
    if force_rebuild and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    _EXTENSION = load(
        name=module_name,
        sources=[str(source_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG", "-march=native", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=verbose,
        with_cuda=False,
        is_python_module=True,
    )
    return _EXTENSION
