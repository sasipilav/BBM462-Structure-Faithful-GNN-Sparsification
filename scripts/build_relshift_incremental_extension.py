from __future__ import annotations

import argparse
import json

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the native RelShift incremental extension for Linux/Colab.")
    parser.add_argument("--force", action="store_true", help="Delete the current hashed build directory before compiling.")
    parser.add_argument("--verbose", action="store_true", help="Show compiler output from torch cpp_extension.load.")
    parser.add_argument("--smoke", action="store_true", help="Run a post-build smoke import and table query.")
    args = parser.parse_args()

    output = build_relshift_incremental_extension(force=args.force, verbose=args.verbose, smoke=args.smoke)
    print(json.dumps(output, indent=2))


def build_relshift_incremental_extension(*, force: bool = False, verbose: bool = False, smoke: bool = False) -> dict[str, object]:
    from structure_faithful_gnn.pruning._incremental_ext import extension_source_path, require_incremental_extension

    module = require_incremental_extension(force_rebuild=force, verbose=verbose)
    result = {
        "source": str(extension_source_path()),
        "module_name": str(module.__name__),
    }
    if smoke:
        tables = module.canonical_tables()
        result["smoke"] = {
            "size2_shape": list(tables["size2"].shape),
            "size3_shape": list(tables["size3"].shape),
            "size4_shape": list(tables["size4"].shape),
        }
    return result


if __name__ == "__main__":
    main()
