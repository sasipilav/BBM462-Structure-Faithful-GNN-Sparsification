from __future__ import annotations

import argparse
import json

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join exact RelShift orbit checkpoints with matched dense GNN outcomes."
    )
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--dense-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "phase2_orbit_sensitivity" / "dataset"),
    )
    parser.add_argument(
        "--allow-missing-controls",
        action="store_true",
        help="Use zero only for legacy artifacts missing Step-4 controls. Not suitable for final analysis.",
    )
    args = parser.parse_args()

    from structure_faithful_gnn.analysis.orbit_sensitivity import (
        build_orbit_sensitivity_dataset,
    )

    result = build_orbit_sensitivity_dataset(
        training_root=args.training_root,
        dense_root=args.dense_root,
        output_dir=args.output_dir,
        require_full_controls=not args.allow_missing_controls,
    )
    print(
        json.dumps(
            {
                "row_count": len(result.rows),
                "csv_path": str(result.csv_path),
                "json_path": str(result.json_path),
                "manifest_path": str(result.manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
