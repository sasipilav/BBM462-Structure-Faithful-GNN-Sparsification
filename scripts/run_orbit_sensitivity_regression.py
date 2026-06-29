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
        description="Run leakage-safe controlled orbit-sensitivity regressions."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "phase2_orbit_sensitivity" / "regression"),
    )
    parser.add_argument(
        "--target",
        choices=["accuracy_delta", "macro_f1_delta", "accuracy_loss", "macro_f1_loss"],
        default="accuracy_loss",
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        default=["dataset", "model", "reduction_band", "pruning_seed"],
        choices=["dataset", "model", "reduction_band", "pruning_seed", "pruning_artifact"],
    )
    parser.add_argument(
        "--orbit-feature-family",
        default="standardized_absolute",
        choices=[
            "raw_relative_absolute",
            "standardized_absolute",
            "standardized_mean_absolute",
            "standardized_l2",
            "transition_destroyed",
            "transition_out",
        ],
    )
    parser.add_argument("--estimators", nargs="+", default=["ridge", "elastic_net"])
    parser.add_argument("--artifact-folds", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=32)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    from structure_faithful_gnn.analysis.orbit_sensitivity import (
        run_controlled_orbit_regression,
    )

    result = run_controlled_orbit_regression(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        target=args.target,
        split_modes=args.split_modes,
        orbit_feature_family=args.orbit_feature_family,
        estimators=args.estimators,
        artifact_folds=args.artifact_folds,
        permutation_repeats=args.permutation_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "summary_path": str(result.summary_path),
                "manifest_path": str(result.manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
