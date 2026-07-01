from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def _parse_filters(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Filter {value!r} must use key=value syntax.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Filter {value!r} has an empty key.")
        result[key] = item.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a non-negative, versioned 15-orbit weight specification from a regression table."
    )
    parser.add_argument("--table", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter", action="append", default=[], help="Exact key=value row filter; repeatable.")
    parser.add_argument("--transform", choices=["positive", "absolute", "softplus"], default="positive")
    parser.add_argument("--aggregation", choices=["mean", "median"], default="median")
    parser.add_argument("--normalization", choices=["none", "mean_one", "sum_one"], default="mean_one")
    parser.add_argument("--minimum-sign-consistency", type=float, default=None)
    parser.add_argument("--minimum-nonzero-fraction", type=float, default=None)
    parser.add_argument("--floor", type=float, default=0.0)
    args = parser.parse_args()

    from structure_faithful_gnn.pruning.orbit_weights import (
        derive_orbit_weight_spec_from_table,
        write_orbit_weight_spec,
    )

    spec = derive_orbit_weight_spec_from_table(
        table_path=args.table,
        value_column=args.value_column,
        filters=_parse_filters(args.filter),
        transform=args.transform,
        aggregation=args.aggregation,
        normalization=args.normalization,
        minimum_sign_consistency=args.minimum_sign_consistency,
        minimum_nonzero_fraction=args.minimum_nonzero_fraction,
        floor=args.floor,
    )
    output = write_orbit_weight_spec(args.output, spec)
    print(json.dumps({"output": str(output), "spec": spec.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
