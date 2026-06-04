from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a canonical results table from RelShift and baseline artifacts.")
    parser.add_argument("--relshift-root", required=True)
    parser.add_argument("--dspar-root", required=True)
    parser.add_argument("--lsp-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "analysis"))
    args = parser.parse_args()

    output = build_master_results_table(
        relshift_root=args.relshift_root,
        dspar_root=args.dspar_root,
        lsp_root=args.lsp_root,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2))


def build_master_results_table(
    *,
    relshift_root: str | Path,
    dspar_root: str | Path,
    lsp_root: str | Path | None = None,
    seed: int | None,
    output_root: str | Path,
) -> dict[str, str]:
    from structure_faithful_gnn.analysis.artifacts import (
        MASTER_RESULTS_COLUMNS,
        load_dspar_rows,
        load_lsp_rows,
        load_relshift_rows,
        write_rows_csv,
        write_rows_json,
    )
    from structure_faithful_gnn.utils.io import ensure_dir

    output_root = ensure_dir(output_root)
    rows = load_relshift_rows(relshift_root, seed=seed) + load_dspar_rows(dspar_root, seed=seed)
    if lsp_root:
        rows += load_lsp_rows(lsp_root, seed=seed)
    rows = sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["model"],
            row["method"],
            float(row["achieved_edge_reduction"]),
        ),
    )
    csv_path = output_root / "master_results.csv"
    json_path = output_root / "master_results.json"
    write_rows_csv(csv_path, rows, fieldnames=MASTER_RESULTS_COLUMNS)
    write_rows_json(json_path, rows)
    return {
        "master_results_csv": str(csv_path),
        "master_results_json": str(json_path),
    }


if __name__ == "__main__":
    main()
