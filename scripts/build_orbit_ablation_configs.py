from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic uniform, leave-one-orbit-out, and leave-group-out RelShift configs."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--orbit-weight-file", default=None)
    parser.add_argument("--normalization", choices=["none", "mean_one", "sum_one"], default="none")
    parser.add_argument("--skip-orbits", action="store_true")
    parser.add_argument("--skip-groups", action="store_true")
    args = parser.parse_args()

    from structure_faithful_gnn.gdv.orbits import ANALYSIS_ORBIT_GROUPS, ORBIT_DIM
    from structure_faithful_gnn.config import load_yaml

    base_path = Path(args.base_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_yaml(base_path)
    common = dict(base)
    for key in (
        "orbit_weights",
        "orbit_weight_file",
        "orbit_weight_normalization",
        "orbit_leave_out_orbits",
        "orbit_leave_out_groups",
    ):
        common.pop(key, None)
    common["orbit_weight_normalization"] = args.normalization
    if args.orbit_weight_file:
        common["orbit_weight_file"] = str(Path(args.orbit_weight_file).resolve())

    records: list[dict[str, object]] = []

    def write_variant(name: str, payload: dict[str, object]) -> None:
        path = output_dir / f"{_safe_name(name)}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        records.append({"name": name, "path": str(path), "payload": payload})

    write_variant("orbit_uniform", dict(common))
    if not args.skip_orbits:
        for orbit_id in range(ORBIT_DIM):
            payload = dict(common)
            payload["orbit_leave_out_orbits"] = [orbit_id]
            write_variant(f"leave_orbit_{orbit_id:02d}", payload)
    if not args.skip_groups:
        for group_name in ANALYSIS_ORBIT_GROUPS:
            payload = dict(common)
            payload["orbit_leave_out_groups"] = [group_name]
            write_variant(f"leave_group_{group_name}", payload)

    manifest_path = output_dir / "orbit_ablation_manifest.json"
    manifest = {
        "base_config": str(base_path),
        "orbit_weight_file": str(Path(args.orbit_weight_file).resolve()) if args.orbit_weight_file else None,
        "normalization": args.normalization,
        "variant_count": len(records),
        "analysis_groups": {name: list(values) for name, values in ANALYSIS_ORBIT_GROUPS.items()},
        "variants": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "variant_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
