"""Run one declarative Phase-1 YAML pipeline and write its benchmark report."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwisim.pipeline import Phase1Pipeline

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "examples" / "phase1_hardening.yaml"),
        help="Phoenix YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "docs" / "phase1_pipeline_run"),
        help="directory for PNG/JSON/Markdown artifacts",
    )
    parser.add_argument("--seed", type=int, default=None, help="override corruption seed")
    args = parser.parse_args(argv)

    print(f"[P1-Pipeline] config={Path(args.config).resolve()}")
    result = Phase1Pipeline(args.config).run(
        seed_override=args.seed,
        output_dir=args.output_dir,
    )
    summary = result["problem_summary"]
    print(
        f"[P1-Pipeline] scene={summary['scene_name']} seed={summary['seed']} "
        f"N={summary['n_cells']} views={summary['n_views']} rx={summary['n_receivers']}"
    )
    print("method     chi-rel-L2      SSIM      data-residual    total-time[s]")
    for key in ("born", "dbim", "csi"):
        record = result["methods"][key]
        metric = record["metrics"]
        print(
            f"{record['name']:<8}   {metric['rel_l2']:>10.4f}   {metric['ssim']:>7.4f}   "
            f"{metric['data_residual']:>13.4e}   {record['runtime_s']:>12.3f}"
        )
    print("[P1-Pipeline] artifacts")
    for path in result.get("artifacts", {}).values():
        print(" ", path)
    return result


if __name__ == "__main__":
    main()
