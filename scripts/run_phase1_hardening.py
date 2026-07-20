"""Run the three-scenario, multi-seed P1-H suite and generate aggregate reports."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwisim.evaluation.hardening import run_hardening_suite
from mwisim.reporting import HardeningReporter

ROOT = Path(__file__).resolve().parents[1]


def _seed_list(text: str):
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one integer seed is required")
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=_seed_list,
        default=(0, 1, 2),
        help="comma-separated Monte-Carlo seeds (default: 0,1,2)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "docs"),
        help="directory for hardening report artifacts",
    )
    args = parser.parse_args(argv)

    print(f"[P1-H] scenarios=3 seeds={args.seeds}; running full Pipeline suite ...")
    suite = run_hardening_suite(seeds=args.seeds)
    paths = HardeningReporter().write(suite, args.output_dir)
    print("\n[P1-H] mean +/- sample std of chi relative L2")
    for scenario, methods in suite["summary"].items():
        values = []
        for method in ("born", "dbim", "csi"):
            stat = methods[method]["rel_l2"]
            values.append(f"{method.upper()}={stat['mean']:.3f}+/-{stat['std']:.3f}")
        print(f"  {scenario:<24} " + "  ".join(values))
    print("\n[P1-H] acceptance")
    for key, passed in suite["acceptance"].items():
        print(f"  {'PASS' if passed else 'FAIL':<4}  {key}")
    print("\n[P1-H] artifacts")
    for path in paths.values():
        print(" ", path.resolve())
    return suite


if __name__ == "__main__":
    main()
