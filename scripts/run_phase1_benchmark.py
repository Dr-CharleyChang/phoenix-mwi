"""Run the complete P1-A/B/C vertical slice and generate its report artifacts.

Run from the repository root:

    python scripts/run_phase1_benchmark.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwisim.evaluation.benchmark import run_phase1_benchmark
from mwisim.reporting import BenchmarkReporter

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def main():
    print("[P1] building one full-wave synthetic problem and running DAS/Born/DBIM/CSI ...")
    result = run_phase1_benchmark()
    paths = BenchmarkReporter().write(result, DOCS)
    print("\n[P1] common full-forward scorecard")
    print("method     chi-rel-L2      SSIM      data-residual    total-time[s]")
    for key in ("born", "dbim", "csi"):
        record = result["methods"][key]
        metric = record["metrics"]
        print(
            f"{record['name']:<8}   {metric['rel_l2']:>10.4f}   {metric['ssim']:>7.4f}   "
            f"{metric['data_residual']:>13.4e}   {record['runtime_s']:>12.3f}"
        )
    print("\n[P1] acceptance gates")
    for name, passed in result["acceptance"].items():
        print(f"  {'PASS' if passed else 'FAIL':<4}  {name}")
    print("\n[P1] generated artifacts")
    for path in paths.values():
        print(" ", os.path.abspath(path))


if __name__ == "__main__":
    main()
