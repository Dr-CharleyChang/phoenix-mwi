"""Reproduce the public UM-BMID Gen-One subtraction + ICZT example.

Examples:
    python scripts/run_p2_um_bmid.py --download
    python scripts/run_p2_um_bmid.py --sample-id 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwisim.data.um_bmid import (  # noqa: E402
    UM_BMID_GEN_ONE_DATA_FILE,
    UM_BMID_GEN_ONE_METADATA_FILE,
    UMBMIDDataSource,
    download_um_bmid_gen_one,
    safe_extract_zip,
    verify_extracted_zip_members,
    verify_um_bmid_gen_one_archive,
)
from mwisim.evaluation.measured import reproduce_um_bmid_reference_example  # noqa: E402


def _paths(data_root: Path):
    archive = data_root / "gen-one.zip"
    extracted = data_root / "gen-one"
    return (
        archive,
        extracted,
        extracted / UM_BMID_GEN_ONE_DATA_FILE,
        extracted / UM_BMID_GEN_ONE_METADATA_FILE,
    )


def prepare_dataset(data_root: Path, *, download: bool) -> tuple[Path, Path, dict]:
    archive, extracted, data_path, metadata_path = _paths(data_root)
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"{archive} is missing; rerun with --download or place the official "
                "Zenodo Gen-One archive there"
            )
        download_um_bmid_gen_one(archive)
    verification = verify_um_bmid_gen_one_archive(archive)
    if not data_path.exists() or not metadata_path.exists():
        if extracted.exists() and any(extracted.iterdir()):
            raise FileNotFoundError(
                f"{extracted} is incomplete; move it aside before safe re-extraction"
            )
        safe_extract_zip(archive, extracted)
    verify_extracted_zip_members(archive, extracted)
    return data_path, metadata_path, verification


def write_figure(time_data: np.ndarray, times: np.ndarray, output: Path, title: str):
    magnitude = np.abs(time_data)
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    image = axis.imshow(
        magnitude,
        origin="upper",
        aspect="auto",
        extent=[1, magnitude.shape[1], times[-1] * 1e9, times[0] * 1e9],
        cmap="inferno",
    )
    axis.set_title(title)
    axis.set_xlabel("Clockwise antenna position")
    axis.set_ylabel("Time of response (ns)")
    figure.colorbar(image, ax=axis, label="|normalized inverse Fourier response|")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "external" / "um_bmid",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample-id", type=int, default=1)
    parser.add_argument("--time-points", type=int, default=1024)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs" / "phase2_um_bmid"
    )
    args = parser.parse_args(argv)

    data_path, metadata_path, verification = prepare_dataset(
        args.data_root, download=args.download
    )
    record = UMBMIDDataSource(
        data_path,
        metadata_path,
        generation="one",
        s_parameter="s11",
        trusted_pickle=True,
    ).measurements()
    report, time_data, times = reproduce_um_bmid_reference_example(
        record,
        sample_id=args.sample_id,
        start_time_s=0.0,
        stop_time_s=6e-9,
        n_time_points=args.time_points,
    )
    report["archive"] = verification
    report["record_summary"] = record.summary()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark.json"
    figure_path = output_dir / "sinogram.png"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_figure(
        time_data,
        times,
        figure_path,
        f"UM-BMID Gen-One: sample {report['sample_id']} minus reference {report['reference_id']}",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"report: {report_path.resolve()}")
    print(f"figure: {figure_path.resolve()}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
