"""Run the P2-B UM-BMID Gen-One DAS/ORR/artifact-removal benchmark.

Examples:
    python scripts/run_p2b_measured_imaging.py
    python scripts/run_p2b_measured_imaging.py --download
    python scripts/run_p2b_measured_imaging.py --n-pixels 32 --orr-iterations 15
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
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwisim.data.schema import json_safe  # noqa: E402
from mwisim.data.um_bmid import (  # noqa: E402
    UM_BMID_GEN_ONE_DATA_FILE,
    UM_BMID_GEN_ONE_METADATA_FILE,
    UMBMIDDataSource,
    download_um_bmid_gen_one,
    safe_extract_zip,
    verify_extracted_zip_members,
    verify_um_bmid_gen_one_archive,
)
from mwisim.evaluation.measured_benchmark import (  # noqa: E402
    DEFAULT_P2B_CALIBRATION_IDS,
    DEFAULT_P2B_EVALUATION_IDS,
    DEFAULT_P2B_SPEEDS_M_S,
    run_p2b_benchmark,
)


METHOD_LABELS = {
    "das_empty_reference": "DAS: empty reference",
    "das_empty_plus_angular_mean": "DAS: empty + angular mean",
    "das_empty_plus_low_rank_1": "DAS: empty + rank-1 SVD",
    "das_adipose_reference": "DAS: adipose reference",
    "das_healthy_reference": "DAS: healthy reference",
    "orr_healthy_reference": "ORR: healthy reference",
}


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


def write_reconstruction_figure(report: dict, images: dict, grid, output: Path) -> None:
    sample_id = report["evaluation_ids"][0]
    methods = list(METHOD_LABELS)
    row_lookup = {
        row["method"]: row
        for row in report["per_scan"]
        if row["sample_id"] == sample_id
    }
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), constrained_layout=True)
    image_artist = None
    for axis, method in zip(axes.ravel(), methods):
        image = images[(sample_id, method)]
        image_artist = axis.imshow(
            image,
            origin="lower",
            extent=np.asarray(grid.extent_m) * 100.0,
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        row = row_lookup[method]
        truth = Circle(
            (row["tumor_x_m"] * 100.0, row["tumor_y_m"] * 100.0),
            row["tumor_radius_m"] * 100.0,
            fill=False,
            color="cyan",
            linewidth=1.8,
            label="documented target",
        )
        axis.add_patch(truth)
        axis.plot(
            row["peak_x_m"] * 100.0,
            row["peak_y_m"] * 100.0,
            marker="x",
            color="white",
            markersize=7,
            markeredgewidth=1.6,
            label="image peak",
        )
        axis.set_title(
            f"{METHOD_LABELS[method]}\nLE={row['localization_error_m'] * 100:.1f} cm, "
            f"SCR={row['signal_to_clutter_db']:.1f} dB"
        )
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.set_aspect("equal")
    if image_artist is not None:
        figure.colorbar(
            image_artist, ax=axes.ravel().tolist(), label="normalized image intensity"
        )
    figure.suptitle(
        f"UM-BMID Gen-One P2-B, held-out sample {sample_id}\n"
        "cyan circle: metadata target; white x: image maximum",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_sensitivity_figure(report: dict, output: Path) -> None:
    speed_rows = report["speed_sensitivity"]
    speed = np.asarray([row["propagation_speed_m_s"] for row in speed_rows]) / 1e8
    median = np.asarray([row["localization_error_median_m"] for row in speed_rows]) * 100
    mean = np.asarray([row["localization_error_mean_m"] for row in speed_rows]) * 100
    methods = list(METHOD_LABELS)
    aggregate = report["aggregate"]
    localization = np.asarray(
        [aggregate[method]["localization_error_median_m"] for method in methods]
    ) * 100
    localized_fraction = np.asarray(
        [aggregate[method]["localized_fraction"] for method in methods]
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(speed, median, "o-", label="median LE")
    axes[0].plot(speed, mean, "s--", label="mean LE")
    axes[0].axvline(
        report["selected_propagation_speed_m_s"] / 1e8,
        color="black",
        linestyle=":",
        label="selected globally",
    )
    axes[0].set_xlabel(r"homogeneous speed ($10^8$ m/s)")
    axes[0].set_ylabel("calibration localization error (cm)")
    axes[0].set_title("Calibration-only speed sensitivity")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x_positions = np.arange(len(methods))
    bars = axes[1].bar(x_positions, localization, color="tab:blue", alpha=0.85)
    axes[1].axhline(3.0, color="black", linestyle=":", label="3 cm median gate")
    axes[1].set_xticks(
        x_positions,
        [METHOD_LABELS[method].replace(": ", "\n") for method in methods],
        rotation=25,
        ha="right",
    )
    axes[1].set_ylabel("held-out median localization error (cm)")
    axes[1].set_title("Reference/artifact/algorithm ablation")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, fraction in zip(bars, localized_fraction):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{fraction:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_markdown_report(report: dict, output: Path) -> None:
    lines = [
        "# P2-B UM-BMID measured-imaging benchmark report",
        "",
        f"**Scope:** {report['scope']}.",
        "",
        f"The benchmark used calibration scan IDs {report['calibration_ids']} and disjoint held-out evaluation scan IDs {report['evaluation_ids']}. It selected one global homogeneous propagation speed of {report['selected_propagation_speed_m_s'] / 1e8:.2f}×10⁸ m/s from the declared candidate grid; it did not tune speed per scan. Frequencies were restricted to {report['frequency_selection']['minimum_hz'] / 1e9:.1f}–{report['frequency_selection']['maximum_hz'] / 1e9:.1f} GHz and deterministically decimated to {report['frequency_selection']['n_points']} points. The phase-centre radial offset was fixed at {report['radial_phase_center_offset_m'] * 1000:.1f} mm.",
        "",
        "## Held-out aggregate results",
        "",
        "| Method | n | Median LE (cm) | Mean LE (cm) | Localized fraction | Median SCR (dB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_LABELS:
        row = report["aggregate"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['n_scans']} | {row['localization_error_median_m'] * 100:.2f} | {row['localization_error_mean_m'] * 100:.2f} | {row['localized_fraction']:.1%} | {row['signal_to_clutter_median_db']:.2f} |"
        )
    gate = report["acceptance_gate"]
    lines.extend(
        [
            "",
            "## Acceptance gate",
            "",
            f"The predeclared spatial gate is evaluated on `{gate['method']}` over held-out scans: median localization error ≤ {gate['median_localization_limit_m'] * 100:.1f} cm and localized fraction ≥ {gate['localized_fraction_minimum']:.0%}. Result: **{'PASS' if gate['pass'] else 'FAIL'}**.",
            "",
            "## Interpretation boundary",
            "",
            "This report validates a coordinate-aware qualitative radar reconstruction workflow on controlled breast phantoms. Healthy-reference subtraction is an experimental oracle because it uses a matched scan of the same phantom without the tumour; it is not available in ordinary clinical use. ORR reconstructs a radar reflectivity proxy under a homogeneous, primary-scatter model, not the dielectric contrast χ and not a diagnosis. A lower localization error or higher SCR here must not be translated into clinical sensitivity or specificity.",
            "",
            "## Reproduction",
            "",
            "Run `python scripts/run_p2b_measured_imaging.py`. The script verifies the pinned Gen-One archive, uses metadata-linked references, records every parameter in JSON, and regenerates this report and both figures.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "external" / "um_bmid",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--n-pixels", type=int, default=36)
    parser.add_argument("--max-frequency-points", type=int, default=51)
    parser.add_argument("--orr-iterations", type=int, default=25)
    parser.add_argument("--orr-regularization", type=float, default=1e-4)
    parser.add_argument("--radial-offset-mm", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs" / "phase2b_um_bmid"
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
    report, images, grid = run_p2b_benchmark(
        record,
        calibration_ids=DEFAULT_P2B_CALIBRATION_IDS,
        evaluation_ids=DEFAULT_P2B_EVALUATION_IDS,
        candidate_speeds_m_s=DEFAULT_P2B_SPEEDS_M_S,
        max_frequency_points=args.max_frequency_points,
        n_pixels=args.n_pixels,
        radial_offset_m=args.radial_offset_mm * 1e-3,
        orr_iterations=args.orr_iterations,
        orr_regularization=args.orr_regularization,
    )
    report["archive"] = verification
    report["record_summary"] = record.summary()
    report["sources"] = {
        "dataset": "https://doi.org/10.5281/zenodo.5120981",
        "dataset_code": "https://github.com/UManitoba-BMS/UM-BMID",
        "orr_paper": "https://doi.org/10.3390/s21248172",
        "orr_code": "https://github.com/TysonReimer/ORR-Algorithm",
    }
    report = json_safe(report)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    report_path = output_dir / "report.md"
    reconstruction_path = output_dir / "reconstructions.png"
    sensitivity_path = output_dir / "sensitivity-and-ablation.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_reconstruction_figure(report, images, grid, reconstruction_path)
    write_sensitivity_figure(report, sensitivity_path)
    write_markdown_report(report, report_path)
    print(json.dumps({
        "pass": report["pass"],
        "selected_propagation_speed_m_s": report["selected_propagation_speed_m_s"],
        "acceptance_gate": report["acceptance_gate"],
        "aggregate": report["aggregate"],
        "output_dir": str(output_dir.resolve()),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
