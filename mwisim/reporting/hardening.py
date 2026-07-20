"""Figures, JSON, and Markdown reporting for the P1-H Monte-Carlo suite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .report import _extent_cm, _json_safe


class HardeningReporter:
    """Render representative maps and mean-plus-standard-deviation statistics."""

    def __init__(self, dpi: int = 150):
        self.dpi = int(dpi)

    def _save_examples(self, suite: dict, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scenarios = list(suite["representatives"])
        columns = ("truth", "das", "born", "dbim", "csi")
        fig, axes = plt.subplots(
            len(scenarios),
            len(columns),
            figsize=(16, 3.2 * len(scenarios)),
            constrained_layout=True,
            squeeze=False,
        )
        for row, scenario in enumerate(scenarios):
            result = suite["representatives"][scenario]
            shape = tuple(result["grid_shape"])
            centers = result["problem"]["centers"]
            extent = _extent_cm(centers)
            truth = np.asarray(result["problem"]["chi_true"]).real.reshape(shape)
            vmax = float(max(np.max(truth), np.finfo(float).eps))
            panels = [
                (truth, "true chi", "viridis", vmax),
                (result["das"]["image"].reshape(shape), "DAS", "inferno", 1.0),
                (result["methods"]["born"]["estimate"].real.reshape(shape), "Born", "viridis", vmax),
                (result["methods"]["dbim"]["estimate"].real.reshape(shape), "DBIM", "viridis", vmax),
                (result["methods"]["csi"]["estimate"].real.reshape(shape), "CSI", "viridis", vmax),
            ]
            for col, (image, title, cmap, panel_max) in enumerate(panels):
                ax = axes[row, col]
                shown = ax.imshow(
                    image,
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=panel_max,
                )
                ax.set_title(f"{scenario}: {title}")
                ax.set_xlabel("x [cm]")
                ax.set_ylabel("y [cm]")
                fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Phase-1 hardening representative maps (seed {suite['seeds'][0]})",
            fontweight="bold",
        )
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def _save_statistics(self, suite: dict, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scenarios = list(suite["summary"])
        quantitative = ("born", "dbim", "csi")
        all_methods = ("das", "born", "dbim", "csi")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

        def grouped_bars(ax, methods, metric, title, log=False):
            x = np.arange(len(scenarios), dtype=float)
            width = 0.8 / len(methods)
            for index, method in enumerate(methods):
                means, stds = [], []
                for scenario in scenarios:
                    entry = suite["summary"][scenario].get(method, {}).get(metric)
                    means.append(np.nan if entry is None else entry["mean"])
                    stds.append(0.0 if entry is None else entry["std"])
                offset = (index - (len(methods) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=method.upper(),
                )
            ax.set_xticks(x, scenarios, rotation=12)
            ax.set_title(title)
            ax.grid(True, axis="y", which="both", alpha=0.3)
            if log:
                ax.set_yscale("log")
            ax.legend(fontsize=8)

        grouped_bars(axes[0, 0], quantitative, "rel_l2", "Contrast relative L2 (lower)")
        grouped_bars(
            axes[0, 1], quantitative, "data_residual", "Full data residual (lower)", log=True
        )
        grouped_bars(axes[1, 0], all_methods, "ssim", "SSIM (higher)")
        grouped_bars(axes[1, 1], all_methods, "support_iou", "Support IoU (higher)")
        fig.suptitle(
            f"Phase-1 hardening Monte-Carlo statistics: mean +/- sample std, n={len(suite['seeds'])}",
            fontweight="bold",
        )
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def _write_markdown(self, suite: dict, path: Path, examples: Path, statistics: Path):
        lines = [
            "# Phoenix Phase-1 hardening report",
            "",
            f"> Generated automatically at {suite['generated_at']} from seeds {suite['seeds']}.",
            "",
            "This report evaluates off-centre, dual-target, and nested heterogeneous scenes under controlled complex noise and receiver-position model error. Every seed is a complete scene → data → DAS/Born/DBIM/CSI → common-score Pipeline run.",
            "",
            f"![Representative reconstructions]({examples.name})",
            "",
            f"![Monte-Carlo statistics]({statistics.name})",
            "",
            "## Mean and sample standard deviation",
            "",
        ]
        for scenario, methods in suite["summary"].items():
            corruption = suite["scenario_specs"][scenario]["corruption"]
            lines.extend(
                [
                    f"### {scenario}",
                    "",
                    f"Requested corruption: SNR {corruption.get('snr_db')} dB and per-coordinate receiver-position standard deviation {1000 * float(corruption.get('receiver_position_std_m', 0.0)):.3f} mm.",
                    "",
                    "| Method | chi rel-L2 | SSIM | Support IoU | Components error | Full data residual | Runtime [s] |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for method in ("das", "born", "dbim", "csi"):
                metric = methods[method]

                def pm(name, digits=4):
                    if name not in metric:
                        return "—"
                    item = metric[name]
                    return f"{item['mean']:.{digits}f} ± {item['std']:.{digits}f}"

                lines.append(
                    f"| {method.upper()} | {pm('rel_l2')} | {pm('ssim')} | {pm('support_iou')} | {pm('component_count_error', 2)} | {pm('data_residual')} | {pm('runtime_s', 3)} |"
                )
            lines.append("")
        lines.extend(
            [
                "## Suite acceptance",
                "",
            ]
        )
        for key, passed in suite["acceptance"].items():
            lines.append(f"- [{'x' if passed else ' '}] {key}")
        lines.extend(
            [
                "",
                "## Interpretation boundary",
                "",
                "Mean describes typical performance across the selected random seeds; sample standard deviation describes seed sensitivity. Three seeds are an engineering smoke test, not a publication-grade uncertainty estimate. A paper should normally use more seeds and confidence intervals chosen before examining results.",
                "",
                "These are still idealized 2-D synthetic experiments. Noise and receiver-coordinate mismatch make them harder and more honest than the original centered cylinder, but they do not replace antenna calibration, dispersive tissue, skin/clutter artifacts, sparse clinical arrays, or measured-data validation.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write(self, suite: dict, output_dir) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = output_dir / "fig_phase1_hardening_examples.png"
        statistics = output_dir / "fig_phase1_hardening_statistics.png"
        metrics_json = output_dir / "phase1_hardening_metrics.json"
        report_md = output_dir / "phase1_hardening_report.md"
        self._save_examples(suite, examples)
        self._save_statistics(suite, statistics)
        payload = {
            key: value
            for key, value in suite.items()
            if key not in ("representatives",)
        }
        metrics_json.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._write_markdown(suite, report_md, examples, statistics)
        return {
            "examples_figure": examples,
            "statistics_figure": statistics,
            "metrics_json": metrics_json,
            "report_md": report_md,
        }


__all__ = ["HardeningReporter"]
