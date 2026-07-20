"""Generate reproducible Markdown, JSON, and PNG artifacts from a P1 benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _metric_payload(result: dict) -> dict:
    return {
        "schema_version": result["schema_version"],
        "generated_at": result["generated_at"],
        "problem_summary": result["problem_summary"],
        "methods": {
            key: {
                "name": value["name"],
                "metrics": value["metrics"],
                "runtime_s": value["runtime_s"],
                "refinement_runtime_s": value["refinement_runtime_s"],
                "info": value["info"],
            }
            for key, value in result["methods"].items()
        },
        "das": {
            "name": result["das"]["name"],
            "metrics": result["das"]["metrics"],
            "runtime_s": result["das"]["runtime_s"],
        },
        "acceptance": result["acceptance"],
    }


def _extent_cm(centers: np.ndarray):
    x = np.unique(np.asarray(centers)[:, 0])
    y = np.unique(np.asarray(centers)[:, 1])
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.median(np.diff(y))) if y.size > 1 else 1.0
    return [100 * (x[0] - dx / 2), 100 * (x[-1] + dx / 2), 100 * (y[0] - dy / 2), 100 * (y[-1] + dy / 2)]


class BenchmarkReporter:
    """Render a ``run_phase1_benchmark`` result into human- and machine-readable files."""

    def __init__(self, dpi: int = 150):
        self.dpi = int(dpi)

    def _save_method_figure(self, result: dict, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shape = tuple(result["grid_shape"])
        problem = result["problem"]
        centers = np.asarray(problem["centers"])
        extent = _extent_cm(centers)
        truth = np.asarray(problem["chi_true"]).real.reshape(shape)
        vmax = float(max(np.max(truth), np.finfo(float).eps))
        panels = [
            ("True contrast chi", truth, "viridis", 0.0, vmax),
            ("DAS intensity", result["das"]["image"].reshape(shape), "inferno", 0.0, 1.0),
        ]
        for key in ("born", "dbim", "csi"):
            record = result["methods"][key]
            panels.append(
                (
                    f"{record['name']} chi_hat",
                    np.asarray(record["estimate"]).real.reshape(shape),
                    "viridis",
                    0.0,
                    vmax,
                )
            )
        fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.8), constrained_layout=True)
        for ax, (title, image, cmap, vmin, vmax_panel) in zip(axes, panels):
            shown = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_panel,
            )
            ax.set_title(title)
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("y [cm]")
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Phoenix Phase-1: qualitative preview and quantitative reconstructions", fontweight="bold")
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def _save_residual_figure(self, result: dict, path: Path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = ("born", "dbim", "csi")
        labels = [result["methods"][key]["name"] for key in keys]
        residuals = [result["methods"][key]["metrics"]["data_residual"] for key in keys]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        axes[0].bar(labels, residuals, color=["C0", "C1", "C2"])
        axes[0].set_yscale("log")
        axes[0].set_ylabel("||F(chi_hat) - d|| / ||d||")
        axes[0].set_title("Common full-forward data residual")
        axes[0].grid(True, axis="y", which="both", alpha=0.3)

        born_res = result["methods"]["born"]["metrics"]["data_residual"]
        dbim_hist = result["methods"]["dbim"]["info"].get("res_history", [])
        csi_hist = result["methods"]["csi"]["info"].get("data_res_history", [])
        if dbim_hist:
            axes[1].semilogy(range(len(dbim_hist)), dbim_hist, "o-", label="DBIM full-forward residual")
        if csi_hist:
            axes[1].semilogy(range(len(csi_hist)), csi_hist, "s-", label="CSI source-data residual")
        axes[1].axhline(born_res, color="C3", linestyle="--", label="Born final full-forward residual")
        axes[1].set_xlabel("outer iteration")
        axes[1].set_ylabel("relative residual")
        axes[1].set_title("Internal convergence histories")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def _write_markdown(self, result: dict, path: Path, method_figure: Path, residual_figure: Path):
        summary = result["problem_summary"]
        lines = [
            "# Phoenix Phase-1 benchmark report",
            "",
            f"> Generated automatically at `{result['generated_at']}` by `python scripts/run_phase1_benchmark.py`.",
            "",
            "## Problem",
            "",
            f"All methods used the same full-wave synthetic data: frequency `{summary['frequency_hz'] / 1e9:.3g} GHz`, grid `{summary['grid_shape'][0]} x {summary['grid_shape'][1]}` (`{summary['n_cells']}` cells), `{summary['n_views']}` plane-wave views, `{summary['n_receivers']}` receivers, and maximum true relative permittivity `{summary['max_true_eps_r_for_eps_b_1']:.3g}` for the Phase-1 vacuum background.",
            "",
            "DBIM and CSI use the already-computed Born map as a warm start. Their total runtime includes that shared Born time; `refine time` shows only the nonlinear refinement.",
            "",
            "## Quantitative results",
            "",
            "| Method | chi rel-L2 | eps_r RMSE | SSIM | Localization [mm] | Support IoU | Contrast recovery | Full data residual | Total time [s] | Refine time [s] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for key in ("born", "dbim", "csi"):
            record = result["methods"][key]
            metric = record["metrics"]
            lines.append(
                f"| {record['name']} | {metric['rel_l2']:.4f} | {metric['eps_r_rmse']:.4f} | {metric['ssim']:.4f} | {1000 * metric['localization_error_m']:.3f} | {metric['support_iou']:.4f} | {100 * metric['contrast_recovery']:.1f}% | {metric['data_residual']:.4e} | {record['runtime_s']:.3f} | {record['refinement_runtime_s']:.3f} |"
            )
        das = result["das"]
        lines.extend([
            "",
            "DAS is a normalized qualitative energy image, not an estimate of chi, so RMSE and contrast recovery are intentionally not reported for it.",
            "",
            "| Imager | SSIM | Localization [mm] | Support IoU | Runtime [s] |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| DAS | {das['metrics']['ssim']:.4f} | {1000 * das['metrics']['localization_error_m']:.3f} | {das['metrics']['support_iou']:.4f} | {das['runtime_s']:.4f} |",
            "",
            f"![Phase-1 method comparison]({method_figure.name})",
            "",
            f"![Phase-1 residual comparison]({residual_figure.name})",
            "",
            "## Acceptance gates",
            "",
        ])
        labels = {
            "dbim_data_fit_better_than_born": "DBIM full-forward data fit is better than Born",
            "csi_data_fit_better_than_born": "CSI full-forward data fit is better than Born",
            "das_localizes_inside_true_object": "DAS centroid lands inside the true object",
            "all_outputs_finite": "All reconstructed arrays are finite",
        }
        for key, passed in result["acceptance"].items():
            lines.append(f"- [{'x' if passed else ' '}] {labels.get(key, key)}")
        lines.extend([
            "",
            "## How to read the residual figure",
            "",
            "The left panel is the fair comparison: every method is re-simulated through the same nonlinear forward map and scored as `||F(chi_hat)-d||/||d||`. The right panel exposes each algorithm's own convergence history. DBIM's history is already a full-forward residual, while CSI's history is `||d-SW||/||d||`; those two curves diagnose their own algorithms but must not be compared point-for-point.",
            "",
            "## Scope",
            "",
            "This is a reproducible 2-D, single-frequency, synthetic-data platform benchmark. It validates software wiring and controlled inverse behavior; it is not evidence of clinical diagnostic performance. Phase 2 must add realistic dispersive tissue, calibration, noise/artifacts, and a public measured-data benchmark.",
        ])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write(self, result: dict, output_dir) -> dict:
        """Write two PNGs, one JSON scorecard, and one Markdown report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        method_figure = output_dir / "fig_phase1_methods.png"
        residual_figure = output_dir / "fig_phase1_residuals.png"
        metrics_json = output_dir / "phase1_benchmark_metrics.json"
        report_md = output_dir / "phase1_benchmark_report.md"
        self._save_method_figure(result, method_figure)
        self._save_residual_figure(result, residual_figure)
        metrics_json.write_text(
            json.dumps(_json_safe(_metric_payload(result)), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._write_markdown(result, report_md, method_figure, residual_figure)
        return {
            "method_figure": method_figure,
            "residual_figure": residual_figure,
            "metrics_json": metrics_json,
            "report_md": report_md,
        }


__all__ = ["BenchmarkReporter"]
