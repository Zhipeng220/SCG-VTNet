from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


DISPLAY_NAMES = {
    "A": "Global concat",
    "B": "Global CA",
    "C": "Token concat",
    "D": "Token CA",
    "E": "Token CA + gate",
}
SCENARIOS = ["Visual Gaussian", "Token Dropout", "Patch Occlusion", "Joint Noise"]
COLORS = {
    "Global concat": "#64748b",
    "Global CA": "#475569",
    "Token concat": "#0f766e",
    "Token CA": "#2563eb",
    "Token CA + gate": "#b45309",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def default_figures_dir():
    return Path(__file__).resolve().parents[1] / "figures"


def resolve_output_dir(output_dir):
    return Path(output_dir) if output_dir is not None else default_figures_dir()


def draw_figure3a(ax):
    """
    Fig. 3a: Token-preserving visual context vs global pooling.

    This version reduces internal canvas whitespace by using a cropped
    coordinate range while preserving the 8:3 aspect ratio.
    """

    # Tight internal crop.
    # Keep width / height = 8 / 3 for stable one-row multi-panel layout.
    ax.set_xlim(0.55, 11.95)
    ax.set_ylim(0.35, 4.625)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Left: preserved 7x7 ResNet layer4 visual-token grid
    # ------------------------------------------------------------------
    origin_x, origin_y = 0.95, 0.92
    cell = 0.37
    grid_size = 7
    grid_w = grid_size * cell

    rng = np.random.default_rng(7)
    vals = rng.uniform(0.15, 0.95, size=(grid_size, grid_size))

    for y in range(grid_size):
        for x in range(grid_size):
            ax.add_patch(
                Rectangle(
                    (origin_x + x * cell, origin_y + (grid_size - 1 - y) * cell),
                    cell,
                    cell,
                    facecolor=plt.cm.viridis(vals[y, x]),
                    edgecolor="white",
                    linewidth=0.38,
                )
            )

    grid_cx = origin_x + grid_w / 2.0

    ax.text(
        grid_cx,
        origin_y + grid_w + 0.28,
        "7×7 visual tokens",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#111827",
    )

    ax.text(
        grid_cx,
        origin_y - 0.22,
        "ResNet-50 layer4 token grid",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#475569",
    )

    # ------------------------------------------------------------------
    # Upper branch: global pooling
    # ------------------------------------------------------------------
    top_y = 3.55

    ax.add_patch(
        FancyArrowPatch(
            (origin_x + grid_w + 0.40, top_y),
            (6.30, top_y),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.35,
            color="#334155",
        )
    )

    ax.text(
        5.35,
        top_y + 0.20,
        "mean pool",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#111827",
    )

    vec_x, vec_y = 6.52, top_y - 0.41
    ax.add_patch(
        Rectangle(
            (vec_x, vec_y),
            1.12,
            0.82,
            facecolor="#cbd5e1",
            edgecolor="#475569",
            linewidth=1.05,
        )
    )

    ax.text(
        vec_x + 0.56,
        top_y,
        "global\nvector",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#111827",
        linespacing=0.95,
    )

    ax.text(
        vec_x + 0.56,
        vec_y - 0.20,
        "token layout collapsed",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#64748b",
    )

    # ------------------------------------------------------------------
    # Lower branch: token-preserving context access
    # ------------------------------------------------------------------
    bottom_y = 1.10
    preserve_color = "#0f766e"

    ax.add_patch(
        FancyArrowPatch(
            (origin_x + grid_w + 0.40, bottom_y),
            (8.96, bottom_y),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.35,
            color=preserve_color,
        )
    )

    ax.text(
        6.30,
        bottom_y - 0.25,
        "preserve visual tokens",
        ha="center",
        va="top",
        fontsize=9,
        color=preserve_color,
    )

    # Compact token strip on the right
    token_indices = [4, 10, 17, 24, 31, 38, 45]
    token_x = 9.18
    token_y = 0.84
    token_w = 0.25
    token_h = 0.55
    token_gap = 0.08

    for i, idx in enumerate(token_indices):
        row, col = divmod(idx, 7)
        ax.add_patch(
            Rectangle(
                (token_x + i * (token_w + token_gap), token_y),
                token_w,
                token_h,
                facecolor=plt.cm.viridis(vals[row, col]),
                edgecolor="white",
                linewidth=0.32,
            )
        )

    token_total_w = len(token_indices) * token_w + (len(token_indices) - 1) * token_gap
    token_center_x = token_x + token_total_w / 2.0

    ax.text(
        token_center_x,
        token_y + token_h + 0.18,
        "skeleton queries access\ncoarse token context",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#111827",
        linespacing=1.0,
    )


def draw_figure3b(ax):
    """Fig. 3b: Local corruption patterns on the visual-token grid."""

    cell = 0.34
    grid_size = 7
    grid_w = grid_size * cell
    x_positions = [0.35, 3.55, 6.75]

    left_pad = 0.10
    right_pad = 0.12

    ax.set_xlim(x_positions[0] - left_pad, x_positions[-1] + grid_w + right_pad)
    ax.set_ylim(0.85, 3.92)
    ax.axis("off")

    def grid(x0, title, dropout=False, patch=False):
        y0 = 1.05
        vals = np.linspace(0.2, 0.9, 49).reshape(7, 7)

        for y in range(grid_size):
            for x in range(grid_size):
                face = plt.cm.magma(vals[y, x])
                if dropout and (x + y) % 4 == 0:
                    face = "#e2e8f0"
                if patch and 2 <= x <= 4 and 2 <= y <= 4:
                    face = "#111827"

                ax.add_patch(
                    Rectangle(
                        (x0 + x * cell, y0 + (grid_size - 1 - y) * cell),
                        cell,
                        cell,
                        facecolor=face,
                        edgecolor="white",
                        linewidth=0.45,
                    )
                )

        ax.text(
            x0 + grid_w / 2,
            3.65,
            title,
            ha="center",
            va="center",
            fontsize=12,
        )

    grid(x_positions[0], "clean tokens")
    grid(x_positions[1], "token dropout", dropout=True)
    grid(x_positions[2], "patch occlusion", patch=True)


def load_corruption_metrics(metrics_csv=None):
    if metrics_csv is not None:
        candidates = [Path(metrics_csv)]
    else:
        code_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path("figures") / "fig3_corruption_accuracy_drop.csv",
            code_root / "figures" / "fig3_corruption_accuracy_drop.csv",
        ]

    for path in candidates:
        if path.exists():
            metrics = pd.read_csv(path)
            required = {"model", "experiment", "scenario", "value"}
            if "value" not in metrics and "accuracy_drop" in metrics:
                metrics = metrics.rename(columns={"accuracy_drop": "value"})
            missing = required.difference(metrics.columns)
            if missing:
                raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
            return metrics

    raise FileNotFoundError(
        "Could not find fig3_corruption_accuracy_drop.csv. "
        "Pass metrics_csv explicitly or place it in code/figures."
    )


def draw_figure3c(ax, metrics):
    """Fig. 3c: Accuracy drop under corruption for each visual representation."""
    pivot = metrics.pivot(index="scenario", columns="model", values="value").loc[SCENARIOS]
    x = np.arange(len(SCENARIOS))
    width = 0.15
    models = [DISPLAY_NAMES[key] for key in ("A", "B", "C", "D", "E")]
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for idx, model in enumerate(models):
        ax.bar(
            x + offsets[idx],
            pivot[model].to_numpy(),
            width=width,
            color=COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Visual\nGaussian", "Token\nDropout", "Patch\nOcclusion", "Joint\nNoise"],
        fontsize=12
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", pad=6)

    # 改成两行，避免溢出
    ax.set_ylabel("Accuracy drop\n(percentage points)", fontsize=14, labelpad=10)

    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)

    leg = ax.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.4
    )

    return leg


def save_figure3a(output_dir=None):
    apply_style()

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 3), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])

    draw_figure3a(ax)

    stem = "figure3a_token_preserving_vs_global_pooling"
    for ext in ("pdf", "png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.005,
        )

    plt.close(fig)

    print("Saved:")
    for ext in ("pdf", "png", "svg"):
        print(f"  {out_dir / f'{stem}.{ext}'}")


def save_figure3b(output_dir=None):
    apply_style()

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5.4, 1.85), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_figure3b(ax)

    stem = "figure3b_local_visual_corruption"
    saved = []
    for ext in ("pdf", "png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.005)
        saved.append(path)

    plt.close(fig)
    return saved


def save_figure3c(output_dir=None, metrics_csv=None):
    apply_style()

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_corruption_metrics(metrics_csv)

    fig = plt.figure(figsize=(10.8, 4.4), constrained_layout=False)
    ax = fig.add_axes([0.14, 0.18, 0.84, 0.60])
    leg = draw_figure3c(ax, metrics)

    stem = "figure3c_corruption_response"
    saved = []
    for ext in ("pdf", "png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.01,
            bbox_extra_artists=(leg,)
        )
        saved.append(path)

    plt.close(fig)
    return saved


def save_all_figure3_panels(output_dir=None, metrics_csv=None):
    save_figure3a(output_dir)
    saved = []
    saved.extend(save_figure3b(output_dir))
    saved.extend(save_figure3c(output_dir, metrics_csv))
    print("Saved Fig. 3b/c:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot feature robustness from an evaluation CSV.")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    save_all_figure3_panels(args.output_dir, args.metrics_csv)
