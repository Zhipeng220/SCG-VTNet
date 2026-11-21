import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

from scg_vtnet.evaluation.figure_components import (
    ANGLE_TRIPLETS,
    DEFAULT_CSV_PATH,
    DEFAULT_IMAGE_ROOT,
    PARENT_MAP,
    apply_paper_style,
    column_minmax,
    compute_descriptor_groups,
    coords_from_frame,
    draw_skeleton,
    image_axis,
    load_image_for_row,
    normalized_to_pixel,
    parse_pairs,
    read_landmark_csv,
    save_figure,
)


def choose_sample(df, label):
    subset = df[df["label"].astype(str) == str(label)]
    if "hand_detected" in subset.columns:
        detected = subset[subset["hand_detected"].fillna(0).astype(float) > 0]
        if len(detected) > 0:
            subset = detected
    if len(subset) == 0:
        raise ValueError(f"No sample found for label {label}")
    return subset.iloc[0]


def draw_aar_components(ax, xy):
    draw_descriptor_schematic(ax, xy, mode="combined")
    legend_handles = [
        Line2D([0], [0], color="#2563eb", linewidth=2.0, label="bone vectors"),
        Line2D([0], [0], color="#f97316", linewidth=2.0, label="joint angles"),
        Line2D([0], [0], color="#7c3aed", linewidth=2.0, linestyle=(0, (3, 3)),
               label="thumb-to-fingertip distances"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.82,
              facecolor="white", edgecolor="#cbd5e1", borderpad=0.35, handlelength=1.8)


def set_schematic_view(ax, xy, title=None):
    pad = 28
    x_min, y_min = xy[:, :2].min(axis=0) - pad
    x_max, y_max = xy[:, :2].max(axis=0) + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#ffffff")
    if title:
        ax.set_title(title, pad=7, fontweight="bold")


def draw_highlight_path(ax, xy, joints, color="#0f766e"):
    for left, right in zip(joints[:-1], joints[1:]):
        ax.plot(
            [xy[left, 0], xy[right, 0]],
            [xy[left, 1], xy[right, 1]],
            color=color,
            linewidth=2.6,
            solid_capstyle="round",
            zorder=4,
        )
    ax.scatter(
        xy[joints, 0],
        xy[joints, 1],
        s=34,
        c=color,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )


def add_angle_arc(ax, xy, a=5, b=6, c=7, color="#f97316", scale=0.42):
    radius = scale * (np.linalg.norm(xy[a] - xy[b]) + np.linalg.norm(xy[c] - xy[b]))
    theta1 = np.degrees(np.arctan2(*(xy[a] - xy[b])[::-1]))
    theta2 = np.degrees(np.arctan2(*(xy[c] - xy[b])[::-1]))
    arc = Arc(
        xy[b],
        radius,
        radius,
        angle=0,
        theta1=min(theta1, theta2),
        theta2=max(theta1, theta2),
        color=color,
        linewidth=2.4,
        alpha=0.95,
        zorder=6,
    )
    ax.add_patch(arc)
    return radius


def draw_descriptor_schematic(ax, xy, mode, title=None):
    set_schematic_view(ax, xy, title)
    draw_skeleton(ax, xy, color="#94a3b8", linewidth=1.0, point_size=10, labels=False, alpha=0.34)

    index_chain = [5, 6, 7, 8]
    draw_highlight_path(ax, xy, index_chain, color="#0f766e")

    if mode in ("bone", "combined"):
        start, end = xy[6], xy[7]
        delta = end - start
        ax.arrow(
            start[0],
            start[1],
            delta[0] * 0.84,
            delta[1] * 0.84,
            width=0.65,
            head_width=6.5,
            head_length=7.5,
            color="#2563eb",
            alpha=0.95,
            length_includes_head=True,
            zorder=7,
        )
        if mode == "bone":
            mid = start + delta * 0.52
            ax.text(
                mid[0] + 8,
                mid[1] - 8,
                r"$\vec{b}_{7}=p_{7}-p_{6}$",
                color="#1d4ed8",
                fontsize=9,
                fontweight="bold",
                zorder=8,
            )

    if mode in ("angle", "combined"):
        radius = add_angle_arc(ax, xy, 5, 6, 7)
        if mode == "angle":
            ax.text(
                xy[6, 0] + radius * 0.55,
                xy[6, 1] - radius * 0.28,
                r"$\theta_{6}$",
                color="#ea580c",
                fontsize=11,
                fontweight="bold",
                zorder=8,
            )

    if mode in ("distance", "combined"):
        thumb_tip = xy[4]
        for tip_idx in (8, 12):
            ax.plot(
                [thumb_tip[0], xy[tip_idx, 0]],
                [thumb_tip[1], xy[tip_idx, 1]],
                linestyle=(0, (4, 3)),
                color="#7c3aed",
                linewidth=2.0 if tip_idx == 8 else 1.65,
                alpha=0.9 if tip_idx == 8 else 0.72,
                zorder=6,
            )
        ax.scatter(
            [thumb_tip[0], xy[8, 0], xy[12, 0]],
            [thumb_tip[1], xy[8, 1], xy[12, 1]],
            s=42,
            c="#7c3aed",
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
        )
        if mode == "distance":
            midpoint = 0.5 * (thumb_tip + xy[8])
            ax.text(
                midpoint[0] + 8,
                midpoint[1] - 8,
                r"$d(thumb,index)$",
                color="#6d28d9",
                fontsize=9,
                fontweight="bold",
                zorder=8,
            )


def zscore_group(values):
    flat = values.reshape(values.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    return (flat - mean) / std


def descriptor_distance_table(df, pairs):
    coords = coords_from_frame(df)
    if "hand_detected" in df.columns:
        mask = df["hand_detected"].fillna(0).astype(float).to_numpy() > 0
        coords = coords[mask]
        labels = df.loc[mask, "label"].astype(str).to_numpy()
    else:
        labels = df["label"].astype(str).to_numpy()

    groups = compute_descriptor_groups(coords)
    standardized_groups = {name: zscore_group(values) for name, values in groups.items()}
    rows = []
    for left, right in pairs:
        row = {"pair": f"{left} vs {right}"}
        left_mask = labels == str(left)
        right_mask = labels == str(right)
        if not np.any(left_mask) or not np.any(right_mask):
            raise ValueError(f"Missing label samples for pair {left}:{right}")
        for group_name, flat in standardized_groups.items():
            left_mean = flat[left_mask].mean(axis=0)
            right_mean = flat[right_mask].mean(axis=0)
            row[group_name] = float(np.linalg.norm(left_mean - right_mean))
        rows.append(row)

    raw = pd.DataFrame(rows)
    group_cols = ["coordinates", "bone vectors", "joint angles", "fingertip distances"]
    normalized = raw.copy()
    normalized[group_cols] = column_minmax(raw[group_cols].to_numpy(dtype=np.float32))
    return raw, normalized


def draw_descriptor_distance_heatmap(ax, heatmap_table):
    group_cols = ["coordinates", "bone vectors", "joint angles", "fingertip distances"]
    sns.heatmap(
        heatmap_table[group_cols],
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Normalized difference"},
    )
    # ax.set_title("C. Standardized descriptor distances\nfor similar gesture pairs", pad=6)
    ax.set_yticklabels(heatmap_table["pair"], rotation=0)
    ax.set_xticklabels(["coordinates", "bone\nvectors", "joint\nangles", "fingertip\ndistances"], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")


def save_individual_subfigure_pdfs(image, xy, heatmap_table, out_dir):
    saved = []

    fig_a, ax_a = plt.subplots(figsize=(3.2, 3.2))
    image_axis(ax_a, image)
    draw_skeleton(ax_a, xy, color="#0f766e", labels=True)
    out_a = Path(out_dir) / "figure1a_raw_landmarks.pdf"
    fig_a.savefig(out_a, bbox_inches="tight")
    plt.close(fig_a)
    saved.append(str(out_a))

    fig_b, ax_b = plt.subplots(figsize=(3.2, 3.2))
    draw_aar_components(ax_b, xy)
    out_b = Path(out_dir) / "figure1b_anatomy_aware_descriptors.pdf"
    fig_b.savefig(out_b, bbox_inches="tight")
    plt.close(fig_b)
    saved.append(str(out_b))

    fig_c, ax_c = plt.subplots(figsize=(4.1, 3.35))
    draw_descriptor_distance_heatmap(ax_c, heatmap_table)
    out_c = Path(out_dir) / "figure1c_descriptor_distances.pdf"
    fig_c.savefig(out_c, bbox_inches="tight")
    plt.close(fig_c)
    saved.append(str(out_c))

    return saved


def plot_figure(args):
    apply_paper_style()
    pairs = parse_pairs(args.pairs)
    df = read_landmark_csv(args.csv_path)
    sample = choose_sample(df, args.sample_label)
    image, image_path = load_image_for_row(sample, args.image_root)
    sample_coords = sample[[f"{axis}_{idx}" for idx in range(21) for axis in ("x", "y", "z")]].to_numpy(dtype=np.float32)
    sample_coords = sample_coords.reshape(21, 3)
    xy = normalized_to_pixel(sample_coords, image.shape)

    raw_table, heatmap_table = descriptor_distance_table(df, pairs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_table.to_csv(out_dir / "fig1_descriptor_distances_standardized_raw.csv", index=False)
    raw_table.to_csv(out_dir / "fig1_descriptor_distances_raw.csv", index=False)
    heatmap_table.to_csv(out_dir / "fig1_descriptor_distances_normalized.csv", index=False)

    fig = plt.figure(figsize=(12.2, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.18], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    image_axis(ax_a, image)
    draw_skeleton(ax_a, xy, color="#0f766e", labels=True)

    draw_aar_components(ax_b, xy)

    draw_descriptor_distance_heatmap(ax_c, heatmap_table)

    fig.suptitle("Anatomy-aware skeleton descriptor visualization", y=1.02, fontsize=12)
    saved = save_figure(fig, out_dir, "figure1_aar_descriptor")
    saved.extend(save_individual_subfigure_pdfs(image, xy, heatmap_table, out_dir))
    plt.close(fig)
    return saved, image_path


def dry_run(args):
    pairs = parse_pairs(args.pairs)
    df = read_landmark_csv(args.csv_path, nrows=args.dry_run_rows)
    labels = set(df["label"].astype(str))
    missing = sorted({label for pair in pairs for label in pair if label not in labels})
    sample = choose_sample(df, args.sample_label)
    image_path = Path(args.image_root) / str(sample["image_path"]).replace("\\", "/")
    print(f"CSV rows checked: {len(df)}")
    print(f"Sample label: {args.sample_label}; image candidate: {image_path}")
    print(f"Pairs: {pairs}")
    if missing:
        print(f"Warning: labels not found in first {len(df)} rows: {missing}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot Figure 1: AAR descriptor visualization.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--sample_label", default="A")
    parser.add_argument("--pairs", nargs="+", default=["A:S", "M:N", "U:V", "R:U", "D:F"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_rows", type=int, default=5000)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    saved, image_path = plot_figure(args)
    print(f"Used sample image: {image_path}")
    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
