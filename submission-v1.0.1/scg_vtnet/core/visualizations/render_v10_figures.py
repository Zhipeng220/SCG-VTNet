"""Rebuild publication multi-panel figures (token masking, token trade-off, AAR descriptors, paired degradation, original cases)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image

MM = 1 / 25.4
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
INK = "#161616"
FAMILIES = ("#606060", "#767676", TEAL, "#56B4E9", BLUE)
MARKERS = ("o", "s", "D", "^", "P")
KEYS = ("E7", "E2", "E5")
FIGURES = (
    "fig2_token_masking",
    "figs1_token_tradeoff",
    "figs2_aar_descriptors",
    "figs3_paired_degradation",
    "figs4_original_cases",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def open_rgb(path):
    with Image.open(path) as f:
        return np.asarray(f.convert("RGB"))


def panel(ax, letter, subtitle=None):
    ax.text(0, 1.075, letter, transform=ax.transAxes, size=8, weight="bold", ha="left", va="bottom")
    if subtitle:
        ax.text(0.115, 1.075, subtitle, transform=ax.transAxes, size=7, ha="left", va="bottom")


def skeleton(ax, xy, edges, color, alpha=1, width=0.7, size=4):
    for a, b in edges:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=color, linewidth=width, alpha=alpha)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, alpha=alpha, linewidths=0, zorder=3)


def image(ax, data):
    ax.imshow(data, interpolation="nearest")
    ax.axis("off")


def mask_matrix(records, key, n):
    cells = [r for r in records if r["map_type"] == key]
    values = np.full((n, n), np.nan)
    assert len(cells) == n * n
    for r in cells:
        slot = int(r["row"]), int(r["col"])
        assert np.isnan(values[slot])
        values[slot] = float(r["value"])
    assert np.isfinite(values).all() and (values >= 0).all()
    return values


def heatmap(ax, matrix, maximum, steps):
    obj = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=maximum, interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]), range(1, matrix.shape[1] + 1))
    ax.set_yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
    ax.tick_params(length=2, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(obj, ax=ax, fraction=0.045, pad=0.07, ticks=steps)
    bar.ax.tick_params(length=2, pad=2, labelsize=5.6)
    bar.outline.set_linewidth(0.45)
    return obj


def masking(lang, vis, landmarks, processed, audit):
    tr = lambda e, z: z if lang == "zh" else e
    records = load_rows(audit / "figure6_token_patch_sensitivity.csv")
    token = mask_matrix(records, "token_occlusion", 7)
    patch = mask_matrix(records, "patch_2x2", 6)
    index = 65327
    row = landmarks.iloc[index]
    assert row["label"] == "V" and int(row["hand_detected"]) == 1
    assert str(row["image_path"]).replace(chr(92), "/") == "V/V393.jpg"
    v = open_rgb(processed / "zerofill_images_processed/V/V393.jpg")
    xy = vis.normalized_to_pixel(vis.coords_from_frame(landmarks.iloc[[index]])[0], v.shape)
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 61 * MM))
    fig.subplots_adjust(left=0.025, right=0.971, bottom=0.205, top=0.81, wspace=0.37)
    a, b, c = axs
    image(a, v)
    skeleton(a, xy, vis.HAND_EDGES, "#5BD6C0", width=0.72, size=5)
    panel(a, "a", tr("Input and landmarks", "输入图像与关键点"))
    panel(b, "b", tr("Single-token masking", "单词元屏蔽"))
    panel(c, "c", tr("2 × 2 patch masking", "2 × 2 词元块屏蔽"))
    heatmap(b, token, 0.004, [0, 0.002, 0.004])
    heatmap(c, patch, 0.60, [0, 0.30, 0.60])
    r, col = np.unravel_index(patch.argmax(), patch.shape)
    assert (r, col) == (2, 3)
    c.add_patch(Rectangle((col - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.0))
    b.set_xlabel(tr("Token column", "词元列"))
    b.set_ylabel(tr("Token row", "词元行"))
    c.set_xlabel(tr("Patch top-left column", "块左上角列"))
    c.set_ylabel(tr("Patch top-left row", "块左上角行"))
    fig.text(0.178, 0.08, tr("Clean prediction V", "无扰动预测 V"), ha="center", fontsize=6.5)
    fig.text(
        0.786, 0.026,
        tr("Peak drop: predicted U, P(V) = 0.346", "最大下降处：预测 U，P(V) = 0.346"),
        ha="center", fontsize=6.1,
    )
    return fig


def add_grid(ax, x, y, size, holes=(), window=None):
    step = size / 7
    for r in range(7):
        for c in range(7):
            shade = ("#DCEAF4" if (r + c) % 3 else "#B5D3EA") if (r, c) not in holes else "#FAF2EA"
            ax.add_patch(Rectangle(
                (x + c * step, y + (6 - r) * step), step, step,
                facecolor=shade, edgecolor="#71879D", linewidth=0.37,
            ))
            if (r, c) in holes:
                ax.plot(
                    [x + c * step, x + (c + 1) * step],
                    [y + (6 - r) * step, y + (7 - r) * step],
                    color=ORANGE, linewidth=0.6,
                )
    if window:
        r, c, n = window
        ax.add_patch(Rectangle(
            (x + c * step, y + (7 - r - n) * step), n * step, n * step,
            facecolor="none", edgecolor=ORANGE, linewidth=1,
        ))


def s13_data_from_tex(tex):
    s = tex.read_text(encoding="utf-8")
    region = s[s.index(r"\label{tab:ablation_visual_modulation}"):]
    region = region[:region.index(r"\end{table}")]
    matches = re.findall(r"(?m)^\s*([A-E])\s*&[^\n]+?\n\s*&\s*([^\n]+)\\\\", region)
    assert [a for a, _ in matches] == list("ABCDE")
    values = []
    for letter, raw in matches:
        cells = [t.strip() for t in raw.split("&")]
        assert len(cells) == 9, (letter, cells)
        number = lambda c: None if r"\NA" in c else float(c)
        values.append({
            "configuration": letter,
            "clean_pct": number(cells[0]),
            "skel_gaussian_pct": number(cells[2]),
            "patch_occlusion_pct": number(cells[7]),
            "joint_pct": number(cells[8]),
        })
    return values


def s13_data_from_csv(csv_path):
    values = []
    for row in load_rows(csv_path):
        parse = lambda val: None if val in ("", "None", "nan") else float(val)
        values.append({
            "configuration": row["configuration"],
            "clean_pct": parse(row["clean_pct"]),
            "skel_gaussian_pct": parse(row["skel_gaussian_pct"]),
            "patch_occlusion_pct": parse(row["patch_occlusion_pct"]),
            "joint_pct": parse(row["joint_pct"]),
        })
    assert [r["configuration"] for r in values] == list("ABCDE")
    return values


def tradeoff(lang, records):
    tr = lambda e, z: z if lang == "zh" else e
    fig = plt.figure(figsize=(183 * MM, 104 * MM))
    ax = fig.add_axes([0.04, 0.617, 0.44, 0.285])
    bx = fig.add_axes([0.52, 0.617, 0.44, 0.285])
    panel(ax, "a", tr("Visual representation", "视觉表示"))
    panel(bx, "b", tr("Feature-space perturbations", "特征空间扰动"))
    for p in (ax, bx):
        p.set_xlim(0, 1)
        p.set_ylim(0, 0.65)
        p.set_aspect("equal")
        p.axis("off")
    add_grid(ax, 0.025, 0.09, 0.43)
    ax.annotate("", (0.62, 0.43), (0.45, 0.37), arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.65})
    ax.annotate("", (0.62, 0.20), (0.45, 0.27), arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.65})
    for i in range(7):
        ax.add_patch(Rectangle((0.646 + i * 0.042, 0.412), 0.035, 0.043, facecolor="#A8A8A8", edgecolor="none"))
    for i in range(4):
        ax.add_patch(Rectangle((0.646 + i * 0.075, 0.172), 0.058, 0.045, facecolor=BLUE, edgecolor="none"))
    ax.text(0.644, 0.51, tr("Mean pooled vector", "平均池化向量"), size=6.3)
    ax.text(0.644, 0.093, tr("49 spatial tokens", "49 个空间词元"), size=6.3)
    ax.text(0.025, 0.005, "7 × 7 × 2,048", size=6.1)
    add_grid(bx, 0.008, 0.11, 0.38, holes={(0, 1), (2, 4), (4, 5), (5, 2)})
    cells = {(r, c) for r in range(1, 5) for c in range(2, 6)}
    add_grid(bx, 0.535, 0.11, 0.38, holes=cells, window=(1, 2, 4))
    bx.text(0.197, 0.025, tr("Token dropout", "词元丢弃"), ha="center", fontsize=6.3)
    bx.text(0.725, 0.025, tr("Contiguous patch", "连续块遮挡"), ha="center", fontsize=6.3)
    c = fig.add_axes([0.075, 0.11, 0.867, 0.366])
    panel(c, "c", tr("Accuracy loss from clean input", "相对无扰动准确率的降幅"))
    columns = ("skel_gaussian_pct", "patch_occlusion_pct", "joint_pct")
    labels = (
        tr("Skeleton Gaussian", "骨架高斯噪声"),
        tr("Spatial patch", "空间块遮挡"),
        tr("Joint perturbation", "联合扰动"),
    )
    for i, row in enumerate(records):
        x = np.arange(3) + (i - 2) * 0.132
        vals = np.array([np.nan if row[k] is None else row["clean_pct"] - row[k] for k in columns])
        for x0, v in zip(x, vals):
            if np.isfinite(v):
                c.vlines(x0, 0, v, color=FAMILIES[i], linewidth=1.1)
        c.plot(
            x, vals, ls="none", marker=MARKERS[i], markersize=4.3,
            color=FAMILIES[i], label=row["configuration"],
        )
    c.axhline(0, color="#8D969F", linewidth=0.55, linestyle="--")
    c.grid(axis="y", linewidth=0.4, color="#E1E6EA")
    c.set_axisbelow(True)
    c.set_xticks(range(3), labels)
    c.set_xlim(-0.49, 2.49)
    c.set_ylim(-0.25, 4.35)
    c.set_ylabel(tr("Accuracy drop (pp)", "准确率下降（百分点）"))
    c.yaxis.set_major_locator(MultipleLocator(1))
    c.legend(loc="upper right", ncol=5, handletextpad=0.3, columnspacing=0.8)
    return fig


def aar(lang, vis, landmarks, processed, normalized):
    tr = lambda e, z: z if lang == "zh" else e
    v = open_rgb(processed / "zerofill_images_processed/A/A1.jpg")
    r = landmarks.loc[landmarks["image_path"].str.replace(chr(92), "/", regex=False) == "A/A1.jpg"]
    assert len(r) == 1 and int(r.iloc[0]["hand_detected"]) == 1
    xy = vis.normalized_to_pixel(vis.coords_from_frame(r)[0], v.shape)
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 63 * MM), gridspec_kw={"width_ratios": [1, 1.14, 1.7]})
    fig.subplots_adjust(left=0.024, right=0.97, bottom=0.222, top=0.825, wspace=0.43)
    a, b, c = axs
    image(a, v)
    skeleton(a, xy, vis.HAND_EDGES, "#5BD6C0")
    panel(a, "a", tr("Landmark input", "关键点输入"))
    panel(b, "b", tr("AAR measurements", "AAR 测量"))
    skeleton(b, xy, vis.HAND_EDGES, "#A6ADB5", alpha=0.92, width=0.60, size=4)
    b.set_aspect("equal")
    b.axis("off")
    b.invert_yaxis()
    b.margins(0.13)
    b.annotate("", xy[8], xy[7], arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    ax0, ay0 = xy[6]
    angles = np.degrees(np.arctan2(xy[[5, 7], 1] - ay0, xy[[5, 7], 0] - ax0))
    start, end = sorted(angles)
    if end - start > 180:
        start, end = end, start + 360
    radius = 0.46 * min(np.linalg.norm(xy[5] - xy[6]), np.linalg.norm(xy[7] - xy[6]))
    b.add_patch(Arc(
        (ax0, ay0), 2 * radius, 2 * radius, theta1=start, theta2=end,
        color=ORANGE, linewidth=1.25, zorder=5,
    ))
    p0, p1 = xy[4], xy[12]
    b.plot([p0[0], p1[0]], [p0[1], p1[1]], color=TEAL, ls="--", lw=1.05)
    b.text(
        0.49, -0.035, tr("Vector  ·  Angle  ·  Tip distance", "骨向量 · 关节角 · 指尖距离"),
        transform=b.transAxes, ha="center", va="top", fontsize=6,
    )
    for k, color in enumerate((BLUE, ORANGE, TEAL)):
        b.plot(
            [0.163 + k * 0.277, 0.236 + k * 0.277], [-0.135, -0.135],
            transform=b.transAxes, clip_on=False, color=color, linewidth=1.3,
            ls="--" if k == 2 else "-",
        )
    headers = list(normalized.columns[1:])
    matrix = normalized[headers].to_numpy()
    heat = c.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, j in np.ndindex(matrix.shape):
        c.text(
            j, i, f"{matrix[i, j]:.2f}", color=INK if matrix[i, j] > 0.67 else "white",
            ha="center", va="center", fontsize=6.1,
        )
    c.set_yticks(range(len(matrix)), [pair.replace(" vs ", "–") for pair in normalized["pair"]])
    c.set_xticks(
        range(4),
        tr(["XYZ", "Bone\nvectors", "Joint\nangles", "Tip\ndistances"], ["坐标", "骨向量", "关节角", "指尖距离"]),
    )
    c.tick_params(length=2, pad=3)
    for spine in c.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(heat, ax=c, fraction=0.052, pad=0.05, ticks=[0, 0.5, 1])
    bar.ax.tick_params(labelsize=5.4, length=2, pad=2)
    bar.outline.set_linewidth(0.45)
    panel(c, "c", tr("Within-descriptor scaling", "按描述符分组缩放"))
    return fig


def outcomes(ax, y, vals, truth):
    for x, name in zip((0.147, 0.500, 0.853), KEYS):
        v = vals[name]
        ax.text(
            x, y, name + " " + v + (" ×" if v != truth else ""),
            transform=ax.transAxes, fontsize=6.8, ha="center", va="center",
            color=INK, weight="semibold" if v != truth else "normal",
        )


def plate_s3(lang, folder, predictions):
    tr = lambda e, z: z if lang == "zh" else e
    cases = [
        ("a", "A/A2658.jpg", "blur_heavy", tr("Heavy blur", "重度模糊")),
        ("b", "A/A1160.jpg", "blur_heavy", tr("Heavy blur", "重度模糊")),
        ("c", "D/D2936.jpg", "occlude_heavy", tr("Heavy occlusion", "重度遮挡")),
        ("d", "X/X1375.jpg", "clean", tr("Clean confusion", "原图误判")),
    ]
    fig = plt.figure(figsize=(183 * MM, 132 * MM))
    for i, (letter, path, condition, title) in enumerate(cases):
        x, y = 0.036 + (i % 2) * 0.509, 0.55 - (i // 2) * 0.485
        header = fig.add_axes((x, y, 0.415, 0.358))
        header.axis("off")
        panel(header, letter, title)
        vals = {m: predictions[path, condition, m]["prediction"] for m in KEYS}
        rec = predictions[path, condition, "E7"]
        event = lambda flag: tr("Yes", "检出") if flag == "1" else tr("No", "漏检")
        legend = tr("True ", "真值 ") + rec["true_label"] + "    " + tr("MP ", "MediaPipe ") + event(rec["clean_detected"])
        if condition != "clean":
            legend += " → " + event(rec["condition_detected"])
        figs = ("_clean", "_degraded") if condition != "clean" else ("_clean",)
        for j, suffix in enumerate(figs):
            w = 0.185
            left = x + 0.114 if len(figs) == 1 else x + 0.010 + j * 0.214
            photo = fig.add_axes((left, y + 0.048, w, w * 183 / 132))
            image(photo, open_rgb(folder / f"{letter}{suffix}.png"))
            if condition != "clean":
                photo.text(
                    0.5, 1.027, tr("Original", "原图") if j == 0 else tr("Degraded", "合成退化"),
                    ha="center", va="bottom", transform=photo.transAxes, fontsize=6,
                )
        header.text(0.50, 0.043, legend, transform=header.transAxes, ha="center", fontsize=6.6)
        outcomes(header, -0.044, vals, rec["true_label"])
    return fig


def plate_s4(lang, folder, rows):
    tr = lambda e, z: z if lang == "zh" else e
    cats = (
        ("challenging_light", tr("Difficult lighting", "困难光照")),
        ("oblique_view", tr("Oblique view", "侧斜视图")),
        ("partial_out_of_frame", tr("Partial framing", "手部部分出画")),
        ("finger_self_occlusion", tr("Finger overlap", "手指自遮挡")),
    )
    table = {(r["source_pool"], r["category"]): r for r in rows}
    assert len(table) == 8
    fig = plt.figure(figsize=(183 * MM, 116 * MM))
    for i, (pool, title, top) in enumerate((
        ("letter-stratified screen", tr("Letter-stratified screen", "按字母分层筛查"), 0.817),
        ("error-enriched screen", tr("Error-enriched case screen", "误判富集筛查"), 0.383),
    )):
        fig.text(0.028, top + 0.062, title, weight="semibold", fontsize=6.8)
        for j, (key, category) in enumerate(cats):
            r = table[pool, key]
            center = 0.135 + 0.243 * j
            w = 0.161
            h = w * 183 / 116
            photo = fig.add_axes([center - w / 2, top - h, w, h])
            (path,) = list((folder / "originals").glob(r["case_id"] + ".*"))
            image(photo, open_rgb(path))
            overlay = fig.add_axes([center - 0.10, top - h, 0.20, h])
            overlay.axis("off")
            overlay.text(
                -0.004, 1.033, "abcdefgh"[4 * i + j], transform=overlay.transAxes,
                ha="left", va="bottom", size=8, weight="bold",
            )
            if i == 0:
                fig.text(center, 0.965, category, fontsize=7, weight="semibold", ha="center", va="top")
            detector = tr("Yes", "检出") if r["hand_detected"] == "1" else tr("No", "漏检")
            overlay.text(
                0.50, -0.072, "ID " + r["original_index"] + "   " + tr("True ", "真值 ") + r["true_label"],
                transform=overlay.transAxes, va="top", ha="center", fontsize=6.25,
            )
            overlay.text(
                0.50, -0.157, "MP " + detector,
                transform=overlay.transAxes, va="top", ha="center", fontsize=6.0,
            )
            outcomes(overlay, -0.288, {m: r[m] for m in KEYS}, r["true_label"])
    return fig


def main(args):
    project = args.project.resolve()
    processed = args.processed.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    results = project / "results"
    viz_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(viz_dir))
    import common_viz as vis
    from plot_fig1_aar_descriptor import descriptor_distance_table

    landmark_file = processed / "zerofill_landmarks_mask.csv"
    freeze_path = project / "manifests/asl/freeze.json"
    if freeze_path.is_file():
        expected = json.loads(freeze_path.read_text(encoding="utf-8"))["source_files"][landmark_file.name]["sha256"]
        assert sha256(landmark_file) == expected
    landmarks = pd.read_csv(landmark_file)
    assert len(landmarks) == 87000

    pairs = vis.parse_pairs(["A:S", "M:N", "U:V", "R:U", "D:F"])
    raw, norm = descriptor_distance_table(landmarks, pairs)
    source_counts = landmarks.loc[landmarks.hand_detected.eq(1), "label"].value_counts()
    norm.to_csv(out / "S2_column_normalized_distances.csv", index=False)
    raw.to_csv(out / "S2_standardized_centroid_distances.csv", index=False)
    save_rows(
        out / "S2_class_sample_sizes.csv",
        [{"class": c, "detected_n": int(source_counts[c])} for c in sorted(set(sum((list(p) for p in pairs), [])))],
    )

    if args.s1_csv is not None:
        records = s13_data_from_csv(args.s1_csv.resolve())
    elif args.supplementary_tex is not None:
        records = s13_data_from_tex(args.supplementary_tex.resolve())
    else:
        default_tex = project / "论文/V10/en/supplementary_information.tex"
        if not default_tex.is_file():
            raise FileNotFoundError(
                "Supply --s1-csv (S1_source_accuracy.csv) or --supplementary-tex (supplementary_information.tex)."
            )
        records = s13_data_from_tex(default_tex)
    save_rows(out / "S1_source_accuracy.csv", records)

    diagnostic = results / "historical/visual_modulation/seed_42/diagnostics"
    s3_paths = {"A/A2658.jpg", "A/A1160.jpg", "D/D2936.jpg", "X/X1375.jpg"}
    predictions = {
        (r["image_path"], r["condition"], r["method"]): r
        for r in load_rows(results / "asl_paired_degradation/predictions.csv")
        if r["image_path"] in s3_paths
    }
    for m in KEYS:
        (r,) = [
            x for x in load_rows(results / f"asl_baselines/seed42_predictions/{m}_seed_42.csv")
            if x["image_path"].replace(chr(92), "/") == "X/X1375.jpg"
        ]
        assert int(r["original_index"]) == 69418 and r["true_label"] == "X"
        predictions["X/X1375.jpg", "clean", m] = {
            "prediction": r["predicted_label"],
            "true_label": r["true_label"],
            "clean_detected": r["hand_detected"],
        }

    cases = args.cases_dir.resolve() if args.cases_dir is not None else (results / "reviewer11_14/r11_original_cases")
    case_rows = load_rows(cases / "figure_cases.csv")

    for lang in args.languages:
        panels_dir = (
            args.case_panels_dir.resolve()
            if args.case_panels_dir is not None
            else (project / "论文/V10" / lang / "fig4_case_panels")
        )
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial" if lang == "en" else "Microsoft YaHei"],
            "font.size": 6.5, "axes.labelsize": 6.6, "xtick.labelsize": 6.1, "ytick.labelsize": 6.1,
            "legend.fontsize": 6.1, "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.width": 0.55, "ytick.major.width": 0.55, "xtick.major.size": 2.5,
            "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.65, "legend.frameon": False, "pdf.fonttype": 42,
            "svg.fonttype": "none", "axes.unicode_minus": False,
        })
        figures = (
            masking(lang, vis, landmarks, processed, diagnostic),
            tradeoff(lang, records),
            aar(lang, vis, landmarks, processed, norm),
            plate_s3(lang, panels_dir, predictions),
            plate_s4(lang, cases, case_rows),
        )
        for stem, figure in zip(FIGURES, figures):
            for ext, opts in (("pdf", {}), ("png", {"dpi": 230})):
                figure.savefig(out / f"{stem}_{lang}.{ext}", **opts)
            plt.close(figure)

    print("Figures exported for languages:", ", ".join(args.languages))
    print("S2 detected total:", int(source_counts.sum()))


def parse_args():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=root,
                        help="Project/archive root containing results/ and manifests/.")
    parser.add_argument("--processed", type=Path, required=True,
                        help="Processed ASL Alphabet directory containing zerofill_landmarks_mask.csv and images.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for PDF/PNG figures and CSV source data.")
    parser.add_argument("--s1-csv", type=Path, default=None,
                        help="Optional S1_source_accuracy.csv (avoids parsing supplementary_information.tex).")
    parser.add_argument("--supplementary-tex", type=Path, default=None,
                        help="Optional path to supplementary_information.tex for visual-modulation ablation values.")
    parser.add_argument("--case-panels-dir", type=Path, default=None,
                        help="Directory containing a_clean.png..d_clean.png exported by export_asl_case_panels.py.")
    parser.add_argument("--cases-dir", type=Path, default=None,
                        help="Directory containing figure_cases.csv and originals/ (defaults to results/reviewer11_14/r11_original_cases).")
    parser.add_argument("--languages", nargs="+", choices=("en", "zh"), default=["en", "zh"],
                        help="Languages to render (default: en zh).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
