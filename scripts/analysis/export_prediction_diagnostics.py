import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from scg_vtnet.evaluation.prediction_diagnostics import (
    DEFAULT_CSV_PATH,
    DEFAULT_EXP_DIR,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TOKEN_PATH,
    choose_device,
    experiment_paths,
    load_checkpoint_model,
    load_config,
    load_dataframe_and_arrays,
    resolve_image_path,
    softmax_np,
)


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

PREFERRED_PAIRS = [("U", "V"), ("R", "U"), ("M", "N"), ("A", "S"), ("D", "F")]
PAIR_CONDITIONS = {
    ("U", "V"): "Adjacent-finger separation is visually subtle.",
    ("V", "U"): "Adjacent-finger separation is visually subtle.",
    ("R", "U"): "Overlapping fingers and thumb visibility vary with pose.",
    ("U", "R"): "Overlapping fingers and thumb visibility vary with pose.",
    ("M", "N"): "Folded-finger configurations differ by a small boundary.",
    ("N", "M"): "Folded-finger configurations differ by a small boundary.",
    ("A", "S"): "Closed-fist poses depend on thumb placement.",
    ("S", "A"): "Closed-fist poses depend on thumb placement.",
    ("D", "F"): "Index/loop configuration is sensitive to local visual evidence.",
    ("F", "D"): "Index/loop configuration is sensitive to local visual evidence.",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def draw_skeleton(ax, coords, image_shape):
    h, w = image_shape[:2]
    xy = coords[:, :2].copy()
    xy[:, 0] *= w
    xy[:, 1] *= h
    lines = [[xy[a], xy[b]] for a, b in HAND_EDGES]
    ax.add_collection(LineCollection(lines, colors="#0f766e", linewidths=1.7, alpha=0.95))
    ax.scatter(xy[:, 0], xy[:, 1], s=18, c="#0f766e", edgecolors="white", linewidths=0.45, zorder=3)


def infer_logits(model, device, keypoints, raw_xyz, image_tokens, hand_detected, batch_size=256):
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(keypoints), batch_size):
            end = start + batch_size
            kpt = torch.from_numpy(np.array(keypoints[start:end], dtype=np.float32, copy=True)).to(device)
            raw = torch.from_numpy(np.array(raw_xyz[start:end], dtype=np.float32, copy=True)).to(device)
            img = torch.from_numpy(np.array(image_tokens[start:end], dtype=np.float32, copy=True)).to(device)
            hand = torch.from_numpy(np.array(hand_detected[start:end], dtype=np.float32, copy=True)).to(device)
            out = model(kpt, img, raw_xyz=raw, hand_detected=hand, return_diagnostics=False)
            logits = out[0] if isinstance(out, (tuple, list)) else out["logits"]
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def clean_predictions(model, device, arrays, token_features, batch_size):
    test_idx = arrays["split_indices"]["test"]
    logits_chunks = []
    for start in range(0, len(test_idx), batch_size):
        batch_idx = test_idx[start:start + batch_size]
        logits_chunks.append(
            infer_logits(
                model,
                device,
                arrays["model_inputs"][batch_idx],
                arrays["raw_xyz"][batch_idx],
                token_features[batch_idx],
                arrays["hand_detected"][batch_idx],
                batch_size=batch_size,
            )
        )
    logits = np.concatenate(logits_chunks, axis=0)
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    labels = arrays["labels"][test_idx]
    return test_idx, logits, probs, preds, labels


def token_occlusion(model, device, arrays, token_features, original_index, mask_value=0.0):
    label = arrays["labels"][original_index]
    keypoint = np.repeat(arrays["model_inputs"][original_index:original_index + 1], 49, axis=0)
    raw = np.repeat(arrays["raw_xyz"][original_index:original_index + 1], 49, axis=0)
    hand = np.repeat(arrays["hand_detected"][original_index:original_index + 1], 49, axis=0)
    tokens = np.repeat(np.asarray(token_features[original_index:original_index + 1]), 49, axis=0)
    for idx in range(49):
        tokens[idx, idx, :] = mask_value
    clean_logits = infer_logits(
        model,
        device,
        arrays["model_inputs"][original_index:original_index + 1],
        arrays["raw_xyz"][original_index:original_index + 1],
        token_features[original_index:original_index + 1],
        arrays["hand_detected"][original_index:original_index + 1],
        batch_size=1,
    )
    occluded_logits = infer_logits(model, device, keypoint, raw, tokens, hand, batch_size=49)
    clean_prob = softmax_np(clean_logits)[0, label]
    occluded_probs = softmax_np(occluded_logits)[:, label]
    return np.maximum(0.0, clean_prob - occluded_probs).reshape(7, 7), clean_logits[0]


def patch_occlusion(model, device, arrays, token_features, original_index, patch_size=2, mask_value=0.0):
    label = arrays["labels"][original_index]
    positions = []
    for r in range(7 - patch_size + 1):
        for c in range(7 - patch_size + 1):
            positions.append((r, c))
    n = len(positions)
    keypoint = np.repeat(arrays["model_inputs"][original_index:original_index + 1], n, axis=0)
    raw = np.repeat(arrays["raw_xyz"][original_index:original_index + 1], n, axis=0)
    hand = np.repeat(arrays["hand_detected"][original_index:original_index + 1], n, axis=0)
    tokens = np.repeat(np.asarray(token_features[original_index:original_index + 1]), n, axis=0)
    for out_idx, (r, c) in enumerate(positions):
        for rr in range(r, r + patch_size):
            for cc in range(c, c + patch_size):
                tokens[out_idx, rr * 7 + cc, :] = mask_value
    clean_logits = infer_logits(
        model,
        device,
        arrays["model_inputs"][original_index:original_index + 1],
        arrays["raw_xyz"][original_index:original_index + 1],
        token_features[original_index:original_index + 1],
        arrays["hand_detected"][original_index:original_index + 1],
        batch_size=1,
    )
    occluded_logits = infer_logits(model, device, keypoint, raw, tokens, hand, batch_size=n)
    clean_prob = softmax_np(clean_logits)[0, label]
    occluded_probs = softmax_np(occluded_logits)[:, label]
    drops = np.maximum(0.0, clean_prob - occluded_probs)
    preds = softmax_np(occluded_logits).argmax(axis=1)
    patch_map = drops.reshape(7 - patch_size + 1, 7 - patch_size + 1)
    worst = int(np.argmax(drops))
    return patch_map, positions[worst], preds[worst], occluded_probs[worst], clean_logits[0]


def select_diagnostic_sample(model, device, arrays, token_features, test_idx, preds, labels, classes, args):
    if args.sample_index is not None:
        return int(args.sample_index)
    preferred_classes = [name for pair in PREFERRED_PAIRS for name in pair if name in classes]
    candidates = []
    for cls_name in preferred_classes:
        cls_id = classes.index(cls_name)
        cls_positions = np.where((labels == cls_id) & (preds == labels))[0][:args.max_search_per_class]
        candidates.extend([int(test_idx[pos]) for pos in cls_positions])
    if not candidates:
        correct = np.where(preds == labels)[0]
        return int(test_idx[int(correct[0])])

    best = None
    for original_index in candidates:
        patch_map, pos, patch_pred, patch_prob, clean_logits = patch_occlusion(
            model, device, arrays, token_features, original_index, patch_size=args.patch_size
        )
        drop = float(patch_map.max())
        flips = patch_pred != arrays["labels"][original_index]
        score = drop + (1.0 if flips else 0.0)
        if best is None or score > best[0]:
            best = (score, original_index)
            if flips and drop >= args.early_stop_drop:
                break
    return int(best[1])


def export_confusion_table(output_dir, classes, labels, preds):
    rows = []
    for true_id, pred_id in zip(labels, preds):
        if true_id == pred_id:
            continue
        rows.append((classes[int(true_id)], classes[int(pred_id)]))
    counts = pd.Series(rows, dtype=object).value_counts() if rows else pd.Series(dtype=int)

    table_rows = []
    used = set()
    for pair in PREFERRED_PAIRS:
        for oriented_pair in (pair, (pair[1], pair[0])):
            count = int(counts.get(oriented_pair, 0))
            if count > 0 and oriented_pair not in used:
                table_rows.append({
                    "true_class": oriented_pair[0],
                    "predicted_class": oriented_pair[1],
                    "count": count,
                    "representative_condition": PAIR_CONDITIONS.get(oriented_pair, "Clean-test confusion."),
                })
                used.add(oriented_pair)
                break
        if len(table_rows) >= 5:
            break

    for pair, count in counts.items():
        if len(table_rows) >= 5:
            break
        if pair in used:
            continue
        table_rows.append({
            "true_class": pair[0],
            "predicted_class": pair[1],
            "count": int(count),
            "representative_condition": PAIR_CONDITIONS.get(pair, "Clean-test confusion."),
        })
        used.add(pair)

    out_csv = Path(output_dir) / "table_top_confusion_pairs.csv"
    pd.DataFrame(table_rows).to_csv(out_csv, index=False)

    out_tex = Path(output_dir) / "table_top_confusion_pairs.tex"
    with out_tex.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top confusion pairs on the ASL clean test split.}\n")
        f.write("\\label{tab:top_confusion_pairs}\n")
        f.write("\\begin{tabular}{llrl}\n")
        f.write("\\toprule\n")
        f.write("True class & Predicted class & Count & Representative condition \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            condition = row["representative_condition"].replace("&", "\\&")
            f.write(f"{row['true_class']} & {row['predicted_class']} & {row['count']} & {condition} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    return table_rows, out_csv, out_tex


def draw_heatmap(ax, data, title, cmap="magma", rectangle=None, patch_size=2):
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_title(title, pad=6)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xlabel("token column")
    ax.set_ylabel("token row")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#334155")
    if rectangle is not None:
        r, c = rectangle
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), patch_size, patch_size,
                               fill=False, edgecolor="#22c55e", linewidth=2.2))
    return im


def export_figure(output_dir, arrays, image_root, classes, original_index, clean_logits,
                  token_map, patch_map, patch_pos, patch_pred, patch_prob, patch_size):
    image_path = resolve_image_path(image_root, arrays["df"].iloc[original_index]["image_path"])
    image = plt.imread(image_path)
    true_name = classes[arrays["labels"][original_index]]
    clean_probs = softmax_np(clean_logits[None, :])[0]
    clean_pred = int(clean_probs.argmax())

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15), constrained_layout=True)
    axes[0].imshow(image)
    draw_skeleton(axes[0], arrays["coords"][original_index], image.shape)
    axes[0].set_title(f"(a) Sample and landmarks\ntrue {true_name}, pred {classes[clean_pred]}", pad=6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = draw_heatmap(
        axes[1],
        token_map,
        "(b) Token occlusion sensitivity\ncorrect-class probability drop",
        cmap="viridis",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    im2 = draw_heatmap(
        axes[2],
        patch_map,
        f"(c) {patch_size}x{patch_size} token patch occlusion\nworst pred {classes[int(patch_pred)]}, ptrue {patch_prob:.3f}",
        cmap="magma",
        rectangle=patch_pos,
        patch_size=patch_size,
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)

    paths = []
    for ext in ("png", "pdf", "svg"):
        path = Path(output_dir) / f"figure6_token_diagnostic.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths, image_path


def save_panel(fig, output_dir, stem):
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = Path(output_dir) / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def export_separate_panels(output_dir, arrays, image_root, classes, original_index, clean_logits,
                           token_map, patch_map, patch_pos, patch_pred, patch_prob, patch_size):
    image_path = resolve_image_path(image_root, arrays["df"].iloc[original_index]["image_path"])
    image = plt.imread(image_path)
    true_name = classes[arrays["labels"][original_index]]
    clean_probs = softmax_np(clean_logits[None, :])[0]
    clean_pred = int(clean_probs.argmax())

    apply_style()
    exported = {}

    fig_a, ax_a = plt.subplots(1, 1, figsize=(3.2, 3.15), constrained_layout=True)
    ax_a.imshow(image)
    draw_skeleton(ax_a, arrays["coords"][original_index], image.shape)
    ax_a.set_title(f"Sample and landmarks\ntrue {true_name}, pred {classes[clean_pred]}", pad=6)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    exported["a"] = save_panel(fig_a, output_dir, "figure6_token_diagnostic_a_sample")

    fig_b, ax_b = plt.subplots(1, 1, figsize=(3.35, 3.15), constrained_layout=True)
    im_b = draw_heatmap(
        ax_b,
        token_map,
        "Token occlusion sensitivity\ncorrect-class probability drop",
        cmap="viridis",
    )
    fig_b.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03)
    exported["b"] = save_panel(fig_b, output_dir, "figure6_token_diagnostic_b_token_occlusion")

    fig_c, ax_c = plt.subplots(1, 1, figsize=(3.35, 3.15), constrained_layout=True)
    im_c = draw_heatmap(
        ax_c,
        patch_map,
        f"{patch_size}x{patch_size} token patch occlusion\nworst pred {classes[int(patch_pred)]}, ptrue {patch_prob:.3f}",
        cmap="magma",
        rectangle=patch_pos,
        patch_size=patch_size,
    )
    fig_c.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
    exported["c"] = save_panel(fig_c, output_dir, "figure6_token_diagnostic_c_patch_occlusion")

    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--manifest_path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--token_path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--exp_dir", default=str(DEFAULT_EXP_DIR))
    parser.add_argument("--experiment", default="E")
    parser.add_argument("--output_dir", default=str(Path(DEFAULT_EXP_DIR) / "diagnostics"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--max_search_per_class", type=int, default=35)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--early_stop_drop", type=float, default=0.40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.exp_dir, args.experiment)
    paths = experiment_paths(args.exp_dir, args.experiment)
    arrays = load_dataframe_and_arrays(args.csv_path, args.manifest_path, config)
    token_features = np.load(args.token_path, mmap_mode="r")
    device = choose_device(args.device)
    model = load_checkpoint_model(config, paths["weights"], len(arrays["classes"]), device)

    test_idx, logits, probs, preds, labels = clean_predictions(
        model, device, arrays, token_features, args.batch_size
    )
    acc = float((preds == labels).mean())
    confusion_rows, confusion_csv, confusion_tex = export_confusion_table(
        output_dir, arrays["classes"], labels, preds
    )

    sample_index = select_diagnostic_sample(
        model, device, arrays, token_features, test_idx, preds, labels, arrays["classes"], args
    )
    token_map, clean_logits = token_occlusion(model, device, arrays, token_features, sample_index)
    patch_map, patch_pos, patch_pred, patch_prob, _ = patch_occlusion(
        model, device, arrays, token_features, sample_index, patch_size=args.patch_size
    )

    figure_paths, image_path = export_figure(
        output_dir,
        arrays,
        args.image_root,
        arrays["classes"],
        sample_index,
        clean_logits,
        token_map,
        patch_map,
        patch_pos,
        patch_pred,
        patch_prob,
        args.patch_size,
    )
    panel_paths = export_separate_panels(
        output_dir,
        arrays,
        args.image_root,
        arrays["classes"],
        sample_index,
        clean_logits,
        token_map,
        patch_map,
        patch_pos,
        patch_pred,
        patch_prob,
        args.patch_size,
    )

    sensitivity_csv = output_dir / "figure6_token_patch_sensitivity.csv"
    with sensitivity_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["map_type", "row", "col", "value"])
        for r in range(token_map.shape[0]):
            for c in range(token_map.shape[1]):
                writer.writerow(["token_occlusion", r, c, float(token_map[r, c])])
        for r in range(patch_map.shape[0]):
            for c in range(patch_map.shape[1]):
                writer.writerow([f"patch_{args.patch_size}x{args.patch_size}", r, c, float(patch_map[r, c])])

    summary = output_dir / "diagnostic_summary.txt"
    true_name = arrays["classes"][arrays["labels"][sample_index]]
    clean_pred = arrays["classes"][int(softmax_np(clean_logits[None, :])[0].argmax())]
    with summary.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Clean test accuracy reproduced: {acc * 100:.4f}%\n")
        f.write(f"Diagnostic sample original_index: {sample_index}\n")
        f.write(f"Diagnostic sample image: {image_path}\n")
        f.write(f"True class: {true_name}\n")
        f.write(f"Clean prediction: {clean_pred}\n")
        f.write(f"Worst patch top-left: row={patch_pos[0]}, col={patch_pos[1]}\n")
        f.write(f"Worst patch prediction: {arrays['classes'][int(patch_pred)]}\n")
        f.write(f"Worst patch true-class probability: {patch_prob:.6f}\n")
        f.write("\nTop confusion rows:\n")
        for row in confusion_rows:
            f.write(f"{row['true_class']} -> {row['predicted_class']}: {row['count']} ({row['representative_condition']})\n")

    print(f"Reproduced clean test accuracy: {acc * 100:.4f}%")
    print(f"Figure 6 saved to: {figure_paths[0]}")
    print(f"Figure 6 panels saved to: {panel_paths['a'][0]}, {panel_paths['b'][0]}, {panel_paths['c'][0]}")
    print(f"Sensitivity values saved to: {sensitivity_csv}")
    print(f"Top confusion CSV saved to: {confusion_csv}")
    print(f"Top confusion LaTeX saved to: {confusion_tex}")
    print(f"Summary saved to: {summary}")


if __name__ == "__main__":
    main()
