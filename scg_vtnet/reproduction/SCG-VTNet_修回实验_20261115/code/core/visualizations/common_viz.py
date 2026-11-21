import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CSV_PATH = r"E:\AC-GCAT\AC-GCAT\data\ASL_Alphabet_V2\processed_train\zerofill_landmarks_mask.csv"
DEFAULT_TOKEN_PATH = r"E:\AC-GCAT\AC-GCAT\data\ASL_Alphabet_V2\processed_train\zerofill_visual_tokens_layer4_compact.npy"
DEFAULT_IMAGE_ROOT = r"E:\AC-GCAT\AC-GCAT\data\ASL_Alphabet_V2\processed_train\zerofill_images_processed"
DEFAULT_CORE_EXP_DIR = r"E:\AC-GCAT\AC-GCAT\experiment_results\core_ablation\seed_42"
DEFAULT_VISUAL_MODULATION_DIR = r"E:\AC-GCAT\AC-GCAT\experiment_results\visual_modulation\seed_42"

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
PARENT_MAP = {
    0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}
ANGLE_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
]
FINGER_GROUPS = [
    ("wrist", [0]),
    ("thumb", [1, 2, 3, 4]),
    ("index", [5, 6, 7, 8]),
    ("middle", [9, 10, 11, 12]),
    ("ring", [13, 14, 15, 16]),
    ("pinky", [17, 18, 19, 20]),
]

def apply_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_dir, stem):
    ensure_dir(output_dir)
    paths = []
    for ext in ("png", "pdf", "svg"):
        out = Path(output_dir) / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        paths.append(str(out))
    return paths


def parse_pairs(pairs):
    parsed = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Expected pair format A:S, got {pair}")
        a, b = pair.split(":", 1)
        parsed.append((a.strip(), b.strip()))
    return parsed


def read_landmark_csv(csv_path, nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows)
    required = {"label", "image_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def landmark_columns(df):
    cols = []
    for idx in range(21):
        for axis in ("x", "y", "z"):
            col = f"{axis}_{idx}"
            if col not in df.columns:
                raise ValueError(f"Missing landmark column: {col}")
            cols.append(col)
    return cols


def coords_from_frame(df):
    values = df[landmark_columns(df)].to_numpy(dtype=np.float32)
    return values.reshape(len(df), 21, 3)


def resolve_image_path(image_root, image_path):
    path = Path(str(image_path))
    if path.is_absolute() and path.exists():
        return path
    candidate = Path(image_root) / Path(str(image_path).replace("\\", os.sep))
    if candidate.exists():
        return candidate
    return path


def load_image_for_row(row, image_root):
    path = resolve_image_path(image_root, row["image_path"])
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return plt.imread(path), path


def normalized_to_pixel(coords, image_shape):
    h, w = image_shape[:2]
    xy = coords[:, :2].astype(np.float32).copy()
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy


def draw_skeleton(ax, xy, color="#13a8a8", linewidth=1.8, point_size=18,
                  labels=False, alpha=0.95):
    lines = [[xy[a], xy[b]] for a, b in HAND_EDGES]
    collection = LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection(collection)
    ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=color, edgecolors="white",
               linewidths=0.45, zorder=3, alpha=alpha)
    if labels:
        for idx, (x, y) in enumerate(xy):
            ax.text(x + 3, y - 3, str(idx), color="white", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.45, linewidth=0))


def image_axis(ax, image, title=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, pad=6)


def compute_descriptor_groups(coords):
    coords = np.asarray(coords, dtype=np.float32)
    bone = np.zeros_like(coords)
    for idx in range(1, 21):
        bone[:, idx] = coords[:, idx] - coords[:, PARENT_MAP[idx]]

    angles = np.zeros((coords.shape[0], 21, 1), dtype=np.float32)
    for a, b, c in ANGLE_TRIPLETS:
        v1 = coords[:, a] - coords[:, b]
        v2 = coords[:, c] - coords[:, b]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        cos_theta = np.sum(v1 * v2, axis=1) / denom
        angles[:, b, 0] = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    tip = np.zeros((coords.shape[0], 21, 4), dtype=np.float32)
    thumb = coords[:, 4]
    for out_idx, tip_idx in enumerate((8, 12, 16, 20)):
        dist = np.linalg.norm(thumb - coords[:, tip_idx], axis=1)
        tip[:, :, out_idx] = dist[:, None]

    return {
        "coordinates": coords,
        "bone vectors": bone,
        "joint angles": angles,
        "fingertip distances": tip,
    }


def column_minmax(values):
    arr = np.asarray(values, dtype=np.float32)
    lo = np.nanmin(arr, axis=0, keepdims=True)
    hi = np.nanmax(arr, axis=0, keepdims=True)
    return (arr - lo) / (hi - lo + 1e-8)


def load_experiment_config(path):
    from model import AdvancedAttentionModel, HybridAttentionModel, ImageOnlyBaseline, MLPBaseline

    model_class_registry = {
        "MLPBaseline": MLPBaseline,
        "AdvancedAttentionModel": AdvancedAttentionModel,
        "HybridAttentionModel": HybridAttentionModel,
        "ImageOnlyBaseline": ImageOnlyBaseline,
    }
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_class = config.get("model_class")
    if isinstance(model_class, str):
        config["model_class"] = model_class_registry[model_class]
    return config


def find_experiment_files(exp_dir, experiment):
    safe = experiment
    config = Path(exp_dir) / f"config_{safe}.json"
    weights = Path(exp_dir) / "weights" / f"best_model_{safe}_21kpts.pth"
    scaler = Path(exp_dir) / f"scaler_{safe}.pkl"
    encoder = Path(exp_dir) / f"label_encoder_{safe}.pkl"
    missing = [str(p) for p in (config, weights, scaler, encoder) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment files:\n" + "\n".join(missing))
    return config, weights, scaler, encoder


def build_hybrid_model_from_config(config, num_classes, image_feature_dim=None):
    from model import HybridAttentionModel

    if config["model_class"] is not HybridAttentionModel:
        raise ValueError("Diagnostic attention visualization requires HybridAttentionModel.")
    return HybridAttentionModel(
        input_dim=config.get("input_dim", 11),
        num_classes=num_classes,
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        embed_dim=config.get("embed_dim", 256),
        dropout=config.get("dropout", 0.2),
        image_feature_dim=image_feature_dim or config.get("image_feature_dim", 2048),
        visual_token_count=config.get("visual_token_count", 49),
        use_gated_fusion=config.get("use_gated_fusion", False),
        use_dynamic_graph=config.get("use_dynamic_graph", False),
        use_cross_attn=config.get("use_cross_attn", True),
        use_gcn_branch=config.get("use_gcn_branch", False),
        use_keypoint_refinement=config.get("use_keypoint_refinement", False),
        refine_scale=config.get("refine_scale", 0.05),
    )


def build_device(device_arg="auto"):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latest_summary_report(folder):
    matches = glob.glob(str(Path(folder) / "summary_report_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No summary_report_*.txt found in {folder}")
    return max(matches, key=os.path.getmtime)


def parse_summary_report(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "|" not in line or line.startswith("-") or "Experiment Name" in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 10:
                continue
            name = parts[0]
            if not name or name.startswith("="):
                continue
            rows.append({
                "Experiment": name,
                "Clean": parse_percent(parts[1]),
                "Macro F1": parse_percent(parts[2]),
                "Gaussian 1.0": parse_percent(parts[3]),
                "Occlusion_3": parse_percent(parts[4]),
                "Structured": parse_percent(parts[5]),
                "Visual Gaussian": parse_percent(parts[6]),
                "Token Dropout": parse_percent(parts[7]),
                "Patch Occlusion": parse_percent(parts[8]),
                "Joint Noise": parse_percent(parts[9]),
            })
    if not rows:
        raise ValueError(f"No experiment rows parsed from {path}")
    return pd.DataFrame(rows)


def parse_percent(text):
    match = re.search(r"([-+]?\d+(?:\.\d+)?)%", text)
    return float(match.group(1)) if match else np.nan


def write_rows_csv(path, rows, fieldnames=None):
    ensure_dir(Path(path).parent)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
