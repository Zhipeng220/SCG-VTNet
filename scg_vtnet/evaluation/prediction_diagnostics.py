import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd




DEFAULT_CSV_PATH = ASL_PROCESSED / "zerofill_landmarks_mask.csv"
DEFAULT_TOKEN_PATH = ASL_PROCESSED / "zerofill_visual_tokens_layer4_compact.npy"
DEFAULT_IMAGE_ROOT = ASL_PROCESSED / "zerofill_images_processed"
DEFAULT_MANIFEST_PATH = ASL_SPLITS / "split_manifest_seed42.csv"
DEFAULT_EXP_DIR = RUNS / "historical" / "visual_modulation" / "seed_42"


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


def load_config(exp_dir, experiment):
    path = Path(exp_dir) / f"config_{experiment}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def experiment_paths(exp_dir, experiment):
    exp_dir = Path(exp_dir)
    return {
        "config": exp_dir / f"config_{experiment}.json",
        "weights": exp_dir / "weights" / f"best_model_{experiment}_21kpts.pth",
    }


def feature_columns(df):
    cols = []
    for idx in range(21):
        for axis in ("x", "y", "z"):
            col = f"{axis}_{idx}"
            if col not in df.columns:
                raise ValueError(f"Missing landmark column: {col}")
            cols.append(col)
    return cols


def hand_detected_array(df):
    if "hand_detected" not in df.columns:
        return np.ones(len(df), dtype=np.float32)
    return df["hand_detected"].fillna(0).astype(np.float32).to_numpy()


def add_aar_features(coords, hand_detected=None, descriptor_mode="all"):
    coords = np.asarray(coords, dtype=np.float32)
    num_samples = coords.shape[0]
    if hand_detected is None:
        valid_samples = np.ones(num_samples, dtype=bool)
    else:
        valid_samples = np.asarray(hand_detected).astype(bool)

    bone = np.zeros_like(coords)
    for idx in range(1, 21):
        bone[:, idx] = coords[:, idx] - coords[:, PARENT_MAP[idx]]

    angles = np.zeros((num_samples, 21, 1), dtype=np.float32)
    for a, b, c in ANGLE_TRIPLETS:
        v1 = coords[:, a] - coords[:, b]
        v2 = coords[:, c] - coords[:, b]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        cos_theta = np.sum(v1 * v2, axis=1) / denom
        angles[:, b, 0] = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    tip = np.zeros((num_samples, 21, 4), dtype=np.float32)
    thumb = coords[:, 4]
    for out_idx, tip_idx in enumerate((8, 12, 16, 20)):
        dist = np.linalg.norm(thumb - coords[:, tip_idx], axis=1)
        tip[:, :, out_idx] = dist[:, None]

    mode = str(descriptor_mode or "all").lower()
    if mode in {"aar"}:
        mode = "all"
    if mode in {"none", "coord", "3d"}:
        out = coords.copy()
    else:
        parts = [coords]
        if mode in {"bone", "bone_angle", "all"}:
            parts.append(bone)
        if mode in {"angle", "bone_angle", "all"}:
            parts.append(angles)
        if mode in {"tip", "all"}:
            parts.append(tip)
        if mode not in {"bone", "angle", "tip", "bone_angle", "all"}:
            raise ValueError(f"Unknown descriptor_mode: {descriptor_mode}")
        out = np.concatenate(parts, axis=2)
    out[~valid_samples] = 0.0
    return out


def load_dataframe_and_arrays(csv_path, manifest_path, config):
    df = pd.read_csv(csv_path)
    manifest = pd.read_csv(manifest_path)
    cols = feature_columns(df)
    coords = df[cols].to_numpy(dtype=np.float32).reshape(len(df), 21, 3)
    hand_detected = hand_detected_array(df)
    if config.get("use_advanced_features", True):
        model_inputs = add_aar_features(coords, hand_detected, config.get("aar_descriptor", "all"))
    else:
        model_inputs = add_aar_features(coords, hand_detected, "none")

    flat = model_inputs.reshape(len(df), -1)
    train_idx = manifest.loc[manifest["split"] == "train", "original_index"].to_numpy(dtype=np.int64)
    fit_mask = hand_detected[train_idx] == 1
    fit_values = flat[train_idx][fit_mask] if np.any(fit_mask) else flat[train_idx]
    mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1.0
    scaled = ((flat - mean) / scale).astype(np.float32)
    scaled[hand_detected == 0] = 0.0

    classes = sorted(df["label"].astype(str).unique().tolist())
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    encoded_labels = df["label"].astype(str).map(class_to_id).to_numpy(dtype=np.int64)
    split_indices = {
        split: manifest.loc[manifest["split"] == split, "original_index"].to_numpy(dtype=np.int64)
        for split in ("train", "val", "test")
    }
    return {
        "df": df,
        "coords": coords,
        "hand_detected": hand_detected,
        "model_inputs": scaled.reshape(len(df), 21, -1),
        "raw_xyz": coords,
        "labels": encoded_labels,
        "classes": classes,
        "split_indices": split_indices,
    }


def build_model(config, num_classes):
    from scg_vtnet.models.scg_vtnet import HybridAttentionModel

    if config.get("model_class") != "HybridAttentionModel":
        raise ValueError("Diagnostics currently require HybridAttentionModel checkpoints.")
    return HybridAttentionModel(
        input_dim=config.get("input_dim", 11),
        num_classes=num_classes,
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        embed_dim=config.get("embed_dim", 256),
        dropout=config.get("dropout", 0.2),
        image_feature_dim=config.get("image_feature_dim", 2048),
        visual_token_count=config.get("visual_token_count", 49),
        use_gated_fusion=config.get("use_gated_fusion", False),
        use_dynamic_graph=config.get("use_dynamic_graph", False),
        use_cross_attn=config.get("use_cross_attn", True),
        use_gcn_branch=config.get("use_gcn_branch", False),
        use_keypoint_refinement=config.get("use_keypoint_refinement", False),
        refine_scale=config.get("refine_scale", 0.05),
    )


def load_checkpoint_model(config, weights_path, num_classes, device):
    import torch

    model = build_model(config, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def choose_device(device_arg):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def resolve_image_path(image_root, image_path):
    path = Path(str(image_path))
    if path.is_absolute() and path.exists():
        return path
    candidate = Path(image_root) / Path(str(image_path).replace("\\", os.sep))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image not found for {image_path}; tried {candidate}")
