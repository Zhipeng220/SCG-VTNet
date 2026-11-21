import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scg_vtnet.evaluation.feature_perturbations import corrupt_tokens, stable_scenario_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from scg_vtnet.evaluation.figure_components import (
    DEFAULT_CSV_PATH,
    DEFAULT_TOKEN_PATH,
    DEFAULT_VISUAL_MODULATION_DIR,
    build_device,
    build_hybrid_model_from_config,
    compute_descriptor_groups,
    coords_from_frame,
    find_experiment_files,
    load_experiment_config,
    read_landmark_csv,
)


EXPERIMENTS = {
    "A": "Global concat",
    "B": "Global CA",
    "C": "Token concat",
    "D": "Token CA",
    "E": "Token CA + gate",
}
SCENARIOS = {
    "Clean": {},
    "Visual Gaussian": {"visual_sigma": 0.1},
    "Token Dropout": {"visual_token_dropout": 0.25},
    "Patch Occlusion": {"visual_patch_occlusion": 0.25},
    "Joint Noise": {"feature_sigma": 0.5, "raw_sigma": 0.005, "visual_sigma": 0.1},
}


def prepare_test_split(df, seed):
    labels = LabelEncoder().fit_transform(df["label"].astype(str).to_numpy())
    indices = np.arange(len(df))
    _, idx_test, _, _ = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(idx_test, dtype=np.int64)


def build_keypoint_inputs(df, indices, scaler):
    rows = df.iloc[indices].copy()
    coords = coords_from_frame(rows)
    groups = compute_descriptor_groups(coords)
    engineered = np.concatenate(
        [groups["coordinates"], groups["bone vectors"], groups["joint angles"], groups["fingertip distances"]],
        axis=2,
    )
    flat = engineered.reshape(engineered.shape[0], -1)
    hand_detected = rows.get("hand_detected", pd.Series(np.ones(len(rows)))).fillna(0).astype(np.float32).to_numpy()
    scaled = scaler.transform(flat)
    scaled[hand_detected == 0] = 0.0
    return (
        scaled.reshape(engineered.shape[0], 21, engineered.shape[2]).astype(np.float32),
        coords.astype(np.float32),
        hand_detected,
        rows["label"].astype(str).to_numpy(),
    )




def forward_model(model, keypoints, image_features, raw_xyz, hand_detected):
    outputs = model(keypoints, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs[0]




def evaluate_one(model, loader, scenario, visual_feature, device, seed):
    all_preds = []
    all_labels = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for keypoints, raw_xyz, tokens, hand_detected, labels in loader:
            keypoints = keypoints.to(device)
            raw_xyz = raw_xyz.to(device)
            tokens = tokens.to(device)
            hand_detected = hand_detected.to(device)
            labels = labels.to(device)
            noisy_tokens, noisy_keypoints, noisy_raw_xyz = corrupt_tokens(
                tokens, keypoints, raw_xyz, scenario, generator
            )
            valid = hand_detected.reshape(-1, 1, 1).to(noisy_keypoints.dtype)
            noisy_keypoints = noisy_keypoints * valid
            noisy_raw_xyz = noisy_raw_xyz * valid
            image_features = noisy_tokens if visual_feature == "token" else noisy_tokens.mean(dim=1)
            logits = forward_model(model, noisy_keypoints, image_features, noisy_raw_xyz, hand_detected)
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(labels.detach().cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return float((preds == labels).mean() * 100.0)


def load_model_bundle(exp_dir, experiment, device):
    config_path, weights_path, scaler_path, encoder_path = find_experiment_files(exp_dir, experiment)
    config = load_experiment_config(config_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    num_classes = len(encoder.classes_)
    model = build_hybrid_model_from_config(config, num_classes, image_feature_dim=config.get("image_feature_dim", 2048))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    return config, scaler, encoder, model


def evaluate_fair_corruption(args):
    device = build_device(args.device)
    df = read_landmark_csv(args.csv_path)
    tokens_np = np.load(args.token_features_path, mmap_mode="r")
    if tokens_np.ndim != 3 or tokens_np.shape[1] != 49:
        raise ValueError(f"Expected token features with shape (N, 49, C), got {tokens_np.shape}")

    test_indices = prepare_test_split(df, args.seed)
    if args.split_manifest is not None:
        manifest = pd.read_csv(args.split_manifest)
        held_out = manifest.loc[manifest["split"] == "test"]
        test_indices = held_out["original_index"].to_numpy(dtype=np.int64)
        if (not len(test_indices) or held_out["original_index"].duplicated().any()
                or (test_indices < 0).any() or (test_indices >= len(df)).any()
                or not np.array_equal(df.iloc[test_indices]["label"].astype(str).to_numpy(),
                                      held_out["label"].astype(str).to_numpy())):
            raise ValueError("The frozen test manifest does not match the landmark rows.")
    if args.max_samples:
        test_indices = test_indices[:args.max_samples]

    rows = []
    clean_acc = {}
    for experiment, display in EXPERIMENTS.items():
        config, scaler, encoder, model = load_model_bundle(args.visual_modulation_dir, experiment, device)
        keypoints, raw_xyz, hand_detected, labels_str = build_keypoint_inputs(df, test_indices, scaler)
        labels = encoder.transform(labels_str).astype(np.int64)
        tokens = np.asarray(tokens_np[test_indices], dtype=np.float32)

        dataset = TensorDataset(
            torch.from_numpy(keypoints),
            torch.from_numpy(raw_xyz),
            torch.from_numpy(tokens),
            torch.from_numpy(hand_detected.astype(np.float32)),
            torch.from_numpy(labels),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        visual_feature = config.get("visual_feature", "token" if config.get("visual_token_count", 49) == 49 else "global")

        for scenario_name, scenario_config in SCENARIOS.items():
            acc = evaluate_one(
                model,
                loader,
                scenario_config,
                visual_feature,
                device,
                seed=stable_scenario_seed(args.eval_seed, experiment, scenario_name),
            )
            if scenario_name == "Clean":
                clean_acc[experiment] = acc
            rows.append({
                "experiment": experiment,
                "model": display,
                "visual_feature": visual_feature,
                "scenario": scenario_name,
                "clean_acc": clean_acc.get(experiment, acc),
                "corrupted_acc": acc,
                "accuracy_drop": clean_acc.get(experiment, acc) - acc,
            })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path, rows


def dry_run(args):
    token_shape = np.load(args.token_features_path, mmap_mode="r").shape
    found = sorted(Path(args.visual_modulation_dir).glob("config_*.json"))
    missing = []
    for experiment in EXPERIMENTS:
        for suffix in (
            f"config_{experiment}.json",
            f"scaler_{experiment}.pkl",
            f"label_encoder_{experiment}.pkl",
            f"weights/best_model_{experiment}_21kpts.pth",
        ):
            path = Path(args.visual_modulation_dir) / suffix
            if not path.exists():
                missing.append(str(path))
    print(f"Token features: {args.token_features_path} shape={token_shape}")
    print(f"Config files: {[path.name for path in found]}")
    print(f"CSV: {args.csv_path}")
    print(f"Seed: {args.seed}")
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print("Dry run OK: A-E configs, weights, scalers, encoders are present.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fair Figure 3 visual corruption evaluation.")
    parser.add_argument("--visual_modulation_dir", default=DEFAULT_VISUAL_MODULATION_DIR)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--token_features_path", default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--output_csv", default="../runs/asl_alphabet/feature_robustness/metrics.csv")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    out_path, rows = evaluate_fair_corruption(args)
    print(f"Saved fair corruption results to: {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
