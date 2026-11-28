"""Evaluate matched ASL checkpoints with clean or zeroed skeleton inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44)
VARIANTS = ("F", "A0", "G0", "Uniform")
MODEL_OPTIONS = (
    "input_dim", "image_feature_dim", "visual_token_count", "dropout",
    "embed_dim", "num_layers", "num_heads", "use_gated_fusion",
    "use_dynamic_graph", "use_cross_attn", "use_gcn_branch",
    "use_keypoint_refinement", "refine_scale", "uniform_visual_attention",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(torch_mod, name):
    if name == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(name)


def run(args):
    import joblib
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(args.baselines.resolve()))
    from model import HybridAttentionModel
    from train import forward_model, parse_model_outputs, unpack_batch
    from utils import IndexedMultimodalDataset, add_advanced_features, get_feature_columns

    torch.set_num_threads(2)
    device = resolve_device(torch, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        "landmarks": sha256(args.landmarks),
        "visual_tokens": sha256(args.features),
    }
    print("Input SHA-256:", source_hashes, flush=True)
    landmarks = pd.read_csv(args.landmarks)
    columns = get_feature_columns(landmarks)
    features = np.load(args.features, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(landmarks), 49, 2048):
        raise ValueError(f"Unexpected visual features: {features.shape}")
    summary = []
    paired = []
    digests = {}
    for seed in SEEDS:
        manifest_path = args.manifests / f"split_manifest_seed{seed}.csv"
        manifest = pd.read_csv(manifest_path)
        manifest_digest = sha256(manifest_path)
        if (len(manifest) != 87000
                or not np.array_equal(np.sort(manifest["original_index"]), np.arange(len(landmarks)))
                or manifest["split"].value_counts().to_dict() != {"train": 52200, "val": 17400, "test": 17400}):
            raise ValueError(f"Invalid frozen split {seed}")
        test = manifest.loc[manifest["split"] == "test"]
        selected = test.loc[test["hand_detected"] == 1].sort_values("original_index")
        indices = selected["original_index"].to_numpy()
        samples = landmarks.iloc[indices]
        if (not np.array_equal(samples["label"], selected["label"])
                or not np.array_equal(samples["hand_detected"], selected["hand_detected"])
                or not np.array_equal(samples["image_path"].str.replace("\\", "/", regex=False),
                                      selected["image_path"].str.replace("\\", "/", regex=False))):
            raise ValueError(f"Landmark rows disagree with manifest {seed}")
        labels = samples["label"].to_numpy()
        hand = samples["hand_detected"].to_numpy(dtype=np.float32)
        xyz = samples[columns].to_numpy().reshape(-1, 21, 3)
        points = add_advanced_features(xyz, hand, "all")
        raw = xyz.astype(np.float32)
        clean_correct_by_variant = {}
        zero_correct_by_variant = {}
        for variant in VARIANTS:
            path = args.runs / f"seed_{seed}" / variant
            config_path = path / f"config_{variant}.json"
            checkpoint = path / "weights" / f"best_model_{variant}_21kpts.pth"
            scaler_path = path / f"scaler_{variant}.pkl"
            encoder_path = path / f"label_encoder_{variant}.pkl"
            reference_path = path / "predictions_clean.csv"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (config["safe_name"] != variant
                    or config["model_class"] != "HybridAttentionModel"
                    or config["frozen_manifest_sha256"] != manifest_digest
                    or config["aar_descriptor"] != "all"
                    or not config["use_advanced_features"]
                    or config.get("uniform_visual_attention", False) != (variant == "Uniform")):
                raise ValueError(f"Unexpected configuration: {config_path}")
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            scaled = scaler.transform(points.reshape(len(points), -1)).reshape(points.shape)
            encoded = encoder.transform(labels)
            dataset = IndexedMultimodalDataset(scaled, raw, features, indices, hand, encoded)
            model = HybridAttentionModel(
                num_classes=len(encoder.classes_),
                **{key: config[key] for key in MODEL_OPTIONS if key in config},
            )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.to(device).eval()
            clean_prediction = []
            zero_prediction = []
            with torch.inference_mode():
                for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
                    keypoints, raw_xyz, visual, detection, _ = unpack_batch(batch, device)
                    clean, *_ = parse_model_outputs(forward_model(model, keypoints, visual, raw_xyz, detection))
                    zero, *_ = parse_model_outputs(forward_model(
                        model, torch.zeros_like(keypoints), visual, torch.zeros_like(raw_xyz),
                        torch.zeros_like(detection),
                    ))
                    clean_prediction.extend(clean.argmax(dim=1).cpu().tolist())
                    zero_prediction.extend(zero.argmax(dim=1).cpu().tolist())
            clean_labels = encoder.inverse_transform(clean_prediction)
            zero_labels = encoder.inverse_transform(zero_prediction)
            reference = pd.read_csv(reference_path).set_index("original_index", verify_integrity=True)
            reference_detected = reference.loc[reference["hand_detected"] == 1].sort_index()
            if (not np.array_equal(reference_detected.index, indices)
                    or not np.array_equal(reference_detected["true_label"], labels)):
                raise ValueError(f"Clean reference rows disagree for {seed}/{variant}")
            mismatch = np.count_nonzero(clean_labels != reference_detected["predicted_label"].to_numpy())
            if mismatch:
                raise ValueError(f"{seed}/{variant}: clean replay differs for {mismatch} of {len(labels)} images")
            clean_correct = clean_labels == labels
            zero_correct = zero_labels == labels
            clean_correct_by_variant[variant] = clean_correct
            zero_correct_by_variant[variant] = zero_correct
            pd.DataFrame({
                "original_index": indices, "true_label": labels,
                "clean_prediction": clean_labels, "zero_prediction": zero_labels,
            }).to_csv(args.output / f"seed_{seed}_{variant}.csv", index=False)
            row = dict(seed=seed, variant=variant, n=len(indices),
                       clean_correct=int(clean_correct.sum()), zero_correct=int(zero_correct.sum()),
                       clean_accuracy_pct=100 * clean_correct.mean(),
                       zero_accuracy_pct=100 * zero_correct.mean(),
                       zero_minus_clean_pp=100 * (zero_correct.mean() - clean_correct.mean()),
                       clean_replay_disagreements=int(mismatch))
            summary.append(row)
            digests[f"{seed}/{variant}"] = {
                "manifest": manifest_digest, "config": sha256(config_path),
                "checkpoint": sha256(checkpoint), "scaler": sha256(scaler_path),
                "label_encoder": sha256(encoder_path), "clean_reference": sha256(reference_path),
            }
            print(json.dumps(row), flush=True)
            del model
        for comparator in VARIANTS[1:]:
            for condition, group in (("clean", clean_correct_by_variant), ("zero", zero_correct_by_variant)):
                first, second = group["F"], group[comparator]
                paired.append(dict(seed=seed, condition=condition, comparator=comparator, n=len(first),
                                   both_correct=int((first & second).sum()),
                                   only_F_correct=int((first & ~second).sum()),
                                   only_comparator_correct=int((~first & second).sum()),
                                   both_wrong=int((~first & ~second).sum()),
                                   F_minus_comparator_pp=100 * (first.mean() - second.mean())))
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output / "paired.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        torch_version=str(torch.__version__),
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        batch_size=args.batch_size,
        source_sha256=source_hashes, runs=digests,
        baselines_sha256={name: sha256(args.baselines / name) for name in ("model.py", "train.py", "utils.py")},
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=Path(__file__).resolve().parents[1] / "baselines")
    parser.add_argument("--landmarks", type=Path,
                        default=root / "data/asl_alphabet/processed_train/zerofill_landmarks_mask.csv")
    parser.add_argument("--features", type=Path,
                        default=root / "data/asl_alphabet/processed_train/zerofill_visual_tokens_layer4_compact.npy")
    parser.add_argument("--manifests", type=Path, default=root / "manifests/asl")
    parser.add_argument("--runs", type=Path, default=root / "results/asl_matched")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())
