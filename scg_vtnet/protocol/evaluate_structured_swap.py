"""Evaluate a structured index/middle-finger landmark swap on matched F/A0 checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 2026112314
KEYS = ("label", "image_path", "hand_detected")


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

    base = args.baselines.resolve()
    inputs = args.processed_dir.resolve()
    runs = args.runs.resolve()
    manifest = args.manifest.resolve()
    selected_path = args.sample.resolve()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(base))
    from model import HybridAttentionModel
    from train import forward_model, parse_model_outputs, unpack_batch
    from utils import IndexedMultimodalDataset, add_advanced_features, get_feature_columns

    paths = {
        "manifest": manifest,
        "sample": selected_path,
        "landmarks": inputs / "zerofill_landmarks_mask.csv",
        "tokens": inputs / "zerofill_visual_tokens_layer4_compact.npy",
    }
    expected = None
    if args.expected_sha256 is not None and not args.skip_hash_check:
        expected = json.loads(args.expected_sha256.read_text(encoding="utf-8"))
        for role, path in paths.items():
            assert sha256(path) == expected[role], role
        print("Frozen manifest, landmarks, and visual token array verified.", flush=True)

    selected = pd.read_csv(selected_path).sort_values("original_index")
    assert len(selected) == 213 and selected["original_index"].is_unique and selected["label"].nunique() == 26
    split = pd.read_csv(manifest).set_index("original_index", verify_integrity=True)
    ids = selected["original_index"].to_numpy()
    frozen = split.loc[ids]
    assert (frozen["split"] == "test").all() and (frozen["hand_detected"] == 1).all()

    landmarks = pd.read_csv(paths["landmarks"])
    inputs_subset = landmarks.iloc[ids].copy()
    inputs_subset["image_path"] = inputs_subset["image_path"].str.replace(chr(92), "/", regex=False)
    for rows in (inputs_subset, frozen):
        assert np.array_equal(selected[list(KEYS)].to_numpy(), rows[list(KEYS)].to_numpy())

    truth = selected["label"].to_numpy()
    hand = selected["hand_detected"].to_numpy(dtype=np.float32)
    xyz = inputs_subset[get_feature_columns(landmarks)].to_numpy().reshape(-1, 21, 3)
    assert np.isfinite(xyz).all()

    swapped = xyz.copy()
    swapped[:, 5:9] = xyz[:, 9:13]
    swapped[:, 9:13] = xyz[:, 5:9]
    unchanged = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20]
    assert np.array_equal(xyz[:, unchanged], swapped[:, unchanged])
    assert np.all((xyz != swapped).any(axis=(1, 2)))

    coordinates = {"clean": xyz, "swap": swapped}
    advanced = {key: add_advanced_features(value, hand, "all") for key, value in coordinates.items()}
    feature_map = np.load(paths["tokens"], mmap_mode="r", allow_pickle=False)
    assert feature_map.shape == (87000, 49, 2048)

    device = resolve_device(torch, args.device)
    torch.set_num_threads(2)
    print(f"Running {len(ids)} paired sample rows on {device}.", flush=True)

    output = selected.copy()
    by_model = {}
    summary = []
    used = {role: sha256(path) for role, path in paths.items()}

    for variant in ("F", "A0"):
        path = runs / variant
        files = {
            "config": path / f"config_{variant}.json",
            "weights": path / "weights" / f"best_model_{variant}_21kpts.pth",
            "scaler": path / f"scaler_{variant}.pkl",
            "encoder": path / f"label_encoder_{variant}.pkl",
            "reference": path / "predictions_clean.csv",
        }
        used[variant] = {role: sha256(file) for role, file in files.items()}
        if expected is not None:
            assert used[variant] == expected[variant], variant
        config = json.loads(files["config"].read_text(encoding="utf-8"))
        assert config["train_args"]["seed"] == 42 and config["use_keypoint_refinement"]
        assert config["use_anatomical_loss"] == (variant == "F") and config["aar_descriptor"] == "all"

        scaler = joblib.load(files["scaler"])
        encoder = joblib.load(files["encoder"])
        ks = (
            "input_dim", "image_feature_dim", "visual_token_count", "embed_dim",
            "num_layers", "num_heads", "dropout", "use_cross_attn", "use_gated_fusion",
            "use_gcn_branch", "use_dynamic_graph", "use_keypoint_refinement", "refine_scale",
        )
        model = HybridAttentionModel(
            num_classes=len(encoder.classes_),
            **{k: config[k] for k in ks if k in config},
        )
        model.load_state_dict(torch.load(files["weights"], map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()

        for condition in ("clean", "swap"):
            points = advanced[condition]
            points = scaler.transform(points.reshape(len(ids), -1)).reshape(points.shape)
            dataset = IndexedMultimodalDataset(
                points, coordinates[condition].astype(np.float32),
                feature_map, ids, hand, encoder.transform(truth),
            )
            pred = []
            with torch.inference_mode():
                for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
                    k, r, v, m, _ = unpack_batch(batch, device)
                    logits, *_ = parse_model_outputs(forward_model(model, k, v, r, m))
                    pred.extend(logits.argmax(dim=1).cpu().tolist())
            output[f"{variant}_{condition}"] = encoder.inverse_transform(pred)

        reference = pd.read_csv(files["reference"]).set_index("original_index", verify_integrity=True).loc[ids]
        assert np.array_equal(reference["true_label"].to_numpy(), truth) and (reference["hand_detected"] == 1).all()
        assert np.array_equal(reference["predicted_label"].to_numpy(), output[f"{variant}_clean"].to_numpy()), variant

        before = output[f"{variant}_clean"].to_numpy() == truth
        after = output[f"{variant}_swap"].to_numpy() == truth
        drop = before.astype(int) - after.astype(int)
        by_model[variant] = drop
        summary.append({
            "variant": variant,
            "n": len(ids),
            "clean_correct": int(before.sum()),
            "swap_correct": int(after.sum()),
            "clean_accuracy_pct": 100 * float(before.mean()),
            "swap_accuracy_pct": 100 * float(after.mean()),
            "drop_pp": 100 * float(drop.mean()),
            "correct_to_wrong": int((before & ~after).sum()),
            "wrong_to_correct": int((~before & after).sum()),
        })
        print(json.dumps(summary[-1]), flush=True)

    delta = by_model["F"] - by_model["A0"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(N_BOOTSTRAP)
    for label in sorted(set(truth)):
        strata = delta[truth == label]
        draws += rng.choice(strata, size=(N_BOOTSTRAP, len(strata)), replace=True).sum(axis=1)
    low, high = np.percentile(100 * draws / len(ids), [2.5, 97.5])

    protocol = {
        "name": "Frozen structured landmark correspondence swap",
        "source": "ASL Alphabet seed 42 held-out",
        "index_middle_swap_pairs": [[5, 9], [6, 10], [7, 11], [8, 12]],
        "visual_features_and_detection_fixed": True,
        "aar_recomputed": True,
        "n": len(ids),
        "bootstrap": {
            "paired": True,
            "stratification": "true letter",
            "resamples": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile",
        },
        "contrast": "accuracy drop F minus accuracy drop A0; positive means F loses more",
        "F_minus_A0_drop_pp": 100 * float(delta.mean()),
        "ci95_pp": [float(low), float(high)],
        "device": str(device),
        "torch": str(torch.__version__),
        "file_sha256": used,
        "baselines_sha256": {name: sha256(base / name) for name in ("model.py", "train.py", "utils.py")},
    }
    output.to_csv(folder / "paired_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    (folder / "protocol_and_contrast.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("F-A0 drop difference and CI:", protocol["F_minus_A0_drop_pp"], [float(low), float(high)], flush=True)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=Path(__file__).resolve().parents[1] / "baselines")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/asl_alphabet/processed_train",
                        help="Processed ASL Alphabet directory containing landmarks CSV and spatial token .npy.")
    parser.add_argument("--runs", type=Path, default=root / "results/asl_matched/seed_42",
                        help="Seed-42 matched ASL directory containing F/ and A0/ checkpoints.")
    parser.add_argument("--manifest", type=Path, default=root / "manifests/asl/split_manifest_seed42.csv")
    parser.add_argument("--sample", type=Path,
                        default=root / "results/reviewer11_14/r14_structured_swap/sample.csv",
                        help="213-image A-Z detected sample CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for swap predictions and summary.")
    parser.add_argument("--expected-sha256", type=Path, default=None,
                        help="Optional expected_sha256.json manifest to verify input and checkpoint digests.")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
