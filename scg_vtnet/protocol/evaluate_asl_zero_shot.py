"""Zero-shot ASL Alphabet to ASL Dataset transfer with frozen source heads."""

import argparse
import hashlib
import json
from pathlib import Path
import string
import sys

import numpy as np
import pandas as pd

LETTERS = tuple(string.ascii_uppercase)
HEADS = {"E2": "E2_EfficientNet-B0", "E5": "E5_Late_Fusion", "E7": "E7_AC-GCAT"}
MODEL_FIELDS = (
    "input_dim", "image_feature_dim", "visual_token_count", "dropout", "embed_dim",
    "num_layers", "num_heads", "use_gated_fusion", "use_dynamic_graph", "use_cross_attn",
    "use_gcn_branch", "use_keypoint_refinement", "refine_scale",
)


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_source_files(source_runs, seed, name):
    """Locate source checkpoint artifacts in standard or flat seed layouts."""
    seed_dir = source_runs / f"seed_{seed}"
    config_path = seed_dir / f"config_{name}.json"
    scaler_path = seed_dir / f"scaler_{name}.pkl"
    encoder_path = seed_dir / f"label_encoder_{name}.pkl"
    checkpoint = seed_dir / "weights" / f"best_model_{name}_21kpts.pth"
    if not checkpoint.is_file():
        checkpoint = seed_dir / f"best_model_{name}_21kpts.pth"
    for path in (config_path, checkpoint, scaler_path, encoder_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source artifact: {path}")
    return config_path, checkpoint, scaler_path, encoder_path


def build_e2_features(output, landmark_path, raw, indices, row_count, extraction_dir, pretrained_weights=None):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    sys.path.insert(0, str(extraction_dir.resolve()))
    from extract_rgb_backbone_features import CsvImageDataset, forward_features

    weights = EfficientNet_B0_Weights.DEFAULT
    if pretrained_weights is not None:
        visual_model = efficientnet_b0(weights=None)
        visual_model.load_state_dict(
            torch.load(pretrained_weights, map_location="cpu", weights_only=True),
            strict=True,
        )
    else:
        visual_model = efficientnet_b0(weights=weights)
    visual_model.classifier = torch.nn.Identity()
    visual_model.eval()
    dataset = Subset(CsvImageDataset(landmark_path, raw, weights.transforms()), indices)
    rgb = np.zeros((row_count, 1280), dtype=np.float16)
    extracted = 0
    for images, rows in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
        rgb[rows.numpy()] = forward_features(visual_model, images, "efficientnet_b0").numpy()
        extracted += len(rows)
        if extracted % 256 == 0 or extracted == len(indices):
            print(f"E2 target features: {extracted}/{len(indices)}", flush=True)
    del visual_model
    assert np.isfinite(rgb[indices]).all()
    target_path = output / "efficientnet_target_letters.npy"
    temp_path = output / "efficientnet_target_letters.next.npy"
    np.save(temp_path, rgb[indices])
    temp_path.replace(target_path)
    return rgb


def run(args):
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = args.target_dir.resolve()
    landmark_path = target / "zerofill_landmarks_mask.csv"
    token_path = target / "zerofill_visual_tokens_layer4_compact.npy"

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    landmarks = pd.read_csv(landmark_path, dtype={"label": str})
    target_labels = landmarks["label"].str.upper()
    assert len(landmarks) == 2515
    assert not landmarks["image_path"].duplicated().any()
    assert set(target_labels) == set(LETTERS) | set(string.digits)
    indices = np.flatnonzero(target_labels.isin(LETTERS).to_numpy())

    if args.prepare_e2_only:
        if args.raw_images is None:
            raise ValueError("--raw-images is required with --prepare-e2-only")
        build_e2_features(
            output, landmark_path, args.raw_images.resolve(),
            indices, len(landmarks), args.extraction, args.pretrained_weights,
        )
        return

    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(args.baselines.resolve()))
    from model import HybridAttentionModel, ImageOnlyBaseline
    from train import forward_model, parse_model_outputs, unpack_batch
    from utils import IndexedMultimodalDataset, add_advanced_features, get_feature_columns

    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    assert tokens.shape == (2515, 49, 2048)

    subset = landmarks.iloc[indices]
    truth = target_labels.iloc[indices].to_numpy()
    paths = subset["image_path"].str.replace(chr(92), "/", regex=False).to_numpy()
    assert all(Path(str(path)).parts[0].upper() == label for path, label in zip(paths, truth))
    hand = subset["hand_detected"].to_numpy(dtype=np.float32)
    assert set(hand) <= {0, 1}
    xyz = subset[get_feature_columns(landmarks)].to_numpy().reshape(-1, 21, 3)
    raw_xyz = xyz.astype(np.float32)
    raw_xyz[hand == 0] = 0
    points = add_advanced_features(xyz, hand, "all")

    print(f"Target: {len(indices)} A-Z images, {len(landmarks) - len(indices)} digit images excluded", flush=True)
    pd.DataFrame({
        "original_index": indices,
        "image_path": paths,
        "true_label": truth,
        "hand_detected": hand.astype(int),
    }).to_csv(output / "target_letters.csv", index=False)

    e5_validation_rows = 0
    if args.source_features is not None:
        source_dir = args.source_features.resolve()
        source_tokens = np.load(source_dir / "zerofill_visual_tokens_layer4_compact.npy", mmap_mode="r")
        source_pool = np.load(source_dir / "zerofill_visual_global_from_tokens.npy", mmap_mode="r")
        source_sample = np.linspace(0, len(source_tokens) - 1, 96, dtype=int)
        recomputed = source_tokens[source_sample].astype(np.float32).mean(axis=1).astype(source_pool.dtype)
        assert np.array_equal(recomputed, source_pool[source_sample]), "E5 pooling mismatch on source samples"
        e5_validation_rows = len(source_sample)
        del source_tokens, source_pool

    pool = np.zeros((len(landmarks), 2048), dtype=np.float16)
    for chunk in np.array_split(indices, 20):
        pool[chunk] = tokens[chunk].astype(np.float32).mean(axis=1)

    e2_path = args.e2_features.resolve() if args.e2_features is not None else (output / "efficientnet_target_letters.npy")
    if e2_path.is_file():
        rgb = np.zeros((len(landmarks), 1280), dtype=np.float16)
        cached = np.load(e2_path, allow_pickle=False)
        assert cached.shape == (len(indices), 1280) and cached.dtype == np.float16 and np.isfinite(cached).all()
        rgb[indices] = cached
        print(f"Using pre-extracted E2 target features: {e2_path}", flush=True)
    else:
        if args.raw_images is None:
            raise ValueError("Missing cached E2 features; supply --e2-features or --raw-images to extract them")
        rgb = build_e2_features(
            output, landmark_path, args.raw_images.resolve(),
            indices, len(landmarks), args.extraction, args.pretrained_weights,
        )
        e2_path = output / "efficientnet_target_letters.npy"

    features = {"E2": rgb, "E5": pool, "E7": tokens}
    protocol = {
        "source": "ASL Alphabet historical training splits 42-46",
        "target": "All A-Z rows of ASL Dataset; target digits excluded",
        "primary": "29-class source argmax; del/space/nothing predictions count as errors",
        "sensitivity": "argmax restricted to the 26 shared letters",
        "target_n": len(indices),
        "detected_n": int(hand.sum()),
        "class_counts": pd.Series(truth).value_counts().to_dict(),
        "labels": {"target": "lowercase letters mapped to uppercase", "source": list(LETTERS)},
        "target_landmarks_sha256": sha256(landmark_path),
        "target_tokens_sha256": sha256(token_path),
        "target_e2_sha256": sha256(e2_path),
        "e5_pool_validation_source_rows": e5_validation_rows,
        "torch_version": str(torch.__version__),
        "training_or_target_fitting": False,
    }
    if args.pretrained_weights is not None:
        protocol["e2_backbone_sha256"] = sha256(args.pretrained_weights.resolve())
    write_json(output / "protocol.json", protocol)

    expected = None
    if args.expected_source is not None:
        expected = json.loads(args.expected_source.read_text(encoding="utf-8"))

    source_runs = args.source_runs.resolve()
    source_hashes = {}
    summaries = []
    for seed in range(42, 47):
        for key, name in HEADS.items():
            config_path, checkpoint, scaler_path, encoder_path = resolve_source_files(source_runs, seed, name)
            for path in (config_path, checkpoint, scaler_path, encoder_path):
                digest_val = sha256(path)
                rel_key = f"source/seed_{seed}/{path.name}"
                source_hashes[rel_key] = digest_val
                if expected is not None and rel_key in expected:
                    exp_entry = expected[rel_key]
                    exp_hash = exp_entry["sha256"] if isinstance(exp_entry, dict) else exp_entry
                    assert digest_val == exp_hash, f"Source hash mismatch for {path}"
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            assert cfg["method_id"] == key and cfg["aar_descriptor"] == "all"
            assert features[key].shape[1:] == tuple(cfg["image_feature_shape"])
            encoder = joblib.load(encoder_path)
            assert set(encoder.classes_) == set(LETTERS) | {"del", "space", "nothing"}
            letter_indices = encoder.transform(list(LETTERS))
            scaler = joblib.load(scaler_path)
            scaled = scaler.transform(points.reshape(len(points), -1)).reshape(points.shape)
            scaled[hand == 0] = 0
            dset = IndexedMultimodalDataset(scaled, raw_xyz, features[key], indices, hand, encoder.transform(truth))
            if key == "E2":
                assert cfg["model_class"] == "ImageOnlyBaseline"
                model = ImageOnlyBaseline(image_feature_dim=1280, num_classes=len(encoder.classes_), dropout=cfg["dropout"])
            else:
                assert cfg["model_class"] == "HybridAttentionModel"
                model = HybridAttentionModel(
                    num_classes=len(encoder.classes_),
                    **{k: cfg[k] for k in MODEL_FIELDS if k in cfg},
                )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.to(device).eval()
            predictions, restricted = [], []
            with torch.inference_mode():
                for batch in DataLoader(dset, batch_size=args.batch_size, shuffle=False):
                    points_batch, xyz_batch, vision, detected, _ = unpack_batch(batch, device)
                    logits, *_ = parse_model_outputs(forward_model(model, points_batch, vision, xyz_batch, detected))
                    logits_cpu = logits.cpu()
                    predictions.extend(logits_cpu.argmax(dim=1).tolist())
                    restricted.extend(letter_indices[logits_cpu[:, letter_indices].argmax(dim=1).numpy()])
            clean = encoder.inverse_transform(predictions)
            shared = encoder.inverse_transform(restricted)
            pd.DataFrame({
                "original_index": indices,
                "true_label": truth,
                "hand_detected": hand.astype(int),
                "prediction_29": clean,
                "prediction_26": shared,
            }).to_csv(output / f"predictions_{key}_{seed}.csv", index=False)
            row = {
                "source_seed": seed,
                "method": key,
                "n": len(truth),
                "accuracy_29_pct": 100 * accuracy_score(truth, clean),
                "macro_f1_29_pct": 100 * f1_score(truth, clean, labels=LETTERS, average="macro", zero_division=0),
                "non_letter_predictions": int(np.count_nonzero(~np.isin(clean, LETTERS))),
                "accuracy_26_pct": 100 * accuracy_score(truth, shared),
                "macro_f1_26_pct": 100 * f1_score(truth, shared, labels=LETTERS, average="macro", zero_division=0),
            }
            summaries.append(row)
            print(json.dumps(row), flush=True)
            del model

    write_json(output / "source_hashes.json", source_hashes)
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "per_seed.csv", index=False)
    metrics = ["accuracy_29_pct", "macro_f1_29_pct", "accuracy_26_pct", "macro_f1_26_pct"]
    frame.groupby("method")[metrics].agg(["mean", "std"]).to_csv(output / "summary.csv")


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output directory for zero-shot predictions.")
    parser.add_argument("--target-dir", type=Path, required=True,
                        help="Processed ASL Dataset directory containing zerofill_landmarks_mask.csv and tokens.")
    parser.add_argument("--source-runs", type=Path, default=root / "results/historical/fair_e1_e7",
                        help="Directory containing seed_42..seed_46 source checkpoints, configs, scalers and encoders.")
    parser.add_argument("--source-features", type=Path, default=None,
                        help="Optional processed ASL Alphabet directory to verify E5 global pooling parity.")
    parser.add_argument("--raw-images", type=Path, default=None,
                        help="Raw ASL Dataset image root (needed if cached E2 target features are not supplied).")
    parser.add_argument("--e2-features", type=Path, default=None,
                        help="Optional path to pre-extracted efficientnet_target_letters.npy.")
    parser.add_argument("--pretrained-weights", type=Path, default=None,
                        help="Optional local torchvision EfficientNet-B0 checkpoint for offline extraction.")
    parser.add_argument("--expected-source", type=Path, default=None,
                        help="Optional JSON hash manifest to verify source artifacts before evaluation.")
    parser.add_argument("--baselines", type=Path, default=Path(__file__).resolve().parents[1] / "baselines")
    parser.add_argument("--extraction", type=Path, default=Path(__file__).resolve().parents[1] / "extraction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prepare-e2-only", action="store_true",
                        help="Only extract and save efficientnet_target_letters.npy.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
