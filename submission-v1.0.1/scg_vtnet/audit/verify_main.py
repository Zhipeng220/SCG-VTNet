"""Re-evaluate historical ASL Alphabet checkpoints on the frozen test splits."""

import argparse
import csv
import json
import logging
from pathlib import Path, PureWindowsPath
import random
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader


def make_model(config, num_classes):
    model_class = config["model_class"]
    common = dict(num_classes=num_classes, dropout=config["dropout"])
    attention = dict(embed_dim=config["embed_dim"], num_layers=config["num_layers"],
                     num_heads=config["num_heads"])
    image_dim = config["image_feature_dim"]
    name = model_class.__name__
    if name == "ImageOnlyBaseline":
        return model_class(**common, image_feature_dim=image_dim)
    if name == "VisualTokenTransformerBaseline":
        return model_class(**common, **attention, image_feature_dim=image_dim,
                           visual_token_count=config["visual_token_count"])
    if name == "AdvancedAttentionModel":
        return model_class(**common, **attention, input_dim=config["input_dim"])
    if name != "HybridAttentionModel":
        raise ValueError(f"Unexpected main-table model {name}")
    return model_class(
        **common, **attention, input_dim=config["input_dim"], image_feature_dim=image_dim,
        visual_token_count=config["visual_token_count"],
        use_gated_fusion=config.get("use_gated_fusion", False),
        use_dynamic_graph=config.get("use_dynamic_graph", False),
        use_cross_attn=config.get("use_cross_attn", True),
        use_gcn_branch=config.get("use_gcn_branch", False),
        use_keypoint_refinement=config.get("use_keypoint_refinement", False),
        refine_scale=config.get("refine_scale", 0.05),
    )


def evaluate(args):
    revision = Path(__file__).resolve().parents[1]
    data_dir = args.root / "data/asl_alphabet/processed_train"
    history = args.root / "results/historical/fair_e1_e7"
    sys.path.insert(0, str(revision / "baselines"))
    from main import load_experiment_config
    from train import forward_model, parse_model_outputs, test_robustness, unpack_batch
    from utils import IndexedMultimodalDataset, add_advanced_features, get_feature_columns

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    landmarks = pd.read_csv(data_dir / "zerofill_landmarks_mask.csv")
    columns = get_feature_columns(landmarks)
    historical_rows = pd.read_csv(revision / "audit/table_70_file_mapping.csv")
    historical_rows = historical_rows[historical_rows["table"] == "table1"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = None
        for method in args.methods:
            for seed in args.seeds:
                start = perf_counter()
                saved = history / f"seed_{seed}"
                config_path, = saved.glob(f"config_{method}_*.json")
                config = load_experiment_config(config_path)
                safe_name = config["safe_name"]
                weight_path = saved / "weights" / f"best_model_{safe_name}_21kpts.pth"
                scaler = joblib.load(saved / f"scaler_{safe_name}.pkl")
                encoder = joblib.load(saved / f"label_encoder_{safe_name}.pkl")
                frozen = pd.read_csv(revision / "audit" / f"split_manifest_seed{seed}.csv")
                current = pd.read_csv(data_dir / f"split_manifest_seed{seed}.csv")
                if not frozen.equals(current):
                    raise ValueError(f"Seed {seed}: installed split differs from the audited split")
                if (len(frozen) != 87000 or set(frozen["original_index"]) != set(range(len(landmarks)))
                        or frozen["split"].value_counts().to_dict() != {"train": 52200, "val": 17400, "test": 17400}):
                    raise ValueError(f"Seed {seed}: incomplete split")
                selected = frozen.loc[frozen["split"] == "test"]
                indices = selected["original_index"].to_numpy()
                samples = landmarks.iloc[indices]
                keys = ["label", "image_path", "hand_detected"]
                if not selected[keys].reset_index(drop=True).equals(samples[keys].reset_index(drop=True)):
                    raise ValueError(f"Seed {seed}: landmarks disagree with split rows")
                hand = samples["hand_detected"].to_numpy(dtype=np.float32)
                xyz = samples[columns].to_numpy().reshape(len(samples), 21, 3)
                raw = xyz.astype(np.float32)
                raw[hand == 0] = 0
                if config.get("use_advanced_features", True):
                    points = add_advanced_features(xyz, hand, config.get("aar_descriptor", "all"))
                else:
                    points = xyz
                    points[hand == 0] = 0
                scaled = scaler.transform(points.reshape(len(samples), -1)).reshape(points.shape)
                scaled[hand == 0] = 0
                if scaled.shape[-1] != config["input_dim"]:
                    raise ValueError(f"{safe_name}: landmark feature dimensions disagree")
                feature_path = data_dir / PureWindowsPath(config["image_features_path"]).name
                features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                if features.shape != (len(landmarks), *config["image_feature_shape"]):
                    raise ValueError(f"{safe_name}: visual feature array does not match saved configuration")
                labels = encoder.transform(samples["label"].to_numpy())
                if args.cache_test_features:
                    order = np.argsort(indices)
                    cached = np.empty((len(indices), *features.shape[1:]), dtype=features.dtype)
                    for chunk in np.array_split(order, (len(order) + 255) // 256):
                        cached[chunk] = features[indices[chunk]]
                    del features
                    dataset = IndexedMultimodalDataset(scaled, raw, cached, np.arange(len(indices)), hand, labels)
                    del cached
                else:
                    dataset = IndexedMultimodalDataset(scaled, raw, features, indices, hand, labels)
                model = make_model(config, len(encoder.classes_))
                model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
                model.to(device).eval()
                predictions = []
                with torch.no_grad():
                    for batch in DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False):
                        batch_points, batch_raw, visual, batch_hand, _ = unpack_batch(batch, device)
                        logits, *_ = parse_model_outputs(forward_model(model, batch_points, visual, batch_raw, batch_hand))
                        predictions.extend(logits.argmax(dim=1).cpu().tolist())
                if args.prediction_dir is not None:
                    args.prediction_dir.mkdir(parents=True, exist_ok=True)
                    prediction_path = args.prediction_dir / f"{method}_seed_{seed}.csv"
                    predicted_labels = encoder.inverse_transform(predictions)
                    with prediction_path.open("x", encoding="utf-8", newline="") as prediction_file:
                        prediction_writer = csv.writer(prediction_file)
                        prediction_writer.writerow([
                            "seed", "method", "original_index", "image_path",
                            "true_label", "predicted_label", "hand_detected",
                        ])
                        prediction_writer.writerows(
                            (seed, method, index, path, label, predicted, int(detected))
                            for index, path, label, predicted, detected in zip(
                                indices, samples["image_path"], samples["label"],
                                predicted_labels, hand, strict=True,
                            )
                        )
                historical, = historical_rows.loc[
                    (historical_rows["seed"] == seed) & (historical_rows["experiment"] == config["experiment_name"]),
                    "summary_line",
                ]
                _, old_accuracy, old_f1, *old_perturbations = [item.strip() for item in historical.split("|")]
                accuracy = accuracy_score(labels, predictions) * 100
                f1 = f1_score(labels, predictions, average="macro", zero_division=0) * 100
                row = {
                    "method": config["experiment_name"], "seed": seed, "device": str(device),
                    "test_samples": len(labels), "correct": int(np.sum(labels == predictions)),
                    "accuracy_pct": f"{accuracy:.8f}", "macro_f1_pct": f"{f1:.8f}",
                    "historical_accuracy_pct": old_accuracy.rstrip("%"),
                    "historical_macro_f1_pct": old_f1.rstrip("%"),
                }
                print(json.dumps(row), flush=True)
                if args.robust:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    robust = test_robustness(
                        config["model_class"], weight_path, dataset, device, encoder,
                        argparse.Namespace(batch_size=args.batch_size, embed_dim=config["embed_dim"],
                                           num_layers=config["num_layers"], num_heads=config["num_heads"],
                                           dropout=config["dropout"]),
                        config["input_dim"], config, image_feature_dim=config["image_feature_dim"],
                    )
                    if robust is None:
                        raise RuntimeError(f"Robustness evaluation failed for {safe_name} seed {seed}")
                    row.update({key + "_pct": f"{value * 100:.8f}" for key, value in robust.items()})
                    row["historical_joint_pct"] = old_perturbations[6].rstrip("%")
                row["seconds"] = f"{perf_counter() - start:.2f}"
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                output.flush()
                del dataset, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=["E7", "E1", "E2", "E4", "E5", "E6", "E3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--cache-test-features", action="store_true")
    evaluate(parser.parse_args())
