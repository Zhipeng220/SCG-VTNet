"""Run the validation-only ASL image-to-feature alignment check for Experiment A."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torchvision
import cv2
import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/asl_alphabet"
RAW = ROOT / "data/asl_alphabet/images"
sys.path[:0] = [str(REVISION / "baselines"), str(REVISION / "extraction"),
                str(REVISION / "audit")]

from data_spatial_tokens import build_resnet50_extractor, canonical_rgb_image, extract_21x3_landmarks
from main import load_experiment_config
from train import forward_model, parse_model_outputs
from utils import add_advanced_features, get_feature_columns
from verify_main import make_model


def predict(model, config, scaler, encoder, xyz, detected, visual):
    hand = np.array([detected], dtype=np.float32)
    xyz = xyz.copy().reshape(1, 21, 3)
    xyz[hand == 0] = 0
    points = xyz
    if config.get("use_advanced_features", True):
        points = add_advanced_features(xyz, hand, config.get("aar_descriptor", "all"))
    scaled = scaler.transform(points.reshape(1, -1)).reshape(points.shape)
    scaled[hand == 0] = 0
    logits, *_ = parse_model_outputs(forward_model(
        model, torch.as_tensor(scaled, dtype=torch.float32),
        torch.as_tensor(visual.astype(np.float32))[None],
        torch.as_tensor(xyz, dtype=torch.float32), torch.from_numpy(hand),
    ))
    return encoder.classes_[logits.argmax(dim=1).item()]


@torch.inference_mode()
def run(output, per_class, paired=False):
    torch.set_num_threads(4)
    processed = DATA / "processed_train"
    manifest = pd.read_csv(ROOT / "manifests/asl/split_manifest_seed42.csv")
    if paired:
        selected = (manifest[manifest.split == "test"].sample(frac=1, random_state=20261120)
                    .groupby("label", sort=True).head(per_class).sort_values("original_index"))
    else:
        selected = (manifest[manifest.split == "val"].sort_values("original_index")
                    .groupby("label", sort=True).head(per_class))
    landmarks = pd.read_csv(processed / "zerofill_landmarks_mask.csv")
    columns = get_feature_columns(landmarks)
    history = ROOT / "results/historical/fair_e1_e7/seed_42"
    artifacts = {}
    for method in ("E7", "E2", "E5"):
        config = load_experiment_config(next(history.glob(f"config_{method}_*.json")))
        name = config["safe_name"]
        scaler = joblib.load(history / f"scaler_{name}.pkl")
        encoder = joblib.load(history / f"label_encoder_{name}.pkl")
        model = make_model(config, len(encoder.classes_)).eval()
        model.load_state_dict(torch.load(
            history / "weights" / f"best_model_{name}_21kpts.pth",
            map_location="cpu", weights_only=True,
        ))
        feature_path = processed / PureWindowsPath(config["image_features_path"]).name
        artifacts[method] = model, config, scaler, encoder, np.load(feature_path, mmap_mode="r")

    resnet = build_resnet50_extractor("spatial_tokens", torch.device("cpu"))
    resnet_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    efficient_weights = models.EfficientNet_B0_Weights.DEFAULT
    efficient = models.efficientnet_b0(weights=efficient_weights).eval()
    efficient.classifier = torch.nn.Identity()
    efficient_transform = efficient_weights.transforms()
    if paired:
        return run_paired(output, selected, artifacts, resnet, resnet_transform,
                          efficient, efficient_transform)
    rows = []
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=.3,
    ) as detector:
        for sample in selected.itertuples(index=False):
            path = RAW / sample.image_path
            canonical = canonical_rgb_image(path, 224, "letterbox")
            original_xyz = landmarks.loc[sample.original_index, columns].to_numpy(dtype=np.float64)
            result = detector.process(canonical)
            detected = int(bool(result.multi_hand_landmarks))
            xyz = np.array(extract_21x3_landmarks(result) if detected else np.zeros(63))
            tokens = resnet(resnet_transform(Image.fromarray(canonical))[None])["feat"]
            tokens = tokens.flatten(2).transpose(1, 2)[0].numpy().astype(np.float16)
            with Image.open(path) as image:
                global_e2 = efficient(efficient_transform(image.convert("RGB"))[None])[0].numpy()
            features = {"E7": tokens, "E2": global_e2, "E5": tokens.astype(np.float32).mean(axis=0)}
            for method, feature in features.items():
                model, config, scaler, encoder, history_features = artifacts[method]
                baseline = history_features[sample.original_index]
                if feature.shape != baseline.shape:
                    raise ValueError(f"Feature shape mismatch for {method}: {feature.shape} != {baseline.shape}")
                scale = np.linalg.norm(baseline.astype(np.float64))
                relative_l2 = np.linalg.norm(feature.astype(np.float64) - baseline) / max(scale, 1e-12)
                rows.append({
                    "original_index": sample.original_index, "image_path": sample.image_path,
                    "true_label": sample.label, "method": method,
                    "historical_detected": sample.hand_detected, "reextracted_detected": detected,
                    "landmark_rmse": float(np.sqrt(np.mean((xyz - original_xyz) ** 2))),
                    "visual_relative_l2": float(relative_l2),
                    "historical_prediction": predict(model, config, scaler, encoder,
                                                     original_xyz, sample.hand_detected, baseline),
                    "reextracted_prediction": predict(model, config, scaler, encoder,
                                                      xyz, detected, feature),
                })

    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": "validation", "split_seed": 42, "checkpoint_seed": 42,
        "sampling": f"first {per_class} validation image(s) by original index per label",
        "versions": {"torch": torch.__version__, "mediapipe": mp.__version__,
                     "resnet_weights": str(models.ResNet50_Weights.DEFAULT),
                     "efficientnet_weights": str(efficient_weights)},
        "detection_threshold": 0.3, "methods": {},
    }
    for method in artifacts:
        subset = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "samples": len(subset),
            "detection_switches": sum(row["historical_detected"] != row["reextracted_detected"] for row in subset),
            "prediction_changes": sum(row["historical_prediction"] != row["reextracted_prediction"] for row in subset),
            "historical_correct": sum(row["historical_prediction"] == row["true_label"] for row in subset),
            "reextracted_correct": sum(row["reextracted_prediction"] == row["true_label"] for row in subset),
            "median_visual_relative_l2": float(np.median([row["visual_relative_l2"] for row in subset])),
            "max_visual_relative_l2": max(row["visual_relative_l2"] for row in subset),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


DEGRADATIONS = {
    "clean": ("clean", 0),
    "dark_light": ("dark", 0.75),
    "dark_heavy": ("dark", 0.45),
    "blur_light": ("blur", 1.5),
    "blur_heavy": ("blur", 4.0),
    "occlude_light": ("occlude", 0.08),
    "occlude_heavy": ("occlude", 0.20),
}
PERTURBATION_SEED = 20261120


def degrade(image, kind, value, index):
    params = {"kind": kind, "value": value}
    if kind == "dark":
        image = ImageEnhance.Brightness(image).enhance(value)
    elif kind == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=value))
    elif kind == "occlude":
        rng = np.random.default_rng([PERTURBATION_SEED, index])
        width = max(1, round(image.width * np.sqrt(value)))
        height = max(1, round(image.height * np.sqrt(value)))
        x = int(rng.integers(0, image.width - width + 1))
        y = int(rng.integers(0, image.height - height + 1))
        image = image.copy()
        image.paste((0, 0, 0), (x, y, x + width, y + height))
        params.update(rect=[x, y, width, height], fill=[0, 0, 0])
    return image, params


def letterbox(image):
    scale = 224 / max(image.size)
    width, height = (round(side * scale) for side in image.size)
    padded = Image.new("RGB", (224, 224))
    padded.paste(image.resize((width, height), Image.Resampling.BILINEAR),
                 ((224 - width) // 2, (224 - height) // 2))
    return padded


def run_paired(output, selected, artifacts, resnet, resnet_transform,
               efficient, efficient_transform):
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({
        "split": "seed42 test", "selection": "fixed uniform random sample per label",
        "counts": selected["label"].value_counts().to_dict(),
        "seed": PERTURBATION_SEED, "checkpoint_seed": 42, "image_conditions": DEGRADATIONS,
        "skeleton_conditions": {"xyz_jitter": {"sigma": 0.015, "scope": "clean detected",
            "coordinate_system": "MediaPipe normalized XYZ", "xy_clip": [0, 1]},
            "zero_skeleton": {"scope": "clean detected", "mask": 0}},
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "mediapipe": mp.__version__, "opencv": cv2.__version__,
        "resnet_weights": str(models.ResNet50_Weights.DEFAULT),
        "efficientnet_weights": str(models.EfficientNet_B0_Weights.DEFAULT),
        "frozen_manifest_sha256": hashlib.sha256(
            (ROOT / "manifests/asl/split_manifest_seed42.csv").read_bytes()
        ).hexdigest(),
    }, indent=2) + "\n", encoding="utf8")
    selected.to_csv(output / "samples.csv", index=False)
    fields = ["original_index", "image_path", "true_label", "condition", "params", "image_sha256",
              "historical_detected", "clean_detected", "condition_detected", "input_hand_mask",
              "method", "prediction"]
    with (output / "predictions.csv").open("w", encoding="utf8", newline="") as stream, mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=.3,
    ) as detector:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, sample in enumerate(selected.itertuples(index=False), start=1):
            path = RAW / sample.image_path
            with Image.open(path) as file:
                original = file.convert("RGB")
            clean_detected = None
            for condition, (kind, value) in DEGRADATIONS.items():
                image, params = degrade(original, kind, value, sample.original_index)
                canonical = letterbox(image)
                result = detector.process(np.asarray(canonical))
                detected = int(bool(result.multi_hand_landmarks))
                xyz = np.array(extract_21x3_landmarks(result) if detected else np.zeros(63))
                if condition == "clean":
                    clean_detected = detected
                tokens = resnet(resnet_transform(canonical)[None])["feat"]
                tokens = tokens.flatten(2).transpose(1, 2)[0].numpy().astype(np.float16)
                features = {
                    "E7": tokens, "E2": efficient(efficient_transform(image)[None])[0].numpy(),
                    "E5": tokens.astype(np.float32).mean(axis=0),
                }
                trials = [(condition, params, detected, xyz)]
                if condition == "clean" and detected:
                    rng = np.random.default_rng([PERTURBATION_SEED, sample.original_index, 1])
                    jittered = xyz.reshape(21, 3) + rng.normal(0, .015, size=(21, 3))
                    jittered[:, :2] = jittered[:, :2].clip(0, 1)
                    trials.extend([
                        ("xyz_jitter", {"sigma": .015}, 1, jittered),
                        ("zero_skeleton", {"mask": 0}, 0, np.zeros(63)),
                    ])
                for name, parameters, mask, coordinates in trials:
                    for method, feature in features.items():
                        model, config, scaler, encoder, _ = artifacts[method]
                        writer.writerow({
                            "original_index": sample.original_index, "image_path": sample.image_path,
                            "true_label": sample.label, "condition": name, "params": json.dumps(parameters),
                            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "historical_detected": sample.hand_detected, "clean_detected": clean_detected,
                            "condition_detected": detected, "input_hand_mask": mask, "method": method,
                            "prediction": predict(model, config, scaler, encoder, coordinates, mask, feature),
                        })
            stream.flush()
            if i % 29 == 0:
                print(f"Finished {i}/{len(selected)} test images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--paired-test", action="store_true")
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    run(args.output, args.per_class, args.paired_test)

