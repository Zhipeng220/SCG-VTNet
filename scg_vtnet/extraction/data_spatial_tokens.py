"""MediaPipe + ResNet50 preprocessing for ASL Alphabet.

This is a revised copy of ``mediapipe_resnet50.py``. It supports both no-hand
policies and can save ResNet50 layer4 spatial visual tokens instead of only a
global pooled 2048-D image vector.
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VALID_NO_HAND_POLICIES = ("skip", "zero_fill")
VALID_VISUAL_FEATURE_MODES = ("spatial_tokens", "global")
VALID_SAVE_DTYPES = ("float16", "float32")
VALID_RESIZE_MODES = ("letterbox", "stretch")


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet50_extractor(visual_feature_mode, device):
    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)

    if visual_feature_mode == "spatial_tokens":
        extractor = create_feature_extractor(resnet, return_nodes={"layer4": "feat"})
    elif visual_feature_mode == "global":
        resnet.fc = torch.nn.Identity()
        extractor = resnet
    else:
        raise ValueError(f"Unsupported visual_feature_mode: {visual_feature_mode}")

    extractor.to(device)
    extractor.eval()
    return extractor


def canonical_rgb_image(original_full_path, image_size, resize_mode="letterbox"):
    image_bgr = cv2.imread(str(original_full_path))
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    if resize_mode == "stretch":
        pil_image = pil_image.resize((image_size, image_size), Image.BILINEAR)
    elif resize_mode == "letterbox":
        width, height = pil_image.size
        scale = min(image_size / width, image_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = pil_image.resize((resized_width, resized_height), Image.BILINEAR)
        pil_image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        paste_x = (image_size - resized_width) // 2
        paste_y = (image_size - resized_height) // 2
        pil_image.paste(resized, (paste_x, paste_y))
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    return np.asarray(pil_image)


def extract_21x3_landmarks(results):
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks


def extract_visual_features(feature_extractor, image_tensor, visual_feature_mode):
    with torch.no_grad():
        output = feature_extractor(image_tensor)

    if visual_feature_mode == "spatial_tokens":
        fmap = output["feat"]  # [1, 2048, 7, 7] for 224x224 input
        return fmap.flatten(2).transpose(1, 2).squeeze(0).cpu().numpy()

    return output.squeeze(0).cpu().numpy()


def infer_feature_metadata(feature_extractor, preprocess, device, visual_feature_mode, image_size):
    dummy_rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_rgb)
    dummy_tensor = preprocess(dummy_image).unsqueeze(0).to(device)
    if visual_feature_mode == "spatial_tokens":
        with torch.no_grad():
            fmap = feature_extractor(dummy_tensor)["feat"]
        _, channels, height, width = fmap.shape
        return (height * width, channels), [height, width]
    dummy_features = extract_visual_features(feature_extractor, dummy_tensor, visual_feature_mode)
    return dummy_features.shape, None


def count_candidate_images(root_dir):
    total = 0
    root_path = Path(root_dir)

    # 1. 统计子文件夹中的图片 (Train 模式)
    for item in root_path.iterdir():
        if item.is_dir():
            total += len([p for p in item.iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS)])

    # 2. 统计直接放在根目录的图片 (Test 模式)
    total += len([p for p in root_path.iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS)])

    return total


def write_failed_images_csv(path, failed_rows):
    fieldnames = ["label", "image_path", "reason", "retained"]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)


def write_class_coverage_csv(path, class_stats):
    rows = []
    for label in sorted(class_stats):
        stats = class_stats[label]
        total = stats["total"]
        kept = stats["kept"]
        no_hand = stats["no_hand"]
        read_fail = stats["read_fail"]
        error = stats["error"]
        retained_no_hand = stats["retained_no_hand"]
        rows.append(
            {
                "label": label,
                "total_images": total,
                "kept_samples": kept,
                "no_hand": no_hand,
                "retained_no_hand": retained_no_hand,
                "read_fail": read_fail,
                "error": error,
                "coverage": (kept / total) if total else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_preprocessing_report(path, report):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)


def create_memmap(path, sample_count, feature_shape, save_dtype):
    dtype = np.float16 if save_dtype == "float16" else np.float32
    return np.lib.format.open_memmap(
        path,
        dtype=dtype,
        mode="w+",
        shape=(sample_count, *feature_shape),
    )


def append_or_write_feature(feature_store, feature_index, feature, save_dtype):
    if save_dtype == "float16":
        feature_store[feature_index] = feature.astype(np.float16)
    else:
        feature_store[feature_index] = feature.astype(np.float32)


def process_full_dataset(
    root_dir,
    output_dir,
    no_hand_policy="skip",
    visual_feature_mode="spatial_tokens",
    image_size=224,
    save_dtype="float16",
    copy_images=True,
    min_detection_confidence=0.3,
    resize_mode="letterbox",
    compact_chunk_size=512,
):
    """Process the dataset and save landmarks, visual features, and reports.

    ``spatial_tokens`` saves ResNet50 layer4 features as ``N x 49 x 2048`` for a
    224x224 canonical image. MediaPipe and ResNet50 both consume the same
    resized 224x224 RGB image, so landmark x/y coordinates are in the same
    canonical coordinate system as the 7x7 visual token grid.
    """

    if no_hand_policy not in VALID_NO_HAND_POLICIES:
        raise ValueError(f"no_hand_policy must be one of {VALID_NO_HAND_POLICIES}")
    if visual_feature_mode not in VALID_VISUAL_FEATURE_MODES:
        raise ValueError(f"visual_feature_mode must be one of {VALID_VISUAL_FEATURE_MODES}")
    if save_dtype not in VALID_SAVE_DTYPES:
        raise ValueError(f"save_dtype must be one of {VALID_SAVE_DTYPES}")
    if resize_mode not in VALID_RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {VALID_RESIZE_MODES}")

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root_dir.is_dir():
        print(f"Error: Directory not found -> {root_dir}")
        return

    prefix = "zerofill" if no_hand_policy == "zero_fill" else "skip"
    feature_suffix = "visual_tokens_layer4" if visual_feature_mode == "spatial_tokens" else "global_features"
    landmarks_csv_path = output_dir / (
        f"{prefix}_landmarks_mask.csv" if no_hand_policy == "zero_fill" else f"{prefix}_landmarks.csv"
    )
    feature_path = output_dir / f"{prefix}_{feature_suffix}.npy"
    failed_images_csv_path = output_dir / f"{prefix}_failed_images.csv"
    class_coverage_csv_path = output_dir / f"{prefix}_class_coverage.csv"
    report_json_path = output_dir / f"{prefix}_preprocessing_report.json"
    image_copy_dir = output_dir / f"{prefix}_images_processed"

    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    print("Initializing ResNet50 extractor...")
    device = build_device()
    print(f"Using device: {device}")
    feature_extractor = build_resnet50_extractor(visual_feature_mode, device)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    feature_shape, visual_grid_shape = infer_feature_metadata(
        feature_extractor,
        preprocess,
        device,
        visual_feature_mode,
        image_size,
    )

    total_candidates = count_candidate_images(root_dir)
    sample_capacity = total_candidates
    if no_hand_policy == "skip":
        print("Using memmap capacity equal to candidate image count; final kept count is in the report.")

    visual_features = create_memmap(feature_path, sample_capacity, feature_shape, save_dtype)
    landmark_rows = []
    failed_rows = []
    class_stats = defaultdict(
        lambda: {
            "total": 0,
            "kept": 0,
            "no_hand": 0,
            "read_fail": 0,
            "error": 0,
            "retained_no_hand": 0,
        }
    )
    kept_count = 0

    label_dirs = sorted(
        [path for path in root_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )


    image_tasks = []

    # 1. 扫描子文件夹 (Train 结构)
    for label_path in sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        label = label_path.name
        for img_path in sorted(
                [p for p in label_path.iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS)],
                key=lambda p: p.name):
            image_tasks.append((label, img_path, str(Path(label) / img_path.name)))

    # 2. 扫描根目录下的平铺图片 (Test 结构)
    for img_path in sorted(
            [p for p in root_dir.iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS)],
            key=lambda p: p.name):
        # 从文件名提取标签 (例如 'A_test.jpg' -> 提取出 'A')
        label = img_path.stem.split('_')[0]
        image_tasks.append((label, img_path, img_path.name))

    print(f"Starting data scan from directory: {root_dir}")
    with tqdm(total=total_candidates, desc="Processing Images") as pbar:
        for label, image_path, relative_img_path in image_tasks:
            pbar.update(1)
            class_stats[label]["total"] += 1

            try:
                image_rgb = canonical_rgb_image(image_path, image_size, resize_mode)
                if image_rgb is None:
                    class_stats[label]["read_fail"] += 1
                    failed_rows.append(
                        {
                            "label": label,
                            "image_path": relative_img_path,
                            "reason": "cv2_read_fail",
                            "retained": False,
                        }
                    )
                    tqdm.write(f"SKIPPING (CV2 read fail): {image_path}")
                    continue

                results = hands.process(image_rgb)
                hand_detected = int(bool(results.multi_hand_landmarks))

                if hand_detected:
                    landmarks = extract_21x3_landmarks(results)
                else:
                    class_stats[label]["no_hand"] += 1
                    failed_rows.append(
                        {
                            "label": label,
                            "image_path": relative_img_path,
                            "reason": "no_hand",
                            "retained": no_hand_policy == "zero_fill",
                        }
                    )
                    if no_hand_policy == "skip":
                        tqdm.write(f"SKIPPING (No hand): {image_path}")
                        continue
                    class_stats[label]["retained_no_hand"] += 1
                    landmarks = [0.0] * 63

                pil_image = Image.fromarray(image_rgb)
                image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                visual_feature = extract_visual_features(
                    feature_extractor,
                    image_tensor,
                    visual_feature_mode,
                )
                append_or_write_feature(visual_features, kept_count, visual_feature, save_dtype)

                row_data = [label, relative_img_path, hand_detected]
                row_data.extend(landmarks)
                landmark_rows.append(row_data)

                if copy_images:
                    dest_label_dir = image_copy_dir / label
                    dest_label_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, dest_label_dir / image_path.name)

                class_stats[label]["kept"] += 1
                kept_count += 1

            except Exception as exc:
                class_stats[label]["error"] += 1
                failed_rows.append(
                    {
                        "label": label,
                        "image_path": relative_img_path,
                        "reason": f"error: {exc}",
                        "retained": False,
                    }
                )
                tqdm.write(f"SKIPPING (Error): {image_path} -> {exc}")

    hands.close()
    visual_features.flush()

    if kept_count == 0:
        print("No samples were saved. Halting after writing failure reports.")
        del visual_features
        feature_path.unlink(missing_ok=True)
        write_failed_images_csv(failed_images_csv_path, failed_rows)
        write_class_coverage_csv(class_coverage_csv_path, class_stats)
        write_preprocessing_report(
            report_json_path,
            {
                "root_dir": str(root_dir),
                "output_dir": str(output_dir),
                "no_hand_policy": no_hand_policy,
                "visual_feature_mode": visual_feature_mode,
                "image_size": image_size,
                "resize_mode": resize_mode,
                "save_dtype": save_dtype,
                "total_candidates": total_candidates,
                "kept_samples": 0,
                "dropped_samples": total_candidates,
                "retained_no_hand_samples": 0,
                "read_fail_samples": sum(stats["read_fail"] for stats in class_stats.values()),
                "error_samples": sum(stats["error"] for stats in class_stats.values()),
                "feature_path": str(feature_path),
                "feature_shape_saved": [0, *feature_shape],
            },
        )
        return

    compact_feature_path = output_dir / f"{prefix}_{feature_suffix}_compact.npy"
    if kept_count < sample_capacity:
        compact_features = create_memmap(compact_feature_path, kept_count, feature_shape, save_dtype)
        for start in range(0, kept_count, compact_chunk_size):
            end = min(start + compact_chunk_size, kept_count)
            compact_features[start:end] = visual_features[start:end]
        compact_features.flush()
        del compact_features
        del visual_features
        feature_path.unlink(missing_ok=True)
        feature_path = compact_feature_path
    else:
        del visual_features
        compact_feature_path.unlink(missing_ok=True)
        feature_path.replace(compact_feature_path)
        feature_path = compact_feature_path

    headers = ["label", "image_path", "hand_detected"]
    for idx in range(21):
        headers.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])
    pd.DataFrame(landmark_rows, columns=headers).to_csv(landmarks_csv_path, index=False)

    write_failed_images_csv(failed_images_csv_path, failed_rows)
    write_class_coverage_csv(class_coverage_csv_path, class_stats)

    retained_no_hand_samples = sum(stats["retained_no_hand"] for stats in class_stats.values())
    read_fail_samples = sum(stats["read_fail"] for stats in class_stats.values())
    error_samples = sum(stats["error"] for stats in class_stats.values())
    dropped_no_hand_samples = sum(stats["no_hand"] for stats in class_stats.values()) - retained_no_hand_samples
    dropped_samples = dropped_no_hand_samples + read_fail_samples + error_samples
    if visual_grid_shape:
        coordinate_alignment = (
            "MediaPipe and ResNet50 both receive the same canonical RGB image. "
            f"For spatial_tokens, ResNet50 layer4 is flattened from "
            f"{visual_grid_shape[0]}x{visual_grid_shape[1]} to {feature_shape[0]} tokens."
        )
    else:
        coordinate_alignment = "MediaPipe and ResNet50 both receive the same canonical RGB image."

    report = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "no_hand_policy": no_hand_policy,
        "visual_feature_mode": visual_feature_mode,
        "canonical_image_size": [image_size, image_size],
        "resize_mode": resize_mode,
        "coordinate_alignment": coordinate_alignment,
        "visual_grid_shape": visual_grid_shape,
        "visual_token_order": "row-major flatten of the layer4 HxW feature map" if visual_grid_shape else None,
        "save_dtype": save_dtype,
        "total_candidates": total_candidates,
        "kept_samples": kept_count,
        "dropped_samples": dropped_samples,
        "retained_no_hand_samples": retained_no_hand_samples,
        "read_fail_samples": read_fail_samples,
        "error_samples": error_samples,
        "feature_path": str(feature_path),
        "feature_shape_saved": [kept_count, *feature_shape],
        "landmarks_csv_path": str(landmarks_csv_path),
        "failed_images_csv_path": str(failed_images_csv_path),
        "class_coverage_csv_path": str(class_coverage_csv_path),
        "copied_images_dir": str(image_copy_dir) if copy_images else None,
        "copied_image_type": "original_source_image" if copy_images else None,
        "min_detection_confidence": min_detection_confidence,
    }
    write_preprocessing_report(report_json_path, report)

    print("\n" + "=" * 50)
    print("Process Finished Successfully!")
    print(f"Landmarks CSV saved to: {landmarks_csv_path} ({len(landmark_rows)} rows)")
    print(f"Visual features saved to: {feature_path} ({tuple([kept_count, *feature_shape])})")
    print(f"Failed image log saved to: {failed_images_csv_path}")
    print(f"Class coverage saved to: {class_coverage_csv_path}")
    print(f"Preprocessing report saved to: {report_json_path}")
    if copy_images:
        print(f"Processed images copied to: {image_copy_dir}")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks and ResNet50 visual features."
    )
    parser.add_argument("--root-dir", required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed outputs.")
    parser.add_argument(
        "--no-hand-policy",
        choices=VALID_NO_HAND_POLICIES,
        default="skip",
        help="Skip no-hand images or keep them with zero-filled landmarks.",
    )
    parser.add_argument(
        "--visual-feature-mode",
        choices=VALID_VISUAL_FEATURE_MODES,
        default="spatial_tokens",
        help="Save layer4 spatial tokens or global 2048-D features.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dtype", choices=VALID_SAVE_DTYPES, default="float16")
    parser.add_argument(
        "--resize-mode",
        choices=VALID_RESIZE_MODES,
        default="letterbox",
        help="Use aspect-ratio preserving padding or direct square resize.",
    )
    parser.add_argument("--compact-chunk-size", type=int, default=512)
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_full_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        no_hand_policy=args.no_hand_policy,
        visual_feature_mode=args.visual_feature_mode,
        image_size=args.image_size,
        save_dtype=args.save_dtype,
        copy_images=not args.no_copy_images,
        min_detection_confidence=args.min_detection_confidence,
        resize_mode=args.resize_mode,
        compact_chunk_size=args.compact_chunk_size,
    )




