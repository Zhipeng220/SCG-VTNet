"""Extract MediaPipe landmarks and ResNet50 spatial tokens for local datasets.

This script reuses the feature extraction primitives from
``data_spatial_tokens.py`` but makes the image enumeration
explicit so datasets with different layouts can be handled safely.
"""

import argparse
import csv
import importlib.util
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VALID_DATASET_FORMATS = ("class_dirs", "nus_hand_posture")


from scg_vtnet.datasets import visual_features as spatial


def natural_sort_key(path):
    text = path.name if isinstance(path, Path) else str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def iter_image_files(path):
    return sorted(
        [
            p
            for p in path.iterdir()
            if p.is_file()
            and not p.name.startswith("._")
            and p.name.lower().endswith(IMAGE_EXTENSIONS)
        ],
        key=natural_sort_key,
    )


def build_class_dir_tasks(root_dir):
    tasks = []
    for label_path in sorted([p for p in root_dir.iterdir() if p.is_dir()], key=natural_sort_key):
        label = label_path.name
        for image_path in iter_image_files(label_path):
            tasks.append(
                {
                    "label": label,
                    "image_path": image_path,
                    "relative_image_path": str(Path(label) / image_path.name),
                    "metadata": {},
                }
            )
    for image_path in iter_image_files(root_dir):
        label = image_path.stem.split("_")[0]
        tasks.append(
            {
                "label": label,
                "image_path": image_path,
                "relative_image_path": image_path.name,
                "metadata": {},
            }
        )
    return tasks


def parse_nus_label(image_path):
    match = re.match(r"^(g\d+)(?:\s*\(\d+\))?$", image_path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def build_nus_hand_posture_tasks(root_dir):
    tasks = []
    for subset_name in ("BW", "Color"):
        subset_dir = root_dir / subset_name
        if not subset_dir.is_dir():
            continue
        for image_path in iter_image_files(subset_dir):
            label = parse_nus_label(image_path)
            if label is None:
                continue
            tasks.append(
                {
                    "label": label,
                    "image_path": image_path,
                    "relative_image_path": str(Path(subset_name) / image_path.name),
                    "metadata": {"subset": subset_name},
                }
            )
    return sorted(tasks, key=lambda item: (item["metadata"]["subset"], natural_sort_key(Path(item["relative_image_path"]))))


def build_image_tasks(root_dir, dataset_format):
    if dataset_format == "class_dirs":
        return build_class_dir_tasks(root_dir)
    if dataset_format == "nus_hand_posture":
        return build_nus_hand_posture_tasks(root_dir)
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def write_failed_images_csv(path, failed_rows, metadata_fields):
    fieldnames = ["label", "image_path", *metadata_fields, "reason", "retained"]
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)


def process_dataset(
    root_dir,
    output_dir,
    dataset_format,
    no_hand_policy="zero_fill",
    visual_feature_mode="spatial_tokens",
    image_size=224,
    save_dtype="float16",
    copy_images=False,
    min_detection_confidence=0.3,
    resize_mode="letterbox",
    compact_chunk_size=512,
):
    if no_hand_policy not in spatial.VALID_NO_HAND_POLICIES:
        raise ValueError(f"no_hand_policy must be one of {spatial.VALID_NO_HAND_POLICIES}")
    if visual_feature_mode not in spatial.VALID_VISUAL_FEATURE_MODES:
        raise ValueError(f"visual_feature_mode must be one of {spatial.VALID_VISUAL_FEATURE_MODES}")
    if save_dtype not in spatial.VALID_SAVE_DTYPES:
        raise ValueError(f"save_dtype must be one of {spatial.VALID_SAVE_DTYPES}")
    if resize_mode not in spatial.VALID_RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {spatial.VALID_RESIZE_MODES}")

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    image_tasks = build_image_tasks(root_dir, dataset_format)
    if not image_tasks:
        raise RuntimeError(f"No images found for format '{dataset_format}' under {root_dir}")

    metadata_fields = sorted({key for task in image_tasks for key in task["metadata"]})
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
    mp_hands = spatial.mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    print("Initializing ResNet50 extractor...")
    device = spatial.build_device()
    print(f"Using device: {device}")
    feature_extractor = spatial.build_resnet50_extractor(visual_feature_mode, device)
    preprocess = spatial.transforms.Compose(
        [
            spatial.transforms.ToTensor(),
            spatial.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    feature_shape, visual_grid_shape = spatial.infer_feature_metadata(
        feature_extractor,
        preprocess,
        device,
        visual_feature_mode,
        image_size,
    )

    sample_capacity = len(image_tasks)
    visual_features = spatial.create_memmap(feature_path, sample_capacity, feature_shape, save_dtype)
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

    print(f"Starting data scan from directory: {root_dir}")
    print(f"Dataset format: {dataset_format}; images queued: {sample_capacity}")
    with tqdm(total=sample_capacity, desc="Processing Images") as pbar:
        for task in image_tasks:
            pbar.update(1)
            label = task["label"]
            image_path = task["image_path"]
            relative_img_path = task["relative_image_path"]
            metadata = task["metadata"]
            class_stats[label]["total"] += 1

            try:
                image_rgb = spatial.canonical_rgb_image(image_path, image_size, resize_mode)
                if image_rgb is None:
                    class_stats[label]["read_fail"] += 1
                    failed_rows.append(
                        {
                            "label": label,
                            "image_path": relative_img_path,
                            **metadata,
                            "reason": "cv2_read_fail",
                            "retained": False,
                        }
                    )
                    tqdm.write(f"SKIPPING (CV2 read fail): {image_path}")
                    continue

                results = hands.process(image_rgb)
                hand_detected = int(bool(results.multi_hand_landmarks))
                if hand_detected:
                    landmarks = spatial.extract_21x3_landmarks(results)
                else:
                    class_stats[label]["no_hand"] += 1
                    failed_rows.append(
                        {
                            "label": label,
                            "image_path": relative_img_path,
                            **metadata,
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
                visual_feature = spatial.extract_visual_features(
                    feature_extractor,
                    image_tensor,
                    visual_feature_mode,
                )
                spatial.append_or_write_feature(visual_features, kept_count, visual_feature, save_dtype)

                row_data = [label, relative_img_path]
                row_data.extend(metadata.get(field, "") for field in metadata_fields)
                row_data.append(hand_detected)
                row_data.extend(landmarks)
                landmark_rows.append(row_data)

                if copy_images:
                    metadata_parts = [str(metadata[field]) for field in metadata_fields if metadata.get(field)]
                    dest_label_dir = image_copy_dir.joinpath(*metadata_parts, label)
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
                        **metadata,
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
        write_failed_images_csv(failed_images_csv_path, failed_rows, metadata_fields)
        spatial.write_class_coverage_csv(class_coverage_csv_path, class_stats)
        spatial.write_preprocessing_report(
            report_json_path,
            {
                "root_dir": str(root_dir),
                "output_dir": str(output_dir),
                "dataset_format": dataset_format,
                "no_hand_policy": no_hand_policy,
                "visual_feature_mode": visual_feature_mode,
                "image_size": image_size,
                "resize_mode": resize_mode,
                "save_dtype": save_dtype,
                "total_candidates": sample_capacity,
                "kept_samples": 0,
                "dropped_samples": sample_capacity,
                "feature_path": str(feature_path),
                "feature_shape_saved": [0, *feature_shape],
            },
        )
        return

    compact_feature_path = output_dir / f"{prefix}_{feature_suffix}_compact.npy"
    if kept_count < sample_capacity:
        compact_features = spatial.create_memmap(compact_feature_path, kept_count, feature_shape, save_dtype)
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

    headers = ["label", "image_path", *metadata_fields, "hand_detected"]
    for idx in range(21):
        headers.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])
    pd.DataFrame(landmark_rows, columns=headers).to_csv(landmarks_csv_path, index=False)

    write_failed_images_csv(failed_images_csv_path, failed_rows, metadata_fields)
    spatial.write_class_coverage_csv(class_coverage_csv_path, class_stats)

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
        "dataset_format": dataset_format,
        "metadata_fields": metadata_fields,
        "no_hand_policy": no_hand_policy,
        "visual_feature_mode": visual_feature_mode,
        "canonical_image_size": [image_size, image_size],
        "resize_mode": resize_mode,
        "coordinate_alignment": coordinate_alignment,
        "visual_grid_shape": visual_grid_shape,
        "visual_token_order": "row-major flatten of the layer4 HxW feature map" if visual_grid_shape else None,
        "save_dtype": save_dtype,
        "total_candidates": sample_capacity,
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
    spatial.write_preprocessing_report(report_json_path, report)

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
        description="Extract MediaPipe landmarks and ResNet50 visual features for local sign datasets."
    )
    parser.add_argument("--root-dir", required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed outputs.")
    parser.add_argument("--dataset-format", choices=VALID_DATASET_FORMATS, required=True)
    parser.add_argument(
        "--no-hand-policy",
        choices=spatial.VALID_NO_HAND_POLICIES,
        default="zero_fill",
        help="Skip no-hand images or keep them with zero-filled landmarks.",
    )
    parser.add_argument(
        "--visual-feature-mode",
        choices=spatial.VALID_VISUAL_FEATURE_MODES,
        default="spatial_tokens",
        help="Save layer4 spatial tokens or global 2048-D features.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-dtype", choices=spatial.VALID_SAVE_DTYPES, default="float16")
    parser.add_argument(
        "--resize-mode",
        choices=spatial.VALID_RESIZE_MODES,
        default="letterbox",
        help="Use aspect-ratio preserving padding or direct square resize.",
    )
    parser.add_argument("--compact-chunk-size", type=int, default=512)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        dataset_format=args.dataset_format,
        no_hand_policy=args.no_hand_policy,
        visual_feature_mode=args.visual_feature_mode,
        image_size=args.image_size,
        save_dtype=args.save_dtype,
        copy_images=args.copy_images,
        min_detection_confidence=args.min_detection_confidence,
        resize_mode=args.resize_mode,
        compact_chunk_size=args.compact_chunk_size,
    )
