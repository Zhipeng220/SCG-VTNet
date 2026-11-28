"""Prepare and run the ASL Alphabet contiguous index-block split experiment."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np

SPLITS = ("train", "val", "test")
SEEDS = (42, 43, 44)
METHODS = ("F", "Uniform", "E2", "E5")
PARTITION_SEED = 20261122
BUFFER = 10
REFERENCE_MANIFEST_SHA256 = "d450325b7be86ab9ad0c8d1de64cd9c38ee3f69066a55a8928d232a20cdcb2fb"
FEATURES = {
    "zerofill_landmarks_mask.csv": "ac560af6a04e80e00b2eaf0094c4655d94ec11950b2dcfbc5d8be18872f5ce82",
    "zerofill_visual_tokens_layer4_compact.npy": "92cc9937cf30f85261ee416f07a6c9910968d85b7ed7c73783c0061a80e02123",
    "zerofill_visual_global_from_tokens.npy": "43408b3531f76adb323c9b58b3d2b3a67132e346d8b3eed64a57fc20aaf5c1ca",
    "efficientnet_b0_global.npy": "bcb5d94eb1be45f28c7e81548bb8cb1144243e4b223f51f1a4e813ee0d80637d",
}


def digest(file_path):
    hasher = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def split_manifest(previous, dest):
    with previous.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 87000
    records.sort(key=lambda row: int(row["original_index"]))
    assert [int(r["original_index"]) for r in records] == list(range(87000))
    per_label = defaultdict(list)
    for row in records:
        per_label[row["label"]].append(row)
    assert len(per_label) == 29
    rng = random.Random(PARTITION_SEED)
    grid = {}
    for label, rows in sorted(per_label.items()):
        numbered = []
        for row in rows:
            name = Path(row["image_path"]).stem
            match = re.fullmatch(r"(?:[A-Za-z]+)[_-]?(\d+)", name)
            if match is None:
                raise ValueError(f"Cannot identify image index: {row['image_path']}")
            numbered.append((int(match[1]), row))
        numbered.sort(key=lambda item: item[0])
        assert len(numbered) == 3000 and len({n for n, _ in numbered}) == len(numbered), label
        order = list(SPLITS)
        rng.shuffle(order)
        sizes = {"train": 1800, "val": 600, "test": 600}
        start = 0
        ranges = []
        for i, split in enumerate(order):
            end = start + sizes[split]
            first = start + (BUFFER if i > 0 else 0)
            last = end - (BUFFER if i < len(order) - 1 else 0)
            for pos in range(start, end):
                row = numbered[pos][1]
                row["baseline_split"] = row["split"]
                row["split"] = split if first <= pos < last else "buffer"
                row["split_seed"] = str(PARTITION_SEED)
                row["image_number"] = str(numbered[pos][0])
            ranges.append({
                "split": split,
                "min_kept": numbered[first][0],
                "max_kept": numbered[last - 1][0],
                "kept": last - first,
            })
            start = end
        assert start == len(numbered)
        for a, b in zip(ranges, ranges[1:]):
            assert b["min_kept"] - a["max_kept"] >= 2 * BUFFER + 1, (label, a, b)
        grid[label] = ranges
    with dest.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    counts = dict(Counter(row["split"] for row in records))
    assert counts["buffer"] == 29 * 2 * 2 * BUFFER
    return records, {
        "partition_seed": PARTITION_SEED,
        "buffer_per_side": BUFFER,
        "counts": counts,
        "manifest_sha256": digest(dest),
        "per_class": grid,
        "detected": dict(Counter(row["split"] for row in records if row["hand_detected"] == "1")),
        "training_seeds": list(SEEDS),
        "methods": list(METHODS),
        "note": (
            "One fixed grouped split for all models and training seeds. "
            "Numeric blocks approximate grouping; no subject/session identity is established."
        ),
    }


def audit_hashes(records, raw):
    from PIL import Image

    old = defaultdict(Counter)
    new = defaultdict(Counter)
    for i, row in enumerate(records, 1):
        with Image.open(raw / row["image_path"]) as image:
            pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        code = np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes()
        old[code][row["baseline_split"]] += 1
        if row["split"] != "buffer":
            new[code][row["split"]] += 1
        if i % 20000 == 0:
            print(f"dHash: {i}/{len(records)} images", flush=True)

    def stats(groups):
        cross = [count for count in groups.values() if len(count) > 1]
        train_test = [count for count in groups.values() if count["train"] and count["test"]]
        total = sum(sum(c.values()) for c in groups.values())
        rows = sum(sum(c.values()) for c in cross)
        return {
            "images": total,
            "cross_partition_groups": len(cross),
            "cross_partition_images": rows,
            "cross_partition_pct": 100 * rows / total,
            "train_test_groups": len(train_test),
            "test_images_with_train_hash": sum(c["test"] for c in train_test),
        }

    return {
        "definition": "Pillow L mode, Lanczos 9x8, 64 horizontal difference bits",
        "random_seed42": stats(old),
        "blocked": stats(new),
    }


def run(args):
    output = args.output.resolve()
    reference = args.reference_manifest.resolve()
    code = args.baselines.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ref_digest = digest(reference)
    if not args.skip_hash_check and ref_digest != REFERENCE_MANIFEST_SHA256:
        raise ValueError(f"Unexpected reference manifest hash: {ref_digest}")

    manifest = output / "split_manifest.csv"
    records, protocol = split_manifest(reference, manifest)
    protocol["reference_manifest_sha256"] = ref_digest
    if not args.split_only:
        inputs = args.processed_dir.resolve()
        if not args.skip_hash_check:
            for filename, expected in FEATURES.items():
                actual = digest(inputs / filename)
                if actual != expected:
                    raise ValueError(f"{filename}: unexpected hash {actual}")
                print(f"Verified {filename}", flush=True)
        protocol["frozen_input_sha256"] = FEATURES
    save_json(output / "split_protocol.json", protocol)
    print("Split counts:", protocol["counts"], flush=True)

    if args.raw_images is not None:
        audit = audit_hashes(records, args.raw_images.resolve())
        save_json(output / "dhash_audit.json", audit)
        print("dHash audit:", json.dumps(audit), flush=True)

    if args.split_only:
        print(f"Split manifest written to {manifest} (SHA-256: {protocol['manifest_sha256']})")
        return

    import torch

    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --allow-cpu to train on CPU")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    save_json(output / "code_fingerprints.json", {
        name: digest(code / name)
        for name in ("main.py", "utils.py", "train.py", "model.py", "config.py")
    })

    inputs = args.processed_dir.resolve()
    for seed in SEEDS:
        for variant in METHODS:
            target = output / f"seed_{seed}" / variant
            target.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, MPLBACKEND="Agg")
            cmd = [
                sys.executable, "-B", "-u", str(code / "main.py"),
                "--seed", str(seed), "--perturbation_seed", "20261120",
                "--frozen_manifest", str(manifest),
                "--train_csv_path", str(inputs / "zerofill_landmarks_mask.csv"),
                "--image_features_path", str(inputs / "zerofill_visual_tokens_layer4_compact.npy"),
                "--output_dir", str(target),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--include_skeleton_zeroing",
            ]
            if variant in ("F", "Uniform"):
                cmd += ["--study_name", "asl_matched", "--matched_variant", variant]
            else:
                cmd += [
                    "--study_name", "fair_e1_e7",
                    "--baseline_method", variant,
                    "--efficientnet_features_path", str(inputs / "efficientnet_b0_global.npy"),
                    "--global_image_features_path", str(inputs / "zerofill_visual_global_from_tokens.npy"),
                ]
            save_json(target / "command.json", {"argv": cmd, "selected_baseline": None if variant in ("F", "Uniform") else variant})
            print(f"START {seed} {variant}", flush=True)
            with (target / "console.log").open("x", encoding="utf-8") as stream:
                subprocess.run(cmd, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
            for condition in ("clean", "skeleton_zeroing"):
                if not (target / f"predictions_{condition}.csv").is_file():
                    raise ValueError(f"Missing {condition} predictions in {target}")
            print(f"COMPLETE {seed} {variant}", flush=True)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output directory for blocked-split run.")
    parser.add_argument("--reference-manifest", type=Path,
                        default=root / "manifests/asl/split_manifest_seed42.csv",
                        help="Reference seed-42 ASL Alphabet manifest used to index all 87,000 rows.")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/asl_alphabet/processed_train",
                        help="Directory with frozen ASL Alphabet landmarks CSV and .npy visual features.")
    parser.add_argument("--raw-images", type=Path, default=None,
                        help="Optional raw ASL Alphabet image root to compute cross-partition dHash audit.")
    parser.add_argument("--baselines", type=Path,
                        default=Path(__file__).resolve().parents[1] / "baselines",
                        help="Path to scg_vtnet/baselines directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split-only", action="store_true",
                        help="Generate split_manifest.csv and split_protocol.json without training.")
    parser.add_argument("--skip-hash-check", action="store_true",
                        help="Skip SHA-256 checks against the published ASL Alphabet input snapshot.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Permit training without CUDA.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
