"""Run the four matched ASL variants with three paired training seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root, output):
    revision = Path(__file__).resolve().parents[1]
    code = revision / "baselines"
    inputs = root / "data/asl_alphabet/processed_train"
    manifest_dir = root / "manifests/asl"
    freeze = json.loads((manifest_dir / "freeze.json").read_text(encoding="utf8"))
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    for seed in (42, 43, 44):
        manifest = manifest_dir / f"split_manifest_seed{seed}.csv"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != freeze["manifests"][str(seed)]["frozen_sha256"]:
            raise ValueError(f"Unexpected manifest hash for seed {seed}")
        for variant in ("F", "A0", "G0", "Uniform"):
            target = output / f"seed_{seed}" / variant
            command = [
                sys.executable, "-B", "-u", str(code / "main.py"),
                "--study_name", "asl_matched", "--matched_variant", variant,
                "--seed", str(seed), "--perturbation_seed", "20261120",
                "--frozen_manifest", str(manifest),
                "--train_csv_path", str(inputs / "zerofill_landmarks_mask.csv"),
                "--image_features_path", str(inputs / "zerofill_visual_tokens_layer4_compact.npy"),
                "--output_dir", str(target), "--epochs", "50", "--batch_size", "128",
            ]
            target.mkdir(parents=True, exist_ok=False)
            (target / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf8")
            print(f"ASL seed {seed}, variant {variant}", flush=True)
            with (target / "console.log").open("w", encoding="utf8") as log:
                subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

