import argparse
import csv
import os
import random

import numpy as np
import torch

from main import load_experiment_config
from train import test_robustness
from utils import load_and_preprocess_train_data


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_single_file(folder, prefix, suffix):
    matches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file in {folder} with prefix={prefix}, suffix={suffix}, got {matches}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--image_features_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)

    device = build_device()

    config_path = find_single_file(args.exp_dir, "config_", ".json")
    weights_dir = os.path.join(args.exp_dir, "weights")
    model_path = find_single_file(weights_dir, "best_model_", "_21kpts.pth")

    config = load_experiment_config(config_path)
    use_advanced_features = config.get("use_advanced_features", True)

    datasets, _, label_encoder, _, input_dim = load_and_preprocess_train_data(
        args.train_csv_path,
        args.image_features_path,
        use_advanced_features=use_advanced_features,
        seed=args.seed,
    )

    image_feature_dim = datasets["test"].image_feature_dim

    runtime_args = argparse.Namespace(
        batch_size=args.batch_size,
        embed_dim=config.get("embed_dim", 256),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.2),
        use_keypoint_refinement=config.get("use_keypoint_refinement", False),
        refine_scale=config.get("refine_scale", 0.05),
    )

    results = test_robustness(
        config["model_class"],
        model_path,
        datasets["test"],
        device,
        label_encoder,
        runtime_args,
        input_dim,
        config,
        image_feature_dim=image_feature_dim,
    )

    if results is None:
        raise RuntimeError("test_robustness returned None.")

    output_csv = args.output_csv or os.path.join(args.exp_dir, f"robustness_seed_{args.seed}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    row = {
        "seed": args.seed,
        "experiment": config.get("experiment_name", os.path.basename(args.exp_dir)),
        "model_path": model_path,
    }
    for key, value in results.items():
        row[key] = value * 100.0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Saved robustness results to: {output_csv}")


if __name__ == "__main__":
    main()