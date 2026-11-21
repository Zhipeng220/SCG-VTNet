"""Aggregate the completed matched ASL runs and paired differences from F."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
from sklearn.metrics import f1_score


SEEDS = (42, 43, 44)
VARIANTS = ("F", "A0", "G0", "Uniform")


def run(root, output):
    records = []
    for seed in SEEDS:
        baseline = {}
        for variant in VARIANTS:
            directory = root / f"seed_{seed}" / variant
            config = json.loads((directory / f"config_{variant}.json").read_text(encoding="utf8"))
            history = json.loads((directory / "training_history.json").read_text(encoding="utf8"))
            predictions = {path.stem.removeprefix("predictions_"): path
                           for path in directory.glob("predictions_*.csv")}
            if variant == "F" and ("clean" not in predictions or len(predictions) != 9):
                raise ValueError(f"Incomplete evaluation conditions in {directory}")
            if variant != "F" and predictions.keys() != baseline.keys():
                raise ValueError(f"Conditions differ from F in {directory}")
            for condition, path in predictions.items():
                frame = pd.read_csv(path).set_index("original_index", verify_integrity=True).sort_index()
                if len(frame) != 17400:
                    raise ValueError(f"Expected 17,400 ASL test images in {path}")
                truth = frame[["true_label", "hand_detected"]]
                scores = {"accuracy_pct": (frame.true_label == frame.predicted_label).mean() * 100,
                          "macro_f1_pct": f1_score(frame.true_label, frame.predicted_label,
                                                   average="macro", zero_division=0) * 100}
                if variant == "F":
                    baseline[condition] = (truth, scores)
                reference, original_scores = baseline[condition]
                if not truth.equals(reference):
                    raise ValueError(f"Test population differs from F for {path}")
                params = config["parameters"]
                unused = config["unused_query_key_parameters"]
                records.append({
                    "seed": seed, "variant": variant, "condition": condition, "n": len(frame),
                    "selected_epoch": history["selected_epoch"],
                    "parameters": params, "effective_parameters": params - unused,
                    "unused_query_key_parameters": unused,
                    **scores, **{key.replace("_pct", "_delta_pp"): value - original_scores[key]
                                 for key, value in scores.items()},
                })
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_seed.csv", index=False)
    metric_cols = ["accuracy_pct", "macro_f1_pct", "accuracy_delta_pp", "macro_f1_delta_pp"]
    summary = []
    for (variant, condition), group in frame.groupby(["variant", "condition"]):
        row = {"variant": variant, "condition": condition, "seed_count": len(group)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = mean(group[metric])
            row[f"{metric}_sd"] = stdev(group[metric])
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.runs, args.output)

