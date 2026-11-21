"""Recompute paired ASL accuracy, macro-F1, splits and clean confusion counts."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score


def report(frame, output):
    keys = ["method", "condition", "original_index"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate prediction for model, condition and sample")
    clean = frame.loc[frame.condition == "clean", ["method", "original_index", "prediction"]]
    frame = frame.merge(clean, on=["method", "original_index"], how="left",
                        validate="many_to_one", suffixes=("", "_clean"))
    if frame.prediction_clean.isna().any():
        raise ValueError("Condition missing paired clean prediction")
    letters = frame.true_label.str.fullmatch("[A-Z]")
    nothing = frame.true_label == "nothing"
    frame["label_group"] = "del/space"
    frame.loc[letters, "label_group"] = "A-Z"
    frame.loc[nothing, "label_group"] = "nothing"
    frame["transition"] = frame.clean_detected.astype(str) + "->" + frame.condition_detected.astype(str)
    frame["correct"] = frame.prediction == frame.true_label
    frame["clean_correct"] = frame.prediction_clean == frame.true_label
    groupings = {"all": None, "label_group": "label_group", "class": "true_label",
                 "detected": "condition_detected", "transition": "transition"}
    rows = []
    for stratum, column in groupings.items():
        group_cols = ["method", "condition"] + ([column] if column else [])
        for key, subset in frame.groupby(group_cols, sort=True):
            accuracy = subset.correct.mean() * 100
            paired_accuracy = subset.clean_correct.mean() * 100
            rows.append({"method": key[0], "condition": key[1], "stratum": stratum,
                         "level": key[2] if column else "all", "n": len(subset),
                         "correct": int(subset.correct.sum()), "accuracy_pct": accuracy,
                         "macro_f1_pct": f1_score(subset.true_label, subset.prediction,
                                                  labels=sorted(subset.true_label.unique()),
                                                  average="macro", zero_division=0) * 100,
                         "paired_clean_accuracy_pct": paired_accuracy,
                         "delta_pp": accuracy - paired_accuracy})
    pd.DataFrame(rows).to_csv(output / "stratified_metrics.csv", index=False)
    (frame.loc[(frame.condition == "clean") & ~frame.correct]
     .groupby(["method", "true_label", "prediction"]).size().rename("count")
     .sort_values(ascending=False).to_csv(output / "clean_confusions.csv"))
    clean_predictions = frame[frame.condition == "clean"].pivot(
        index="original_index", columns="method", values="prediction")
    if clean_predictions.isna().any().any():
        raise ValueError("Clean predictions do not cover identical samples")
    metadata = frame[frame.condition == "clean"].drop_duplicates("original_index").set_index("original_index")
    clean_predictions = metadata[["image_path", "true_label", "clean_detected"]].join(clean_predictions)
    clean_predictions.to_csv(output / "paired_clean_predictions.csv")
    cases = frame[frame.method == "E7"].sort_values(["condition", "original_index"])
    conditions = [
        ("detection_lost_and_prediction_correct", (cases.transition == "1->0") & cases.correct),
        ("detection_lost_and_prediction_wrong", (cases.transition == "1->0") & ~cases.correct),
        ("detected_clean_correct_then_wrong", (cases.transition == "1->1") & cases.clean_correct & ~cases.correct),
    ]
    candidates = []
    for label, mask in conditions:
        candidates.append(cases[mask].groupby("condition", sort=True).head(1).assign(selection_rule=label))
    pd.concat(candidates).to_csv(output / "case_candidates.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--historical-clean", action="store_true")
    args = parser.parse_args()
    if args.historical_clean:
        frame = pd.concat([pd.read_csv(path) for path in sorted(args.input.glob("E*_seed_42.csv"))])
        frame = frame.rename(columns={"predicted_label": "prediction"})
        frame = frame.assign(condition="clean", clean_detected=frame.hand_detected,
                             condition_detected=frame.hand_detected)
        report(frame, args.input)
    else:
        frame = pd.read_csv(args.input / "predictions.csv")
        samples = pd.read_csv(args.input / "samples.csv")
        for condition in ("clean", "dark_light", "dark_heavy", "blur_light", "blur_heavy",
                          "occlude_light", "occlude_heavy", "xyz_jitter", "zero_skeleton"):
            rows = frame[frame.condition == condition]
            expected = set(samples.original_index)
            if condition in ("xyz_jitter", "zero_skeleton"):
                expected = set(frame.loc[(frame.condition == "clean") & (frame.clean_detected == 1),
                                         "original_index"])
            for method in ("E2", "E5", "E7"):
                if set(rows.loc[rows.method == method, "original_index"]) != expected:
                    raise ValueError(f"Incomplete coverage: {condition}, {method}")
        report(frame, args.input)

