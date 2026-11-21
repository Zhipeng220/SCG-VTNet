"""Recompute the full, A–Z and nothing detection strata for main-text Table 6."""
import argparse
import csv
from pathlib import Path
from string import ascii_uppercase

LETTERS = set(ascii_uppercase)

METHODS = ("E4", "E7", "E2", "E5")


def strata(directory):
    reference = None
    for method in METHODS:
        path = directory / f"{method}_seed_42.csv"
        with path.open(encoding="utf8", newline="") as stream:
            data = list(csv.DictReader(stream))
        keys = {row["original_index"]: (row["true_label"], row["hand_detected"], row["image_path"])
                for row in data}
        if len(data) != 17400 or len(keys) != len(data):
            raise ValueError(f"Expected 17,400 unique seed-42 test rows in {path}")
        if reference is not None and keys != reference:
            raise ValueError(f"Sample population differs from {METHODS[0]}: {path}")
        reference = keys
        groups = {
            "all": data,
            "A-Z": [row for row in data if row["true_label"] in LETTERS],
            "nothing": [row for row in data if row["true_label"] == "nothing"],
        }
        for group, rows in groups.items():
            for detected in (0, 1):
                group_rows = [row for row in rows if row["hand_detected"] == str(detected)]
                correct = sum(row["true_label"] == row["predicted_label"] for row in group_rows)
                yield method, group, detected, len(group_rows), correct, round(correct / len(group_rows) * 100, 6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(strata(args.prediction_directory))
    with args.output.open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "group", "detected", "n", "correct", "accuracy_pct"])
        writer.writerows(rows)

