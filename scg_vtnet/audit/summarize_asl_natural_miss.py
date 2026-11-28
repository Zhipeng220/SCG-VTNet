"""Summarize and verify natural-miss (hand_detected == 0) accuracy on matched ASL runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44)
VARIANTS = ("F", "A0", "G0", "Uniform")


def summarize(matched_dir, output_csv=None, verify_csv=None):
    matched_dir = matched_dir.resolve()
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            rel = Path("asl_matched") / f"seed_{seed}" / variant / "predictions_clean.csv"
            pred_path = matched_dir / f"seed_{seed}" / variant / "predictions_clean.csv"
            frame = pd.read_csv(pred_path)
            undetected = frame.loc[frame["hand_detected"] == 0]
            correct = int((undetected["true_label"] == undetected["predicted_label"]).sum())
            rows.append({
                "seed": seed,
                "variant": variant,
                "test_n": len(frame),
                "undetected_n": len(undetected),
                "undetected_correct": correct,
                "accuracy_pct": float(100.0 * correct / len(undetected)),
                "source_prediction": rel.as_posix(),
            })

    result = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)

    if verify_csv is not None:
        expected = pd.read_csv(verify_csv)
        assert len(result) == len(expected) == 12
        for col in ("seed", "variant", "test_n", "undetected_n", "undetected_correct", "source_prediction"):
            assert np.array_equal(result[col].to_numpy(), expected[col].to_numpy()), f"Mismatch in {col}"
        assert np.allclose(result["accuracy_pct"].to_numpy(), expected["accuracy_pct"].to_numpy(), atol=1e-9)
        print(f"Verified 12 natural-miss rows against {verify_csv}.")
    else:
        print(f"Computed 12 natural-miss rows from {matched_dir}.")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched_dir", type=Path, help="Path to results/asl_matched.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--verify", type=Path, default=None,
                        help="Optional archived per_seed.csv to verify against.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args.matched_dir, args.output, args.verify)
