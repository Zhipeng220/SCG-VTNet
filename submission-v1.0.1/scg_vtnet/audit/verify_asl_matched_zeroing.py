"""Recompute and verify matched ASL skeleton-zeroing metrics from saved predictions."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44)
VARIANTS = ("F", "A0", "G0", "Uniform")


def verify(zeroing_dir, matched_dir=None, output_json=None):
    zeroing_dir = zeroing_dir.resolve()
    saved_summary = pd.read_csv(zeroing_dir / "summary.csv")
    saved_paired = pd.read_csv(zeroing_dir / "paired.csv")

    summary_rows = []
    paired_rows = []
    clean_rows_verified = 0
    zero_acc_by_variant = {v: [] for v in VARIANTS}
    zero_diff_by_comp = {v: [] for v in VARIANTS[1:]}

    for seed in SEEDS:
        clean_correct_by_variant = {}
        zero_correct_by_variant = {}
        ref_indices = None
        for variant in VARIANTS:
            frame = pd.read_csv(zeroing_dir / f"seed_{seed}_{variant}.csv")
            if ref_indices is None:
                ref_indices = frame["original_index"].to_numpy()
            elif not np.array_equal(frame["original_index"].to_numpy(), ref_indices):
                raise ValueError(f"Population mismatch in seed_{seed}_{variant}.csv")

            if matched_dir is not None:
                ref_path = matched_dir / f"seed_{seed}" / variant / "predictions_clean.csv"
                ref = pd.read_csv(ref_path)
                ref_det = ref.loc[ref["hand_detected"] == 1].sort_values("original_index")
                if not np.array_equal(ref_det["original_index"].to_numpy(), ref_indices):
                    raise ValueError(f"Detected index mismatch against {ref_path}")
                if not np.array_equal(ref_det["true_label"].to_numpy(), frame["true_label"].to_numpy()):
                    raise ValueError(f"True label mismatch against {ref_path}")
                disagreements = int(np.count_nonzero(
                    frame["clean_prediction"].to_numpy() != ref_det["predicted_label"].to_numpy()
                ))
                if disagreements != 0:
                    raise ValueError(f"{seed}/{variant}: {disagreements} clean replay disagreements")
                clean_rows_verified += len(frame)

            clean_ok = frame["clean_prediction"].to_numpy() == frame["true_label"].to_numpy()
            zero_ok = frame["zero_prediction"].to_numpy() == frame["true_label"].to_numpy()
            clean_correct_by_variant[variant] = clean_ok
            zero_correct_by_variant[variant] = zero_ok
            zero_acc = float(100.0 * zero_ok.mean())
            zero_acc_by_variant[variant].append(zero_acc)
            summary_rows.append({
                "seed": seed,
                "variant": variant,
                "n": len(frame),
                "clean_correct": int(clean_ok.sum()),
                "zero_correct": int(zero_ok.sum()),
                "clean_accuracy_pct": float(100.0 * clean_ok.mean()),
                "zero_accuracy_pct": zero_acc,
                "zero_minus_clean_pp": float(100.0 * (zero_ok.mean() - clean_ok.mean())),
            })

        for comparator in VARIANTS[1:]:
            for condition, group in (("clean", clean_correct_by_variant), ("zero", zero_correct_by_variant)):
                lhs, rhs = group["F"], group[comparator]
                diff = float(100.0 * (lhs.mean() - rhs.mean()))
                if condition == "zero":
                    zero_diff_by_comp[comparator].append(diff)
                paired_rows.append({
                    "seed": seed,
                    "condition": condition,
                    "comparator": comparator,
                    "n": len(lhs),
                    "both_correct": int((lhs & rhs).sum()),
                    "only_F_correct": int((lhs & ~rhs).sum()),
                    "only_comparator_correct": int((~lhs & rhs).sum()),
                    "both_wrong": int((~lhs & ~rhs).sum()),
                    "F_minus_comparator_pp": diff,
                })

    recomputed_summary = pd.DataFrame(summary_rows)
    recomputed_paired = pd.DataFrame(paired_rows)
    assert len(recomputed_summary) == len(saved_summary) == 12
    assert len(recomputed_paired) == len(saved_paired) == 18

    for col in ("seed", "variant", "n", "clean_correct", "zero_correct"):
        assert np.array_equal(recomputed_summary[col].to_numpy(), saved_summary[col].to_numpy()), col
    for col in ("clean_accuracy_pct", "zero_accuracy_pct", "zero_minus_clean_pp"):
        assert np.allclose(recomputed_summary[col].to_numpy(), saved_summary[col].to_numpy(), atol=1e-9), col

    for col in ("seed", "condition", "comparator", "n", "both_correct",
                "only_F_correct", "only_comparator_correct", "both_wrong"):
        assert np.array_equal(recomputed_paired[col].to_numpy(), saved_paired[col].to_numpy()), col
    assert np.allclose(
        recomputed_paired["F_minus_comparator_pp"].to_numpy(),
        saved_paired["F_minus_comparator_pp"].to_numpy(),
        atol=1e-9,
    )

    report = {
        "runs": len(recomputed_summary),
        "paired_rows_verified": len(recomputed_paired),
        "clean_row_predictions_verified": clean_rows_verified,
        "zero_skeleton_accuracy_pct": {
            v: {"mean": mean(vals), "sample_sd": stdev(vals)}
            for v, vals in zero_acc_by_variant.items()
        },
        "zero_skeleton_F_minus_comparator_pp": {
            c: {
                "mean": mean(vals),
                "minimum": min(vals),
                "maximum": max(vals),
                "F_wins": sum(x > 0 for x in vals),
                "per_seed": vals,
            }
            for c, vals in zero_diff_by_comp.items()
        },
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Verified {report['runs']} matched skeleton-zeroing runs, "
        f"{report['paired_rows_verified']} paired rows, and "
        f"{report['clean_row_predictions_verified']} clean replay predictions."
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zeroing_dir", type=Path, help="Path to results/asl_matched_zeroing.")
    parser.add_argument("--matched-dir", type=Path, default=None,
                        help="Optional path to results/asl_matched to verify 161,076 clean replay predictions.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to write recomputed summary JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.zeroing_dir, args.matched_dir, args.output_json)
