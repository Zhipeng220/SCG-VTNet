"""Recompute and verify ASL blocked-split per-seed and summary metrics."""

import argparse
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44)
MODELS = ("F", "Uniform", "E2", "E5")
CONDITIONS_BY_MODEL = {
    "F": ("clean", "skeleton_zeroing", "visual_patch_occlusion", "gaussian_1.0"),
    "Uniform": ("clean", "skeleton_zeroing", "visual_patch_occlusion", "gaussian_1.0"),
    "E2": ("clean", "skeleton_zeroing"),
    "E5": ("clean", "skeleton_zeroing", "gaussian_1.0"),
}
STRATA = ("all", "detected", "undetected")


def verify(blocked_dir, output_dir=None, verify_existing=False):
    blocked_dir = blocked_dir.resolve()
    manifest = pd.read_csv(blocked_dir / "split_manifest.csv")
    assert len(manifest) == 87000
    assert manifest["original_index"].nunique() == 87000
    assert set(manifest["split"]) == {"train", "val", "test", "buffer"}
    test_manifest = manifest.loc[manifest["split"] == "test"].sort_values("original_index")
    expected_indices = test_manifest["original_index"].to_numpy()
    expected_labels = test_manifest["label"].to_numpy()
    expected_detected = test_manifest["hand_detected"].to_numpy()

    per_seed_rows = []
    for seed in SEEDS:
        for model in MODELS:
            for condition in CONDITIONS_BY_MODEL[model]:
                pred_path = blocked_dir / f"seed_{seed}" / model / f"predictions_{condition}.csv"
                frame = pd.read_csv(pred_path).sort_values("original_index")
                if not np.array_equal(frame["original_index"].to_numpy(), expected_indices):
                    raise ValueError(f"Test index mismatch in {pred_path}")
                if not np.array_equal(frame["true_label"].to_numpy(), expected_labels):
                    raise ValueError(f"True label mismatch in {pred_path}")
                if not np.array_equal(frame["hand_detected"].to_numpy(), expected_detected):
                    raise ValueError(f"Detection mask mismatch in {pred_path}")

                for stratum in STRATA:
                    if stratum == "detected":
                        sub = frame.loc[frame["hand_detected"] == 1]
                    elif stratum == "undetected":
                        sub = frame.loc[frame["hand_detected"] == 0]
                    else:
                        sub = frame
                    correct = int((sub["true_label"] == sub["predicted_label"]).sum())
                    per_seed_rows.append({
                        "seed": seed,
                        "model": model,
                        "condition": condition,
                        "stratum": stratum,
                        "n": len(sub),
                        "correct": correct,
                        "accuracy_pct": float(100.0 * correct / len(sub)),
                    })

    per_seed = pd.DataFrame(per_seed_rows)

    summary_rows = []
    for model in MODELS:
        for condition in CONDITIONS_BY_MODEL[model]:
            for stratum in STRATA:
                sub = per_seed.loc[
                    (per_seed["model"] == model)
                    & (per_seed["condition"] == condition)
                    & (per_seed["stratum"] == stratum)
                ]
                vals = sub["accuracy_pct"].tolist()
                summary_rows.append({
                    "model": model,
                    "condition": condition,
                    "stratum": stratum,
                    "n_training_seeds": len(vals),
                    "accuracy_mean_pct": mean(vals),
                    "accuracy_sd_pct": stdev(vals),
                })
    summary = pd.DataFrame(summary_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(output_dir / "verified_per_seed.csv", index=False)
        summary.to_csv(output_dir / "verified_summary.csv", index=False)

    if verify_existing:
        saved_ps = pd.read_csv(blocked_dir / "verified_per_seed.csv")
        saved_sm = pd.read_csv(blocked_dir / "verified_summary.csv")
        assert len(per_seed) == len(saved_ps) == 117
        assert len(summary) == len(saved_sm) == 39
        for col in ("seed", "model", "condition", "stratum", "n", "correct"):
            assert np.array_equal(per_seed[col].to_numpy(), saved_ps[col].to_numpy()), f"per_seed mismatch in {col}"
        assert np.allclose(per_seed["accuracy_pct"].to_numpy(), saved_ps["accuracy_pct"].to_numpy(), atol=1e-9)
        for col in ("model", "condition", "stratum", "n_training_seeds"):
            assert np.array_equal(summary[col].to_numpy(), saved_sm[col].to_numpy()), f"summary mismatch in {col}"
        for col in ("accuracy_mean_pct", "accuracy_sd_pct"):
            assert np.allclose(summary[col].to_numpy(), saved_sm[col].to_numpy(), atol=1e-9), f"summary mismatch in {col}"
        print(
            f"Verified blocked-split manifest ({len(expected_indices)} test images), "
            f"{len(per_seed)} per-seed rows, and {len(summary)} summary rows."
        )
    else:
        print(f"Computed {len(per_seed)} per-seed rows and {len(summary)} summary rows from {blocked_dir}.")
    return per_seed, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blocked_dir", type=Path, help="Path to results/asl_blocked.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional directory to write verified_per_seed.csv and verified_summary.csv.")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Assert exact agreement with existing verified_per_seed.csv and verified_summary.csv.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.blocked_dir, args.output_dir, args.verify_existing)
