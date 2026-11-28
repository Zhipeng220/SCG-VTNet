"""Recompute and verify structured landmark-swap accuracy drops and stratified bootstrap CI."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 2026112314


def verify(swap_dir, matched_seed42_dir=None, verify_existing=False):
    swap_dir = swap_dir.resolve()
    frame = pd.read_csv(swap_dir / "paired_predictions.csv").sort_values("original_index")
    assert len(frame) == 213 and frame["original_index"].is_unique and frame["label"].nunique() == 26
    truth = frame["label"].to_numpy()
    ids = frame["original_index"].to_numpy()

    by_model = {}
    rows = []
    for variant in ("F", "A0"):
        if matched_seed42_dir is not None:
            ref = pd.read_csv(matched_seed42_dir / variant / "predictions_clean.csv").set_index(
                "original_index", verify_integrity=True
            ).loc[ids]
            assert np.array_equal(ref["true_label"].to_numpy(), truth)
            assert np.array_equal(ref["predicted_label"].to_numpy(), frame[f"{variant}_clean"].to_numpy()), variant

        before = frame[f"{variant}_clean"].to_numpy() == truth
        after = frame[f"{variant}_swap"].to_numpy() == truth
        drop = before.astype(int) - after.astype(int)
        by_model[variant] = drop
        rows.append({
            "variant": variant,
            "n": len(ids),
            "clean_correct": int(before.sum()),
            "swap_correct": int(after.sum()),
            "clean_accuracy_pct": 100.0 * float(before.mean()),
            "swap_accuracy_pct": 100.0 * float(after.mean()),
            "drop_pp": 100.0 * float(drop.mean()),
            "correct_to_wrong": int((before & ~after).sum()),
            "wrong_to_correct": int((~before & after).sum()),
        })

    delta = by_model["F"] - by_model["A0"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(N_BOOTSTRAP)
    for label in sorted(set(truth)):
        strata = delta[truth == label]
        draws += rng.choice(strata, size=(N_BOOTSTRAP, len(strata)), replace=True).sum(axis=1)
    low, high = np.percentile(100.0 * draws / len(ids), [2.5, 97.5])
    diff_pp = 100.0 * float(delta.mean())
    summary = pd.DataFrame(rows)

    if verify_existing:
        saved_summary = pd.read_csv(swap_dir / "summary.csv")
        saved_protocol = json.loads((swap_dir / "protocol_and_contrast.json").read_text(encoding="utf-8"))
        assert len(summary) == len(saved_summary) == 2
        for col in ("variant", "n", "clean_correct", "swap_correct", "correct_to_wrong", "wrong_to_correct"):
            assert np.array_equal(summary[col].to_numpy(), saved_summary[col].to_numpy()), col
        for col in ("clean_accuracy_pct", "swap_accuracy_pct", "drop_pp"):
            assert np.allclose(summary[col].to_numpy(), saved_summary[col].to_numpy(), atol=1e-9), col
        assert abs(diff_pp - saved_protocol["F_minus_A0_drop_pp"]) < 1e-9
        assert np.allclose([float(low), float(high)], saved_protocol["ci95_pp"], atol=1e-9)

    print(
        f"Verified structured landmark swap ({len(ids)} images): "
        f"F-A0 drop diff = {diff_pp:+.2f} pp (95% CI [{low:.2f}, {high:.2f}])."
    )
    return summary, diff_pp, (float(low), float(high))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("swap_dir", type=Path, help="Path to results/reviewer11_14/r14_structured_swap.")
    parser.add_argument("--matched-seed42-dir", type=Path, default=None,
                        help="Optional path to results/asl_matched/seed_42 to verify clean replay predictions.")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Assert exact agreement with saved summary.csv and protocol_and_contrast.json.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.swap_dir, args.matched_seed42_dir, args.verify_existing)
