"""Recompute and verify zero-shot ASL Alphabet to ASL Dataset metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import string

import numpy as np
import pandas as pd

LETTERS = tuple(string.ascii_uppercase)
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("E2", "E5", "E7")


def macro_f1(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        scores.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(zero_shot_dir, source_runs=None, output_dir=None, verify_existing=False):
    zero_shot_dir = zero_shot_dir.resolve()
    target_letters = pd.read_csv(zero_shot_dir / "target_letters.csv")
    assert len(target_letters) == 1815
    expected_indices = target_letters["original_index"].to_numpy()
    expected_labels = target_letters["true_label"].to_numpy()
    expected_detected = target_letters["hand_detected"].to_numpy()

    rows = []
    for seed in SEEDS:
        for method in METHODS:
            pred_path = zero_shot_dir / f"predictions_{method}_{seed}.csv"
            frame = pd.read_csv(pred_path)
            if not np.array_equal(frame["original_index"].to_numpy(), expected_indices):
                raise ValueError(f"Index mismatch in {pred_path}")
            if not np.array_equal(frame["true_label"].to_numpy(), expected_labels):
                raise ValueError(f"Label mismatch in {pred_path}")
            if not np.array_equal(frame["hand_detected"].to_numpy(), expected_detected):
                raise ValueError(f"Detection mismatch in {pred_path}")

            truth = frame["true_label"].to_numpy()
            pred29 = frame["prediction_29"].to_numpy()
            pred26 = frame["prediction_26"].to_numpy()
            rows.append({
                "source_seed": seed,
                "method": method,
                "n": len(frame),
                "accuracy_29_pct": float(100.0 * np.mean(truth == pred29)),
                "macro_f1_29_pct": float(100.0 * macro_f1(truth, pred29, LETTERS)),
                "non_letter_predictions": int(np.count_nonzero(~np.isin(pred29, LETTERS))),
                "accuracy_26_pct": float(100.0 * np.mean(truth == pred26)),
                "macro_f1_26_pct": float(100.0 * macro_f1(truth, pred26, LETTERS)),
            })

    per_seed = pd.DataFrame(rows)
    metrics = ["accuracy_29_pct", "macro_f1_29_pct", "accuracy_26_pct", "macro_f1_26_pct"]
    summary = per_seed.groupby("method")[metrics].agg(["mean", "std"])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(output_dir / "per_seed.csv", index=False)
        summary.to_csv(output_dir / "summary.csv")

    if verify_existing:
        saved_ps = pd.read_csv(zero_shot_dir / "per_seed.csv")
        saved_sm = pd.read_csv(zero_shot_dir / "summary.csv", header=[0, 1], index_col=0)
        assert len(per_seed) == len(saved_ps) == 15
        for col in ("source_seed", "method", "n", "non_letter_predictions"):
            assert np.array_equal(per_seed[col].to_numpy(), saved_ps[col].to_numpy()), f"per_seed mismatch in {col}"
        for col in metrics:
            assert np.allclose(per_seed[col].to_numpy(), saved_ps[col].to_numpy(), atol=1e-9), col
        assert list(summary.index) == list(saved_sm.index)
        assert np.allclose(summary.to_numpy(), saved_sm.to_numpy(), atol=1e-9)

    source_checked = 0
    if source_runs is not None:
        expected_path = zero_shot_dir / "expected_source.json"
        expected_source = json.loads(expected_path.read_text(encoding="utf-8"))
        for key, meta in expected_source.items():
            _, seed_dir, filename = key.split("/")
            candidates = [
                source_runs / seed_dir / filename,
                source_runs / seed_dir / "weights" / filename,
            ]
            found = [c for c in candidates if c.is_file()]
            if not found:
                raise FileNotFoundError(f"Missing archived source file for {key}")
            actual = sha256(found[0])
            if actual != meta["sha256"]:
                raise ValueError(f"Source hash mismatch for {key}")
            source_checked += 1

    print(
        f"Verified {len(per_seed)} zero-shot prediction files ({len(expected_indices)} target A-Z images)"
        + (f" and {source_checked} historical source artifacts." if source_checked else ".")
    )
    return per_seed, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero_shot_dir", type=Path, help="Path to results/asl_zero_shot.")
    parser.add_argument("--source-runs", type=Path, default=None,
                        help="Optional path to results/historical/fair_e1_e7 to verify all 60 source hashes.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional directory to write recomputed per_seed.csv and summary.csv.")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Assert exact agreement with existing per_seed.csv and summary.csv.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.zero_shot_dir, args.source_runs, args.output_dir, args.verify_existing)
