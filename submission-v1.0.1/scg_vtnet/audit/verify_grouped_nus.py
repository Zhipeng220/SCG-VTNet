"""Verify NUS predictions and published metrics against the frozen test split."""

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def verify(run_dir, manifest_path):
    manifest = pd.read_csv(manifest_path)
    held_out = manifest.loc[manifest.split == "test"]
    expected = set(held_out.original_index)
    assert len(expected) == 96 and len(held_out) == 96
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf8"))
    assert len(metrics) == 45
    assert len({(r["config"], r["training_seed"], r["scenario"]) for r in metrics}) == 45
    for record in metrics:
        config, seed, scenario = (record[k] for k in ("config", "training_seed", "scenario"))
        predictions = pd.read_csv(run_dir / f"{config}_seed{seed}" / f"predictions_{scenario}.csv")
        assert len(predictions) == 96 and set(predictions.original_index) == expected
        assert abs(accuracy_score(predictions.true_label, predictions.pred_label) - record["accuracy"]) < 1e-12
        assert abs(f1_score(predictions.true_label, predictions.pred_label, average="macro") - record["macro_f1"]) < 1e-12
        if config == "R0":
            assert record["refined_diagnostic"] is None
    print("Verified 45 NUS prediction files and both metrics.")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    verify(args.run_dir, args.manifest)
