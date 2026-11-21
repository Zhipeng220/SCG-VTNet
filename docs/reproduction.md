# Reproduce the experiments

Run examples from the repository root after following [data preparation](data_preparation.md).
The original environment uses Python 3.10, PyTorch 2.7.1, torchvision 0.22.1,
NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2 and MediaPipe 0.10.14.
The authors used CUDA 11.8 for GPU runs.

| Result | Entry points |
| --- | --- |
| ASL E1–E7 model comparison | `scripts/experiments/train_models.py`, `scripts/experiments/run_model_comparison.ps1`, `scripts/analysis/verify_asl_comparison.py` |
| Figure 1 anatomical descriptors | `scripts/analysis/plot_anatomical_descriptors.py` |
| Figure 3 token versus global robustness | `scripts/experiments/evaluate_feature_robustness.py`, `scripts/analysis/plot_feature_robustness.py` |
| Paired input degradation | `scripts/experiments/evaluate_input_robustness.py`, `scripts/analysis/recompute_paired_robustness.py` |
| ASL matched ablations | `scripts/experiments/train_matched_asl.py`, `scripts/analysis/summarize_matched_asl.py` |
| NUS paired grouping | `scripts/experiments/train_grouped_nus.py`, `scripts/analysis/verify_grouped_nus.py` |
| CPU timing | `scripts/benchmarks/time_classifier_heads.py`, `scripts/benchmarks/time_image_to_label.py` |

```sh
python -m scripts.experiments.train_models --help
python -m scripts.experiments.evaluate_feature_robustness --output_csv ../runs/asl_alphabet/feature_robustness/metrics.csv
python -m scripts.analysis.plot_feature_robustness --metrics-csv ../runs/asl_alphabet/feature_robustness/metrics.csv --output-dir ../runs/asl_alphabet/feature_robustness/figures
python -m scripts.experiments.evaluate_input_robustness --paired-test --per-class 10 --output ../runs/asl_alphabet/input_robustness
python -m scripts.benchmarks.time_classifier_heads --output ../runs/asl_alphabet/timing/heads.json
```

The ASL historical re-evaluation additionally requires the archived
`table_70_file_mapping.csv` alongside the `fair_e1_e7` run directories. To
use the PowerShell launcher, install the package into the selected interpreter
and run `./scripts/experiments/run_model_comparison.ps1 -PythonExe python -ValidateOnly`
to check the entry point. Supply the feature and landmark files to run training.
