# SCG-VTNet reproduction code, 2026-11-21

This directory contains the source programs used for the revised evaluations and associated checks. The scripts preserve the original directory layout used by the experiment launchers. The top-level baselines in the parent directory expose the matched ASL configuration in addition to the historical configurations.

| Evaluation | Source entry points |
| --- | --- |
| ASL paired image degradation and skeleton intervention | protocol/asl_experiment_a_validation.py, protocol/summarize_asl_pairs.py, recompute_paired_a.py |
| ASL matched F/A0/G0/Uniform | protocol/run_asl_matched.py, protocol/summarize_asl_matched.py |
| NUS colour/grayscale paired evaluation | protocol/run_nus_strict.py, verify_grouped_nus.py |
| Detection stratification and CPU timing | protocol/summarize_asl_detection.py, audit/benchmark_end_to_end_cpu.py |
| Figure panels and diagnostics | protocol/export_asl_case_panels.py, code/core/visualizations/ |

Paths beginning with protocol/, audit/ or code/ are relative to SCG-VTNet_修回实验_20261115/ in this directory. The adjacent AC-GCAT/data/ directory supplies preprocessing scripts. Input paths and any machine-specific historical defaults must be resolved against the original experimental inputs.

For offline checks, pass the prediction CSV directory and an output directory to summarize_asl_matched.py. The paired ASL resampling script takes predictions.csv and an output CSV path. The NUS check takes the run directory and the frozen NUS manifest. These checks use Python, pandas and scikit-learn. Full training and inference also require original datasets, frozen split manifests, feature extraction dependencies and, for the checkpoint-based evaluations, the corresponding weights.

This Git revision provides software sources. The local results archive contains the frozen manifests, evaluation outputs and model checkpoints. A citable archival DOI is a separate publication step.
