# SCG-VTNet

SCG-VTNet performs single-frame static hand gesture recognition using anatomy-aware 3D skeleton features and a preserved $7\times7$ ResNet-50 spatial visual feature grid. This release contains the complete source code for model training, feature extraction, ablation and robustness protocols, cross-dataset evaluation, CPU timing, and independent metric verification.

> **Archive scope:** This code deposit (`v1.0.1`; supersedes `v1.0.0` at `10.5281/zenodo.24439023`) contains all custom source code, figure-rendering scripts, and documentation only. Public source datasets remain with their original providers. Pre-extracted features, frozen split manifests, per-sample predictions, and trained checkpoints are maintained separately from this software package.

## Repository layout

| Directory | Contents |
|---|---|
| `scg_vtnet/core` | Core SCG-VTNet architecture, AAR descriptors, training objectives, and diagnostic visualisations (`visual_modulation`, `core_ablation`, `aar_construction`, `objective_construction`, secondary datasets) |
| `scg_vtnet/baselines` | Historical `fair_e1_e7` baselines, `asl_matched` (`F`, `A0`, `G0`, `Uniform`), and `asl_blocked` training entry points (`Uniform` is defined in `baselines/model.py`) |
| `scg_vtnet/extraction` | MediaPipe 3D landmark extraction, ResNet-50 spatial token extraction, and EfficientNet-B0 / ViT-S/16 backbone feature extraction |
| `scg_vtnet/protocol` | Protocol launchers for matched ASL training, matched skeleton zeroing, contiguous-block ASL training, paired image degradation, zero-shot ASL-to-ASL-Dataset transfer, and pair-grouped NUS training |
| `scg_vtnet/audit` | Independent prediction verification, paired bootstrap statistics, natural-miss stratification, blocked-split verification, zero-shot verification, and end-to-end CPU timing |

## Environment

Reference environment used for the revision evaluations: Python 3.10.20, PyTorch 2.7.1, NumPy 1.26.4, Pandas 2.2.2, and MediaPipe 0.10.14.

Install PyTorch and `torchvision` for your CPU or CUDA platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

For raw-image feature extraction, paired image degradation, and CPU end-to-end timing, also install `mediapipe`, `opencv-python`, `Pillow`, and `tqdm` (plus `timm` if extracting ViT-S/16 features).

## Expected input layout

Example commands assume execution from the workspace root where local inputs are placed:

- `manifests/asl/split_manifest_seed{42..46}.csv` and `manifests/asl/freeze.json` (frozen ASL Alphabet image-level splits)
- `manifests/nus.csv` (frozen colour--grayscale pair-grouped NUS split)
- `data/asl_alphabet/processed_train/` (`zerofill_landmarks_mask.csv`, `zerofill_visual_tokens_layer4_compact.npy`, `zerofill_visual_global_from_tokens.npy`, `efficientnet_b0_global.npy`)
- `data/asl_alphabet/images/` (raw ASL Alphabet images, for paired image degradation, dHash audit, and CPU timing)
- `data/asl_dataset/processed_spatial_tokens/` and raw images (for zero-shot transfer)
- `data/nus/processed_spatial_tokens/` (for pair-grouped NUS training)

Run any script with `--help` to inspect all path overrides and options. Landmark CSV rows and visual `.npy` arrays must use identical sample ordering.

## Experiment entry points (indexed by study ID)

| Study ID | Protocol / training entry point | Verification / recomputation entry point | Expected result directory |
|---|---|---|---|
| `fair_e1_e7` | `scg_vtnet/baselines/main.py` (`--study_name fair_e1_e7`) | `scg_vtnet/audit/verify_main.py` | `results/historical/fair_e1_e7`; `results/asl_baselines` |
| `visual_modulation` | `scg_vtnet/core/main.py` (`--study_name visual_modulation`) | `scg_vtnet/core/visualizations/plot_fig3_token_vs_global_corruption.py`; `export_fig6_and_confusions.py` | `results/historical/visual_modulation` |
| `core_ablation` | `scg_vtnet/core/main.py` (`--study_name core_ablation`) | `results/asl_baselines/full_model_raw_refined_seed_diagnostics.csv` | `results/historical/core_ablation` |
| `aar_construction` | `scg_vtnet/core/main.py` (`--study_name aar_construction`) | `scg_vtnet/core/visualizations/plot_fig1_aar_descriptor.py` | `results/historical/aar_construction` |
| `objective_construction` | `scg_vtnet/core/main.py` (`--study_name objective_construction`) | — | `results/historical/objective_construction` |
| `three_dataset_runs` | `scg_vtnet/extraction/extract_spatial_tokens_datasets.py`; `scg_vtnet/core/main.py` (`--study_name sota_only`) | — | `results/historical/three_dataset_runs` |
| `nus_grouped` | `scg_vtnet/protocol/run_nus_strict.py` | `scg_vtnet/audit/verify_grouped_nus.py` | `results/nus_grouped` |
| `asl_detection_strata` | `scg_vtnet/protocol/summarize_asl_detection.py` | `results/asl_baselines/seed42_predictions` | `results/asl_baselines/seed42_predictions` |
| `asl_paired_degradation` | `scg_vtnet/protocol/asl_experiment_a_validation.py` | `scg_vtnet/audit/recompute_paired_a.py`; `scg_vtnet/protocol/summarize_asl_pairs.py`; `export_asl_case_panels.py` | `results/asl_paired_degradation` |
| `asl_matched` | `scg_vtnet/protocol/run_asl_matched.py` | `scg_vtnet/protocol/summarize_asl_matched.py` | `results/asl_matched` |
| `asl_matched_zeroing` (Exp A) | `scg_vtnet/protocol/evaluate_asl_matched_zeroing.py` | `scg_vtnet/audit/verify_asl_matched_zeroing.py` | `results/asl_matched_zeroing` |
| `asl_natural_miss` (Exp B) | Derived from `asl_matched` clean predictions | `scg_vtnet/audit/summarize_asl_natural_miss.py` | `results/asl_natural_miss` |
| `asl_blocked` (Exp C) | `scg_vtnet/protocol/run_asl_blocked.py` | `scg_vtnet/audit/verify_asl_blocked.py` | `results/asl_blocked` |
| `asl_zero_shot` (Exp D) | `scg_vtnet/protocol/evaluate_asl_zero_shot.py` | `scg_vtnet/audit/verify_asl_zero_shot.py` | `results/asl_zero_shot` |
| `r14_structured_swap` | `scg_vtnet/protocol/evaluate_structured_swap.py` | `scg_vtnet/audit/verify_structured_swap.py` | `results/reviewer11_14/r14_structured_swap` |
| `r11_original_cases` | `scg_vtnet/core/visualizations/render_original_cases.py` | `results/reviewer11_14/SHA256SUMS.txt` | `results/reviewer11_14/r11_original_cases` |
| `publication_figures` | `scg_vtnet/core/visualizations/render_v10_figures.py` | Source CSVs exported alongside PDF/PNG figures | `results/historical/visual_modulation`; `results/asl_paired_degradation`; `results/reviewer11_14` |
| `cpu_timing` | `scg_vtnet/audit/benchmark_end_to_end_cpu.py`; `benchmark_heads_cpu.py` | `results/cpu_timing/cpu_end_to_end_benchmark_test29.json` | `results/cpu_timing` |

Historical batches (`fair_e1_e7`, `visual_modulation`, `core_ablation`, `aar_construction`, `objective_construction`, and `three_dataset_runs`) were produced with earlier repository revisions; their run configurations, logs, and checkpoints are preserved verbatim in the research archive. Legacy checkpoint identifiers such as `E7_AC-GCAT` correspond to saved historical artifacts.

## Reproducing the revision evaluations

```bash
# Matched 3-seed ASL ablation (F, A0, G0, Uniform)
python scg_vtnet/protocol/run_asl_matched.py --output results/asl_matched_run

# Exp A: Matched ASL fixed-visual skeleton zeroing (12 checkpoints)
python scg_vtnet/protocol/evaluate_asl_matched_zeroing.py --output results/asl_matched_zeroing_run

# Exp C: Contiguous index-block ASL split (10-frame buffer/side) and retraining (F, Uniform, E2, E5)
python scg_vtnet/protocol/run_asl_blocked.py --output results/asl_blocked_run --raw-images data/asl_alphabet/images

# Exp D: Zero-shot cross-dataset transfer (ASL Alphabet -> ASL Dataset, 26 shared letters)
python scg_vtnet/protocol/evaluate_asl_zero_shot.py --output results/asl_zero_shot_run --target-dir data/asl_dataset/processed_spatial_tokens --source-runs results/historical/fair_e1_e7 --raw-images data/asl_dataset/images

# Pair-grouped NUS evaluation
python scg_vtnet/protocol/run_nus_strict.py --processed-dir data/nus/processed_spatial_tokens --split-manifest manifests/nus.csv --output-dir results/nus_grouped_run

# Paired synthetic image degradation and fixed-visual skeleton interventions
python scg_vtnet/protocol/asl_experiment_a_validation.py --paired-test --per-class 10 --output results/asl_degradation_run

# Structured index-middle landmark correspondence swap (matched F vs A0)
python scg_vtnet/protocol/evaluate_structured_swap.py --output results/structured_swap_run

# Render screened unmodified challenging cases and multi-panel publication figures
python scg_vtnet/core/visualizations/render_original_cases.py --cases-dir results/reviewer11_14/r11_original_cases --output-dir results/original_cases_fig
python scg_vtnet/core/visualizations/render_v10_figures.py --project . --processed data/asl_alphabet/processed_train --output results/publication_figures --languages en
```

## Recomputing metrics from saved predictions

```bash
python scg_vtnet/audit/recompute_paired_a.py results/asl_paired_degradation/predictions.csv verification/asl_paired.csv
python scg_vtnet/protocol/summarize_asl_matched.py results/asl_matched verification/asl_matched
python scg_vtnet/audit/verify_grouped_nus.py results/nus_grouped manifests/nus.csv
python scg_vtnet/audit/verify_asl_matched_zeroing.py results/asl_matched_zeroing --matched-dir results/asl_matched
python scg_vtnet/audit/summarize_asl_natural_miss.py results/asl_matched --verify results/asl_natural_miss/per_seed.csv
python scg_vtnet/audit/verify_asl_blocked.py results/asl_blocked --verify-existing
python scg_vtnet/audit/verify_asl_zero_shot.py results/asl_zero_shot --source-runs results/historical/fair_e1_e7 --verify-existing
python scg_vtnet/audit/verify_structured_swap.py results/reviewer11_14/r14_structured_swap --matched-seed42-dir results/asl_matched/seed_42 --verify-existing
```

## License

Released under the [MIT License](LICENSE).
