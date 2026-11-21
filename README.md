# SCG-VTNet

SCG-VTNet performs single-frame static hand gesture recognition using anatomy-aware skeleton features and a spatial visual feature grid. This repository contains the model, experiment entry points, and analysis scripts.

## Layout

| Directory | Contents |
|---|---|
| scg_vtnet/core | Model, anatomical descriptors and training objectives |
| scg_vtnet/baselines | Historical baselines and matched ASL configurations |
| scg_vtnet/extraction | MediaPipe landmarks and visual feature extraction |
| scg_vtnet/protocol | ASL degradation, matched ASL training and grouped NUS training |
| scg_vtnet/audit | Prediction verification, statistics and CPU timing |

## Environment and inputs

Install PyTorch and torchvision for your CPU or CUDA environment, and install the remaining dependencies with:

    python -m pip install -r scg_vtnet/core/requirements.txt

For feature extraction and image evaluation, also install MediaPipe, OpenCV, Pillow and tqdm. Use the feature extractor's help to supply the local image and output paths. Landmark rows and visual arrays must use the same sample ordering.

The example commands run from the release root. Place local frozen splits in manifests/asl and manifests/nus.csv. The ASL training launcher expects the landmark CSV and spatial tokens in data/asl_alphabet/processed_train. The image degradation and CPU timing scripts also use source images from data/asl_alphabet/images and corresponding historical checkpoints from results/historical/fair_e1_e7/seed_42. Paths to other experiment inputs are shown in the relevant command help.

## Reproduce the evaluations

    python scg_vtnet/protocol/run_asl_matched.py --output results/asl_matched_run
    python scg_vtnet/protocol/run_nus_strict.py --processed-dir data/nus/processed_spatial_tokens --split-manifest manifests/nus.csv --output-dir results/nus_grouped_run
    python scg_vtnet/protocol/asl_experiment_a_validation.py --paired-test --per-class 10 --output results/asl_degradation_run

To recompute metrics from saved results:

    python scg_vtnet/audit/recompute_paired_a.py results/asl_paired_degradation/predictions.csv verification/asl_paired.csv
    python scg_vtnet/protocol/summarize_asl_matched.py results/asl_matched verification/asl_matched
    python scg_vtnet/audit/verify_grouped_nus.py results/nus_grouped manifests/nus.csv

Numerical seeds identify the original sampling, training or resampling protocol. Checkpoint identifiers such as E7_AC-GCAT identify saved experiment artifacts. The repository contains source code. Dataset inputs, split manifests, predictions and checkpoints are stored separately.
