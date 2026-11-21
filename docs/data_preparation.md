# Data preparation

Use a workspace containing the SCG-VTNet checkout and these directories. Set
`SCG_VTNET_WORKSPACE` when the workspace is elsewhere. Existing datasets may
be linked into these locations.

| Path relative to the workspace | Inputs |
| --- | --- |
| `data/raw/asl_alphabet/` | ASL Alphabet training images with class folders |
| `data/processed/asl_alphabet/` | Landmark CSV, ResNet-50 spatial/global features, EfficientNet-B0 and ViT-S/16 features |
| `data/splits/asl_alphabet/` | Frozen split CSVs and their freeze metadata |
| `data/splits/nus_hand_posture/nus.csv` | Strict paired NUS split |
| `runs/historical/` | Historical checkpoints, model configuration, scalers and label encoders |
| `internal/ablations/` | Author-side run logs, intermediate predictions and weights |

The ASL inputs use the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
The processed NUS inputs originate from the [NUS Hand Posture Dataset](https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/).
The exact ASL source download version has not been established. To reproduce the
audited split, obtain the frozen manifests from the authors and verify their
checksums with `python -m scripts.data.verify_splits ../data/splits`.
The split indices, landmark CSV rows and feature arrays must use the same sample
order. Run the extractors with `--help` for file, backbone and output options.
