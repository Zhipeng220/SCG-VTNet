# SCG-VTNet

SCG-VTNet recognizes static hand gestures from anatomy-aware hand landmarks and spatial visual features.

- `scg_vtnet/` contains the models, data processing, training and evaluation modules.
- `scripts/` contains entry points for data preparation, experiments, analysis and timing.
- `configs/` records the experimental protocols and frozen split checksums.
- `docs/` describes the inputs and commands used to reproduce the reported results.

## Installation

Use Python 3.10 for the original experiment environment. Install the PyTorch build for your CPU or CUDA environment, then install the remaining requirements and the package from the repository root.

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

See [data preparation](docs/data_preparation.md) and the [reproduction guide](docs/reproduction.md). Run the commands from the repository root. Input datasets and released artifacts are organized in a separate workspace.
