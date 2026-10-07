# Explainable CAD-risk classification from echocardiography

Research code built on [EchoNet-Dynamic](https://github.com/echonet/dynamic) for
binary classification of echocardiographic videos using a single sampled frame
and a ResNet-18 model. The pipeline was developed for the accompanying
research paper and includes class-balanced training and Grad-CAM
explainability.

> **Research-use notice:** `CAD-risk` is an experimental label derived from
> ejection fraction thresholds (`EF >= 50` for Normal and `EF < 40` for
> CAD-risk). It is a proxy label, not a clinically validated diagnosis of
> coronary artery disease. This repository is not medical advice or a
> clinical device.

## What is included

- EF-threshold-based dataset preparation using the existing EchoNet-Dynamic
  train/validation/test split.
- ResNet-18 frame classifier with ImageNet initialization.
- Weighted random sampling and class-weighted cross-entropy to address
  imbalance.
- Model selection by validation F1 score.
- Test accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and training
  history outputs.
- Separate scripts for Grad-CAM and ROC/evaluation visualizations.

## Repository layout

```text
.
├── cad/
│   ├── prepare_data.py       # Create CAD_FileList.csv from EchoNet metadata
│   ├── train_balanced.py     # Train and evaluate the balanced classifier
│   ├── gradcam.py            # Grad-CAM analysis
│   └── evaluation/           # ROC curves and comparison plots
├── echonet/                  # Upstream EchoNet-Dynamic package
├── scripts/                  # Upstream utilities and notebooks
├── data/                     # Local data; ignored by Git
├── outputs/                  # Checkpoints and figures; ignored by Git
├── requirements.txt
└── setup.py
```

The EchoNet-Dynamic videos and metadata are not included. Obtain them from the
[official dataset page](https://echonet.github.io/dynamic/) and comply with
its data-use agreement.

## Installation

Python 3.8+ is recommended. Install a compatible PyTorch build for your
operating system (CPU or CUDA) first, then install the project dependencies:

```bash
git clone https://github.com/<your-account>/<your-repository>.git
cd <your-repository>
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pinned historical dependencies in `requirements.txt` may need adjustment
for a newer Python/PyTorch installation. Keep the PyTorch and torchvision
versions compatible with each other.

## Expected data layout

Place the downloaded EchoNet-Dynamic data under `data/echonet/`:

```text
data/
└── echonet/
    └── a4c-video-dir/
        ├── FileList.csv
        └── Videos/
            ├── <video>.avi
            └── ...
```

`FileList.csv` must contain the EchoNet-Dynamic `FileName`, `EF`, and `Split`
columns. The preparation step excludes borderline cases with `40 <= EF < 50`.

## Reproducing the CAD-risk experiment

### 1. Prepare labels

```bash
python cad/prepare_data.py \
  --data-dir data/echonet \
  --output-dir data/cad
```

This writes `data/cad/CAD_FileList.csv` and reports the class distribution in
each predefined split.

### 2. Train and evaluate

```bash
python cad/train_balanced.py \
  --csv data/cad/CAD_FileList.csv \
  --video-dir data/echonet/a4c-video-dir/Videos \
  --output-dir outputs/cad-balanced \
  --epochs 10 \
  --batch-size 16 \
  --frame-idx 0
```

The default command uses CUDA when available and otherwise falls back to CPU.
The output directory contains:

```text
outputs/cad-balanced/
├── best_cad_model_balanced.pt
├── train.csv
├── val.csv
├── test.csv
├── test_metrics_balanced.csv
├── confusion_matrix_balanced.png
└── training_history_balanced.png
```

Unreadable videos are recorded in `unreadable_videos.log` and skipped by the
data loader. Review this file before interpreting results.

## Explainability and additional analyses

After training, use `cad/gradcam.py` for Grad-CAM visualizations and the
scripts under `cad/evaluation/` for ROC curves and comparison figures. Check
each script's `--help` output and source-level path arguments before running
it; these analyses require the trained checkpoint and the same video/metadata
layout.

## Reproducibility and limitations

The current experiment samples one frame per video (`--frame-idx 0`) and does
not claim temporal modeling. Results depend on the exact dataset version,
preprocessing, PyTorch/torchvision versions, random state, hardware, and
unreadable-video filtering. For a publication-quality rerun, record those
details together with the generated metrics and checkpoint.

This code has not been validated for prospective clinical use. Do not use its
predictions for patient care.

## Citation and attribution

This work extends the publicly released EchoNet-Dynamic implementation and
dataset. Please cite the original work:

> Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat
> assessment of cardiac function. *Nature* (2020).
> https://doi.org/10.1038/s41586-020-2145-8

Please also cite the accompanying research paper for this CAD-risk
classification and Grad-CAM pipeline when using this repository.

## License

Review the upstream EchoNet-Dynamic license and the terms of the dataset
agreement before redistributing or using this code. Add the final project
license and the paper citation here before making the repository public.
