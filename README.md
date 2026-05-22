# CIFAR-10 Dynamic Robustness Benchmark

This repository studies adversarial robustness as a dynamic training-process
signal instead of only a final-checkpoint score. The project evaluates 18
CIFAR-10 image-classification models with checkpoint-wise clean accuracy,
FGSM robust accuracy, clean-robust gap, adaptive epsilon selection, and
path-dependent robustness phase analysis.

The single notebook entry point is:

```text
5329_Assignment2.ipynb
```

All implementation lives in Python modules under `Attack_Method/`,
`Data_Loader/`, `Evaluation/`, `Experiments/`, `Models/`, and `Training/`.

## Core Idea

The project does not redefine adversarial robustness. It converts the same
robustness metric from a static endpoint score into a checkpoint-indexed
trajectory:

```text
static robustness:  RA(T)
dynamic robustness: {RA(t_1), RA(t_2), ..., RA(t_K)}
```

The dynamic trajectory is interpreted with:

```text
CA(t): clean accuracy
RA(t): FGSM robust accuracy
Delta(t): CA(t) - RA(t)
```

The final phase analysis is path-dependent: a checkpoint is interpreted using
the trajectory that led to it, not only its current robust accuracy value.

## Model Registry

Models are registered in:

```text
Models/registry.py
```

The benchmark includes 18 models:

```text
vgg16_bn_cifar
vgg19_bn_cifar
alexnet_cifar
mobilenetv2_cifar
mlp_mixer_tiny
vit_tiny_cifar
resnet20
resnet32
resnet18_general
densenet121_cifar
mlp_mixer_small
vit_small_cifar
wrn_16_8
wrn_28_10
resnet50_general
efficientnet_b0_cifar
mlp_mixer_base
swin_tiny_cifar
```

The registry groups models by design type:

```text
CIFAR-friendly
General-adapted
MLP-adapted
Transformer-adapted
```

## Pipeline

The final experimental pipeline is:

```text
CIFAR-10 training
-> progress-based checkpoints
-> checkpoint-wise clean evaluation
-> checkpoint-specific raw-space FGSM evaluation
-> adaptive epsilon calibration
-> final model-specific dynamic robustness runs
-> path-dependent phase analysis
-> figures, descriptors, and diagnosis tables
```

Final path-dependent phases:

```text
Pre-learning
Robustness Formation
Robustness Maturation
Robustness Stabilization / Peak
Early Robustness Decay
Late Degradation
Late Divergence
Late Stabilization
```

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches your CUDA version
from the official PyTorch instructions before running the benchmark.

## Running the Project

Open and run:

```text
5329_Assignment2.ipynb
```

The notebook is the intended top-level orchestration file. It checks the
environment, lists registered models, and calls the adaptive robustness
pipeline.

The same pipeline can be launched directly from the terminal:

```powershell
python Experiments/run_adaptive_protocol.py --stage all --group all
```

Useful alternatives:

```powershell
python Experiments/run_adaptive_protocol.py --stage all --group cifar_friendly
python Experiments/run_adaptive_protocol.py --stage final --group all --resume
python Experiments/path_dependent_phase_analysis.py
```

## Outputs

Generated data, checkpoints, metrics, and figures are written to:

```text
Results/
```

Final adaptive-protocol outputs are organized under:

```text
Results/adaptive_protocol/calibration/
Results/adaptive_protocol/protocol/
Results/adaptive_protocol/final/
Results/adaptive_protocol/summary_figures/
Results/adaptive_protocol/path_dependent_phase/
```

The key final files are:

```text
Results/adaptive_protocol/path_dependent_phase/path_dependent_phase_timeline.png
Results/adaptive_protocol/path_dependent_phase/path_descriptors.csv
Results/adaptive_protocol/path_dependent_phase/path_phase_occupancy_summary.csv
Results/adaptive_protocol/path_dependent_phase/PATH_DEPENDENT_PHASE_SUMMARY.md
```

## Repository Hygiene

Large generated files are intentionally ignored:

```text
Data/
Results/
.venv/
.venv312/
.tmp/
```

Only one notebook is kept in the repository root: `5329_Assignment2.ipynb`.
