# OmniCloudMask

**Task:** Cloud detection (zero-shot, no training required)  
**Version:** v1.7.1  
**Library:** `geoai-py` / `omnicloudmask` (PyPI)  
**Repository:** https://github.com/DPIRD-DMA/OmniCloudMask  
**Dataset:** Landsat-8 Biome (42,873 patches, same split as CD-Mamba)

---

## Model Overview

OmniCloudMask is a pretrained cloud and cloud shadow segmentation library distributed as a standard PyPI package. It requires no training or fine-tuning; the pretrained weights are downloaded automatically from HuggingFace Hub on the first inference call.

**Architecture:** Ensemble of two `smp.Unet` models with different encoders:

| Encoder | Weights file | Size |
|---|---|---|
| `tu-edgenext_small` | `edgenext_small.pth` | 29.29 MB |
| `tu-regnety_004` | `regnety_004.pth` | 25.84 MB |
| **Total** | | **55.13 MB** |

The model produces a **four-class segmentation mask**:
- Class 0: Clear (no cloud)
- Class 1: Thick Cloud
- Class 2: Thin Cloud
- Class 3: Cloud Shadow

Because the Biome ground-truth masks are binary (cloud / no-cloud), a binarisation step is required before computing metrics. Two binarisation strategies are reported:

- **Shadow-excluded:** classes 1 and 2 → cloud; class 0 → clear; class 3 (shadow) ignored
- **Shadow-inclusive:** classes 1, 2, and 3 → cloud; class 0 → clear

**Input bands:** Red (Band 4), Green (Band 3), NIR (Band 5) — directly compatible with Landsat-8.

OmniCloudMask was selected as an addition to the evaluation suite because it represents the deployment-ready, library-based end of the spectrum: no training pipeline, no custom CUDA extensions, and a well-maintained PyPI package.

---

## Environment Setup

### Iris Cluster

OmniCloudMask was the simplest deployment of the four models. The main pitfall is that the default `conda` resolver may install a CPU-only PyTorch build; install PyTorch separately via the `cu118` index to ensure GPU support.

```bash
conda create -n GeoAI python=3.12 -y
conda activate GeoAI

# Use the Mamba solver for faster dependency resolution (geoai-py has a large dep graph)
conda install -c conda-forge mamba -y
mamba install -c conda-forge geoai-py -y

# Install OmniCloudMask — pinned to v1.7.1 for reproducibility
pip install omnicloudmask==1.7.1

# Ensure a CUDA-enabled PyTorch build (the conda resolver may have installed CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install pynvml   # GPU power measurement on V100
```

**Python 3.12 was chosen deliberately**: `geoai-py` is tested against it, and the environment was designed to serve future geospatial workloads beyond this thesis (ship detection, land cover segmentation).

**Weights location on Windows (where OmniCloudMask was first tested):**
```
C:\Users\<username>\AppData\Local\omnicloudmask\
    edgenext_small.pth   (29.29 MB)
    regnety_004.pth      (25.84 MB)
```
On Linux/Mac, HuggingFace Hub caches weights at `~/.cache/huggingface/hub/`.

See `environment/requirements_omnicloud_iris.txt`.

**Operational issues during large-scale evaluation:**

- **Walltime kill:** The quality evaluation over 42,873 patches was killed after processing 42,786 patches (87 short of completion) because the SLURM scheduler enforced the walltime more aggressively than expected. Fix: add a **resume mechanism** that checks at startup which patches already have results written and skips them on restart. Also extend walltime to 10 hours.
- **Faulty node:** Node `iris-186` failed silently during early runs. If you encounter silent failures on Iris, check which node your job was scheduled on with `squeue -u <username>` or the job output file header.

### Jetson Orin Nano

OmniCloudMask was the easiest Jetson deployment, by a wide margin. The library is a standard PyPI package with well-maintained dependencies, and most of the heavy lifting (environment setup, torchvision stub, tegrastats integration) had already been done for CD-Mamba.

```bash
conda create -n omnicloud python=3.10 -y
conda activate omnicloud

# NVIDIA aarch64 PyTorch wheel (same as CD-Mamba)
pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl

pip install omnicloudmask==1.7.1
pip install segmentation-models-pytorch timm

# The same timm stub used for CD-Mamba works here too
# (OmniCloudMask does not use torchvision at inference time)
export PYTHONPATH=~/timm_stub:$PYTHONPATH
```

The Jetson evaluation uses synthetic dummy inputs of shape `(1, 3, 384, 384)` — the full 42,873-patch Biome run is not repeated on the Jetson, since quality metrics are hardware-independent. Only efficiency metrics (latency, FPS, power, energy) are measured on the device.

See `environment/requirements_omnicloud_jetson.txt`.

---

## HPC Iris — Evaluation

Two SLURM jobs:

1. **Quality evaluation** (`slurm_eval_quality.sh`): Runs over all 42,873 Biome patches. Reports F1, Precision, Recall, Accuracy, IoU under both shadow-excluded and shadow-included binarisation. Includes the resume mechanism (skips already-processed patches on restart). Walltime: 10 hours.

2. **Efficiency evaluation** (`slurm_eval_efficiency.sh`): Dummy input `(1, 3, 384, 384)`. Reports latency, FPS, GPU memory, power (pynvml), energy per inference.

Scripts: `HPC_Iris/scripts/`  
Results: `HPC_Iris/results/`

---

## Jetson Orin Nano — Evaluation

Single script `omnicloud_jetson_efficiency.py` using the `TegraStatsMonitor` context manager. No dataset required.

Run directly from terminal:

```bash
conda activate omnicloud
export PYTHONPATH=~/timm_stub:$PYTHONPATH
cd ~/Desktop/OmniCloudMask/
python omnicloud_jetson_efficiency.py
```

Scripts: `Jetson_Orin_Nano/scripts/omnicloud_jetson_efficiency.py`  
Results: `Jetson_Orin_Nano/results/omnicloud_jetson_efficiency.json`

---

## Results Summary

| Binarisation | F1 | IoU | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Shadow-excluded | — | — | — | — | — |
| Shadow-inclusive | — | — | — | — | — |

For full results see `HPC_Iris/results/`.

| Platform | Latency (ms) | FPS | Power (W) | Energy/inf (J) |
|---|---|---|---|---|
| Iris V100 | — | — | — | — |
| Jetson | — | — | — | — |

(Fill in from your result JSON files.)
