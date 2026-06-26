# CD-Mamba

**Task:** Cloud detection (segmentation)  
**Paper:** arXiv 2509.04729  
**Framework:** PyTorch 2.1.0 + custom Mamba state-space extensions  
**Dataset:** Landsat-8 Biome (42,873 patches, 4-fold cross-validation)

---

## Model Overview

CD-Mamba is a cloud detection network based on the Mamba state-space model architecture. It produces binary cloud/clear segmentation masks from multi-spectral Landsat-8 imagery. Four pretrained checkpoints are provided, one per cross-validation fold:

| Checkpoint | Fold | Segmentation F1 | IoU |
|---|---|---|---|
| `cdm_01_0.82835.pth` | Fold 1 | — | — |
| `cdm_04_0.86710.pth` | Fold 4 | — | — |
| `cdm_07_0.89102.pth` | Fold 7 | — | — |
| `cdm_10_0.89449.pth` | Fold 10 | — | — |
| **Average across folds** | | **0.9323** | **0.8732** |

**Model size:** 111,027 parameters, 0.42 MB  
**Efficiency (Iris V100):** 76.25 ms avg latency, 13.12 FPS, 80.56 W, 6.143 J/inference  
**Cross-platform latency ratio (Iris/Jetson):** 2.65× — best among all models evaluated

### Obtaining the Pretrained Checkpoints

The pretrained checkpoints were not available in the public repository at the time this thesis was conducted. They were obtained by contacting the authors directly (Prof. Kun Zhan, Lanzhou University). The authors subsequently uploaded the checkpoints to the CD-Mamba GitHub repository on May 18, 2026. This exchange is documented in the thesis appendix.

If the checkpoints are now available in the repository, download them from:  
https://github.com/[cd-mamba-repo]/tree/main/pt_models

If not, contact the authors via the email listed in the paper (arXiv 2509.04729).

Store the checkpoints at:
```
CDMamba/pt_models/
    cdm_01_0.82835.pth
    cdm_04_0.86710.pth
    cdm_07_0.89102.pth
    cdm_10_0.89449.pth
```

---

## Platform Independence of Segmentation Metrics

A key finding of this thesis: segmentation quality metrics (F1, IoU, Accuracy, Precision, Recall) are **hardware-independent** when inference runs in float32 precision on both platforms. The CD-Mamba results on the Jetson Orin Nano were identical to the Iris results to four decimal places. This confirms that metrics are a property of the model and data, not of the execution hardware, under IEEE 754 single precision. A difference would only be expected if the Jetson were run in FP16 mode.

---

## Environment Setup

### Iris Cluster

The main challenge is that both `causal-conv1d` and `mamba-ssm` must be compiled from source, with several pinned version constraints.

```bash
conda create -n cdmamba python=3.10 -y
conda activate cdmamba

# PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# NVCC inside the conda environment (Lmod is inaccessible from login node)
# NOTE: use the 'nvidia' channel, not 'conda-forge' (cuda-toolkit=11.8 is not on conda-forge)
conda install -c nvidia cuda-nvcc=11.8 -y

# Version pins required before building the Mamba extensions
pip install setuptools==69.5.1   # must be < 70; version 70+ breaks mamba-ssm build system
pip install "numpy<2"

# --no-build-isolation is required: the build process needs access to the installed
# PyTorch headers, which the default isolated build environment cannot find.
pip install causal-conv1d==1.1.1 --no-build-isolation
pip install mamba-ssm==1.1.1     --no-build-isolation

# transformers 5.x is incompatible with mamba-ssm 1.1.1
pip install transformers==4.40.1

# Replace standard mamba-ssm with the authors' custom version (required by CD-Mamba)
# First, clone the repository and follow the authors' instructions for their custom mamba-ssm
# Clone the working environment as a backup BEFORE this replacement:
conda create --name cdmamba_backup --clone cdmamba
# Then install the custom mamba-ssm in cdmamba
```

**Important:** On Iris, the V100 GPU uses SM 70, which is already in the default architecture list of both `causal-conv1d` and `mamba-ssm`. **No patching of `setup.py` is required on Iris.** Patching is only needed on the Jetson (SM 87).

See `environment/requirements_cdmamba_iris.txt`.

### Jetson Orin Nano

CD-Mamba was the most difficult deployment of the four models. The Mamba extensions assume recent server-grade GPU architectures (SM 80, 86, 90) and do not include SM 87 (Jetson Orin Nano). Both must be compiled from source on the device, which is constrained by 8 GB of unified memory.

**Step 1: Prerequisites**

```bash
# NVCC from CUDA 12.6 is provided by JetPack; add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # should show V12.6.xx

# Add 8 GB swap file (CRITICAL: without it, mamba-ssm compilation OOMs and reboots the device)
sudo fallocate -l 8G /swapfile_extra
sudo chmod 600 /swapfile_extra
sudo mkswap /swapfile_extra
sudo swapon /swapfile_extra
```

**Step 2: NVIDIA aarch64 PyTorch wheel**

```bash
conda create -n cdmamba python=3.10 -y
conda activate cdmamba

pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl
```

**Step 3: causal-conv1d — clone from GitHub (PyPI tarball is missing CUDA source files)**

```bash
# The PyPI tarball for causal-conv1d 1.1.1 is missing CUDA source files.
# A normal 'pip install causal-conv1d==1.1.1' will fail immediately.
# Clone from GitHub instead:
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.1

# Apply the SM 87 patch (see patches/causal_conv1d_sm87.patch)
patch -p1 < ../patches/causal_conv1d_sm87.patch

# Build and install
pip install -e . --no-build-isolation
cd ..
```

The patch adds `'87'` to the architecture list in `setup.py`:
```python
# Before
for arch in ['60', '70', '75', '80', '86']:
    cc_flag.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

# After (added '87' for Orin Nano)
for arch in ['60', '70', '75', '80', '86', '87']:
    cc_flag.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
```

**Step 4: mamba-ssm — SM 87 patch + swap + MAX_JOBS=1**

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.1

# Apply the SM 87 patch
patch -p1 < ../patches/mamba_ssm_sm87.patch

# MAX_JOBS=1 serialises compilation to reduce peak memory pressure
# Without this, the build may OOM even with the swap file
MAX_JOBS=1 pip install -e . --no-build-isolation
cd ..
```

**Step 5: Remaining dependencies**

```bash
pip install setuptools==69.5.1
pip install "numpy<2"
pip install transformers==4.40.1

# torchvision: compile from source (takes ~40 min)
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd vision && MAX_JOBS=1 python setup.py install && cd ..
```

**Step 6: torchvision stub (if torchvision C extension fails to load)**

On the Jetson, the compiled torchvision C extension may fail at runtime with:
```
Operator not found: torchvision::nms on SM 87
```

CD-Mamba only uses torchvision indirectly through `timm`, and `timm` only needs a few symbols (`trunc_normal_`, `DropPath`, `to_2tuple`). A minimal stub package provides those symbols using pure PyTorch equivalents, bypassing the broken C extension:

```bash
# The stub lives at ~/timm_stub/ and is added to PYTHONPATH ahead of real torchvision
# See Jetson_Orin_Nano/scripts/ for the stub implementation
export PYTHONPATH=~/timm_stub:$PYTHONPATH
```

**Step 7: Avoid pip overwriting the NVIDIA PyTorch wheel**

During dependency resolution, pip may replace the carefully-installed NVIDIA PyTorch wheel with the standard PyPI version, silently breaking CUDA support. If this happens:

```bash
pip uninstall torch  # remove the PyPI version
pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl  # reinstall NVIDIA wheel
```

See `environment/requirements_cdmamba_jetson.txt` for the full list and `patches/` for the exact diffs.

---

## HPC Iris — Evaluation

All four checkpoints are evaluated sequentially in a single SLURM job. Per-fold segmentation metrics are written to a single JSON output file.

```bash
sbatch HPC_Iris/scripts/slurm_eval_cdmamba.sh
```

Results: `HPC_Iris/results/cdmamba_metrics_20260531_065920.json`

---

## Jetson Orin Nano — Evaluation

Two separate scripts:
1. `eval_cdmamba_jetson_efficiency.py` — efficiency metrics (latency, FPS, power, energy) using dummy inputs
2. `eval_cdmamba_jetson_segmentation.py` — segmentation metrics on the Biome patches (verifies platform independence)

Run directly from the terminal (no SLURM on the Jetson):

```bash
conda activate cdmamba
export PYTHONPATH=~/timm_stub:$PYTHONPATH

cd ~/Desktop/CDMamba/
python eval_cdmamba_jetson_efficiency.py
python eval_cdmamba_jetson_segmentation.py
```

---

## Known Issues Summary

| # | Component | Issue | Resolution |
|---|---|---|---|
| 1 | `causal-conv1d` | PyPI tarball missing CUDA source files | Clone from GitHub |
| 2 | `causal-conv1d` | Hardcoded CUDA archs (no sm_87) | Patch `setup.py` |
| 3 | `mamba-ssm` | OOM crash during compile | 8 GB swap + MAX_JOBS=1 |
| 4 | `mamba-ssm` | Hardcoded CUDA archs (no sm_87) | Patch `setup.py` |
| 5 | `transformers` | 5.9.0 incompatible with mamba-ssm 1.1.1 | Downgrade to 4.40.1 |
| 6 | PyTorch | pip overwrote NVIDIA wheel during dep resolution | Reinstall NVIDIA wheel |
| 7 | `torchvision` | No prebuilt wheel for aarch64 | Compile from source |
| 8 | `torchvision` | C extension broken on Orin (nms missing) | Bypass via timm stub |
| 9 | `timm` | Requires broken torchvision | Create minimal stub package |
