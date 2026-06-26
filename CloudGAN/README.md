# CloudGAN

**Task:** Cloud detection (AE, U-Net) and cloud removal (SN-PatchGAN)  
**Original repository:** https://github.com/JerrySchonenberg/CloudGAN  
**Framework:** TensorFlow 1.15 / Python 3.6.13  
**Dataset:** 38-Cloud (detection), RICE-I/II (removal)

---

## Model Overview

CloudGAN comprises three sub-models:

| Sub-model | Task | Parameters | Size | Checkpoint |
|---|---|---|---|---|
| Autoencoder (AE) | Cloud detection | 198,593 | 0.78 MB | `ae_checkpoint.h5` (trained from scratch) |
| U-Net | Cloud detection | 1,941,381 | 7.45 MB | `unet_checkpoint.h5` (trained from scratch) |
| SN-PatchGAN | Cloud removal | 4,052,478 | 119.57 MB | `snap-1132000` (pretrained, authors) |

The AE and U-Net were trained from scratch on the 38-Cloud dataset on the Iris cluster. Training the SN-PatchGAN from scratch is not supported by the published repository (the authors modified the code after publication); the five pretrained checkpoints provided in the repo (`snap-1116000` through `snap-1132000`) must be used directly. Checkpoint `snap-1132000` (most recent) was used for all evaluations.

---

## Environment Setup

### Iris Cluster

The main compatibility challenge is that CloudGAN targets TensorFlow 1.15 and Python 3.6.13, both end-of-life. The cluster's system CUDA (11.1) is incompatible with TF 1.15, which requires CUDA 10.0. The solution is to install CUDA 10.0 and cuDNN 7.6 inside the conda prefix via `conda-forge`, so TensorFlow uses the environment-local CUDA rather than the system one.

```bash
conda create -n cloudgan python=3.6.13 -y
conda activate cloudgan

pip install tensorflow-gpu==1.15
pip install numpy==1.19.5 h5py==2.10.0 scikit-image tqdm PyYAML
pip install opencv-python==4.5.5.64   # newer versions have no pre-built wheel for Python 3.6
pip install matplotlib scipy           # missing from the repo's requirements.txt but imported
pip install pynvml                     # for GPU power measurement on V100

# NeuralGym: training utilities for SN-PatchGAN, pinned to a specific commit for reproducibility
pip install git+https://github.com/JiahuiYu/neuralgym@88292adb524186693a32404c0cfdc790426ea441

# CUDA 10.0 + cuDNN 7.6 inside conda (bypasses system CUDA 11.1)
conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6 -y
```

See `environment/requirements_cloudgan.txt` for the full pinned package list.

**Issues encountered and fixes:**
- `opencv-python`: no pre-built wheel for Python 3.6 beyond 4.5.5.64 — pin to exactly that version.
- `matplotlib` and `scipy`: missing from the repo's `requirements.txt` but imported by utilities — add manually.
- System CUDA 11.1 incompatible with TF 1.15: solved by installing CUDA 10.0 from `conda-forge` inside the environment prefix.
- `scikit-image` (`skimage`) was also missing from the environment when running the quality evaluation scripts — install separately if needed.
- Conda `shell.bash hook` path: the SLURM scripts reference Miniconda at `/scratch/users/jfernandezmartinez/miniconda3/`, not `~/miniconda3/`. Always use the full path in SLURM scripts.

### Jetson Orin Nano

There is no pre-built TensorFlow 1.x wheel for aarch64. Compiling TF 1.x from source against CUDA 12 is not feasible. The solution is to run TF 2.16.1 (the NVIDIA-provided JetPack 6.1 wheel) with a TF1→TF2 compatibility shim that re-exposes the TF 1.x API symbols used by CloudGAN.

```bash
conda create -n cloudgan_jetson python=3.10 -y
conda activate cloudgan_jetson

# NVIDIA TF2 wheel for JetPack 6.1
pip install tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl

pip install numpy==1.24.0
pip install h5py==2.10.0          # CRITICAL: h5py ≥ 3.x breaks model.load_weights() for HDF5 checkpoints
pip install opencv-python==4.9.0.80
pip install scikit-image tqdm PyYAML matplotlib scipy pynvml

# Fix CXXABI conflict between L4T 36.4.0 libstdc++ and NumPy/OpenCV binaries
conda install -c conda-forge libstdcxx-ng -y
```

The TF1→TF2 shim (`tf1_compat_shim.py`, integrated into all Jetson evaluation scripts) patches the `tensorflow` namespace before CloudGAN imports run. It re-exposes:
- `tf.contrib` → `tf.compat.v1.keras`
- `tf.layers` → `tf.compat.v1.layers`
- `tf.py_func` → `tf.compat.v1.py_func`
- `keep_dims` keyword → `keepdims`
- Image resize functions and other removed API elements
- Disables eager execution where the original code expects a session-based graph

**Issues encountered and fixes:**
- `h5py ≥ 3.x`: breaks `model.load_weights()` for HDF5 checkpoints because string decoding changed (bytes vs str). Error: `AttributeError: 'str' object has no attribute 'decode'`. Fix: pin `h5py==2.10.0`.
- `nvidia-smi` calls in the original evaluation scripts: `nvidia-smi` does not exist on Tegra. Replace with `TegraStatsMonitor`.
- CXXABI conflict in L4T 36.4.0 `libstdc++`: install `libstdcxx-ng` from `conda-forge`.

---

## HPC Iris — Training

Both AE and U-Net were trained from scratch. Training takes under 30 minutes per model on a single V100.

**AE training (job 5329871, node iris-193):**
- Loss decreased from ~0.246 to ~0.089 over 64 epochs
- Checkpoint saved: `output/AE/ae_checkpoint.h5` (798 KB)

**U-Net training (job 5348654, node iris-171):**
- Trained with `--model UNET` (uppercase — the argument parser is case-sensitive; `UNet` fails)
- Checkpoint saved: `output/UNet/unet_checkpoint.h5`

See `HPC_Iris/scripts/train_AE.sh` and `HPC_Iris/scripts/train_UNet.sh`.

---

## HPC Iris — Evaluation

**AE and U-Net FLOPs / graph issue:**  
The AE and U-Net FLOPs profiler called `reset_default_graph()` after the model was already loaded, causing a graph mismatch. Fix: build the model inside an isolated `tf.Graph()` context for profiling, then reload the checkpoint in the main graph for inference.

A second issue involved `tf.compat.v1.keras.backend.set_session` corrupting the global Keras session state. Fix: remove that call and rely solely on `global_variables_initializer()`.

Both fixes are integrated into `eval_efficiency_AE.py` and `eval_efficiency_UNet.py`.

Scripts: `HPC_Iris/scripts/`  
Results: `HPC_Iris/results/`

---

## Jetson Orin Nano — Evaluation

The Jetson evaluation measures efficiency metrics only (latency, FPS, power, energy). Quality metrics (F1, IoU) are hardware-independent and were obtained on Iris.

All `nvidia-smi` / `pynvml` calls are replaced by `TegraStatsMonitor` (integrated in the scripts).

Scripts: `Jetson_Orin_Nano/scripts/`  
Results: `Jetson_Orin_Nano/results/`

---

## Results Summary

| Sub-model | Platform | Latency (ms) | FPS | Power (W) | Energy/inf (J) | F1 | IoU |
|---|---|---|---|---|---|---|---|
| AE | Iris V100 | 6.16 | 162.0 | 66.94 | 0.413 | 0.6752 | 0.5791 |
| AE | Jetson | 191.75 | 5.22 | 1.66 | 0.318 | — | — |
| U-Net | Iris V100 | 8.09 | 123.63 | — | — | 0.7413 | 0.6285 |
| U-Net | Jetson | 194.93 | 5.13 | — | — | — | — |
| SN-PatchGAN | Iris V100 | 12.81 | 78.09 | 64.42 | 0.825 | — | — |
| SN-PatchGAN | Jetson | 80.18 | 12.47 | — | 0.458 | — | — |

SN-PatchGAN removal quality on RICE (4 example pairs): PSNR 20.03 dB, SSIM 0.493.
