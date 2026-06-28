# Onboard Edge-AI for Cloud Detection and Removal in Earth Observation Satellites

**Master's Thesis Repository**  


---

## Overview

This repository contains all scripts, SLURM job files, environment specifications, deployment notes, and raw results produced during the thesis *"On-board Cloud Detection and Removal Study for Edge-AI-enabled EO Satellites"*. The thesis evaluates whether state-of-the-art deep learning models for cloud detection and removal can be deployed on edge-AI hardware representative of future on-board satellite processors, with emphasis on inference latency, throughput, and energy consumption.

Two compute platforms are compared throughout:

| Platform | Role | GPU | CUDA | Memory |
|---|---|---|---|---|
| UL-HPC Iris cluster | Reference (HPC) | NVIDIA Tesla V100-SXM2 | 11.x (env) | 32 GB HBM2 |
| NVIDIA Jetson Orin Nano | Edge target | Ampere, 1024 CUDA cores, sm_87 | 12.6 | 8 GB unified |

Four models are evaluated:

| Model | Task | Framework | Original repo |
|---|---|---|---|
| CloudGAN (AE, U-Net, SN-PatchGAN) | Cloud detection + removal | TensorFlow 1.15 | [CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) |
| DVPNet | Cloud removal | PyTorch (basicsr) | IEEE TGRS 2025 |
| CD-Mamba | Cloud detection | PyTorch + custom Mamba extensions | [arxiv 2509.04729](https://arxiv.org/abs/2509.04729) |
| OmniCloudMask v1.7.1 | Cloud detection | PyTorch (smp + timm) | [omnicloudmask](https://github.com/DPIRD-DMA/OmniCloudMask) |

The full thesis document (PDF) and the weekly progress report are included at the root of this repository. They contain detailed narrative for every step documented here — any script or configuration that looks cryptic will have its context explained in those documents.

---

## Repository Structure

```
Onboard-Edge-AI-Cloud-Detection-Removal/
├── README.md                                ← Main Project Description file
├── DATASETS.md                              ← Dataset descriptions, download links, and preparation commands
├── Weekly_Report.pdf                        ← Weekly progress report
│
├── CloudGAN/
│   ├── README.md
│   ├── environment/
│   │   └── requirements_cloudgan.txt
│   ├── HPC_Iris/
│   │   ├── scripts/                          ← Training + evaluation Python and SLURM scripts
|   |   |   └── AE
|   |   |   └── UNet/
|   |   |   └── SN-PatchGAN/
|   |   |   └── Uncertainity Filter/
│   │   └── results/                          ← Raw JSON/txt & visual output files
|   |       └── AE
|   |       └── UNet/
|   |       └── SN-PatchGAN/
|   |       └── Uncertainity Filter/
│   └── Jetson_Orin_Nano/
│       ├── scripts/
|       |   └── AE
|       |   └── UNet/
|       |   └── SN-PatchGAN/
│       └── results/
|           └── AE
|           └── UNet/
|           └── SN-PatchGAN/
│
├── DVPNet/
│   ├── README.md
│   ├── environment/
│   │   ├── requirements_dvpnet_iris.txt
│   │   └── requirements_dvpnet_jetson.txt
│   ├── HPC_Iris/
│   │   ├── scripts/
│   │   └── results/
│   └── Jetson_Orin_Nano/
│       ├── scripts/
│       └── results/
│
├── CD-Mamba/
│   ├── README.md
│   ├── environment/
│   │   ├── requirements_cdmamba_iris.txt
│   │   └── requirements_cdmamba_jetson.txt
│   ├── patches/                 ← Critical: sm_87 patches for causal-conv1d and mamba-ssm
│   │   ├── causal_conv1d_sm87.patch
│   │   └── mamba_ssm_sm87.patch
│   ├── HPC_Iris/
│   │   ├── scripts/
│   │   └── results/
│   └── Jetson_Orin_Nano/
│       ├── scripts/
│       └── results/
│
└── OmniCloudMask/
    ├── README.md
    ├── environment/
    │   ├── requirements_omnicloud_iris.txt
    │   └── requirements_omnicloud_jetson.txt
    ├── HPC_Iris/
    │   ├── scripts/
    │   └── results/
    └── Jetson_Orin_Nano/
        ├── scripts/
        └── results/
```

---

## Hardware Platforms

### UL-HPC Iris Cluster

The Iris cluster is the high-performance computing facility of the University of Luxembourg, operated by the HPC team. Access is managed through the IAM portal at [hpc.uni.lu](https://hpc.uni.lu). Connections use SSH on port 8022:

```bash
ssh -p 8022 <username>@access-iris.uni.lu
```

All jobs run through the SLURM scheduler. The relevant partition for this thesis is `gpu`, which provides nodes with four Tesla V100-SXM2 GPUs each. All jobs requested one GPU per job (`--gres=gpu:1`).

A self-contained Miniconda installation was used at `/scratch/users/jfernandezmartinez/miniconda3/` to manage all Python environments. The cluster's system CUDA (11.1) was never used directly; each environment manages its own CUDA libraries through conda channels.

**Account hierarchy note:** During this thesis, the user account was placed under `trainings/students` (fairshare ~0.029), which caused long queue times. The correct research account hierarchy for SnT projects is `snt/<supervisor>`.

### NVIDIA Jetson Orin Nano Developer Kit

| Specification | Value |
|---|---|
| SoC | NVIDIA Tegra Orin (T234) |
| CPU | 6-core Arm Cortex-A78AE @ 1.5 GHz |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor cores, sm_87 |
| Memory | 8 GB LPDDR5, unified CPU/GPU (no separate VRAM) |
| Power modes | 7 W / 15 W |
| OS | Ubuntu 22.04 (L4T 36.4.0) |
| JetPack | 6.1 |
| CUDA | 12.6 |

**Critical architectural differences from a workstation GPU:**

1. **Unified memory.** There is no separate VRAM. The 8 GB pool is shared between OS, CPU processes, and GPU. This makes heavy compilation jobs (e.g. mamba-ssm from source) prone to OOM crashes. An 8 GB swap file was added at `/swapfile_extra` before compiling CD-Mamba.

2. **No `nvidia-smi`.** Power and memory monitoring on Tegra uses `tegrastats` instead. A `TegraStatsMonitor` Python class was written to wrap `tegrastats`, parse its output, and expose per-rail mean power values. This class is used identically across all four model evaluations so that power numbers are directly comparable.

3. **ARM64 (aarch64) architecture.** Standard PyPI wheels for PyTorch do not work. The NVIDIA-provided wheel for JetPack 6.1 / Python 3.10 must be used:
   ```bash
   pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl
   ```

---

## Conda Environments Summary

### Iris Cluster

| Environment | Python | Framework | Models |
|---|---|---|---|
| `cloudgan` | 3.6.13 | TensorFlow-GPU 1.15, CUDA 10.0 (conda-forge) | CloudGAN AE, U-Net, SN-PatchGAN |
| `DVPNet` | 3.10 | PyTorch 1.13.1, CUDA 11.7 | DVPNet |
| `cdmamba` | 3.10 | PyTorch 2.1.0, CUDA 11.8, custom mamba-ssm | CD-Mamba |
| `GeoAI` | 3.12 | PyTorch (cu118), geoai-py | OmniCloudMask |

### Jetson Orin Nano

| Environment | Python | Framework | Models |
|---|---|---|---|
| `cloudgan_jetson` | 3.10 | TensorFlow 2.16.1 (NVIDIA wheel) + TF1 shim | CloudGAN AE, U-Net, SN-PatchGAN |
| `DVPNet` | 3.10 | PyTorch 2.5.0 (NVIDIA aarch64 wheel) | DVPNet |
| `cdmamba` | 3.10 | PyTorch 2.5.0 (NVIDIA aarch64 wheel), custom mamba-ssm | CD-Mamba |
| `omnicloud` | 3.10 | PyTorch 2.5.0 (NVIDIA aarch64 wheel), omnicloudmask 1.7.1 | OmniCloudMask |

---

## Power Measurement Methodology

Power measurement differs between the two platforms and is documented here once to avoid repetition in each model's README.

**Iris / V100:** `pynvml` polls `nvidia-smi` programmatically every 100 ms during the timed inference loop. The reported value is total board power (GPU die + HBM2).

```python
nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits
```

**Jetson Orin Nano:** `tegrastats` is launched as a subprocess with a 100 ms interval. Three power rails are parsed from each output line:

- `VDD_IN`: total SoC input power
- `VDD_CPU_GPU_CV`: CPU + GPU + CV island (primary metric for AI inference)
- `VDD_SOC`: auxiliary rail (memory controller, fabric, peripherals)

`VDD_CPU_GPU_CV` is used as the primary energy indicator throughout the thesis because it isolates the compute subsystem from platform noise. The `TegraStatsMonitor` context manager used in all Jetson scripts works as follows:

```python
with TegraStatsMonitor() as mon:
    for _ in range(100):
        _ = model(dummy_input)
power_w = mon.avg_power('VDD_CPU_GPU_CV')
energy_j = power_w * (latency_ms / 1000.0)
```

---

## Key Results Summary

For full results including segmentation metrics (F1, IoU) and image quality metrics (PSNR, SSIM), see the thesis PDF and `*/results/` folders.

---

## Citation

If you use the scripts, patches, or deployment methodology from this repository, please cite:

```bibtex
@mastersthesis{fernandez2026edgeai,
  author  = {Juan Fern\'{a}ndez Mart\'{i}nez},
  title   = {On-board Cloud Detection and Removal Study for Edge-{AI}-enabled {EO} Satellites},
  school  = {University of Luxembourg, SnT},
  year    = {2026},
  url     = {https://github.com/SeaGraphData/Onboard-Edge-AI-Cloud-Detection-Removal}
}
```

---

## References

- CloudGAN repository: https://github.com/JerrySchonenberg/CloudGAN
- NeuralGym (SN-PatchGAN training utilities): https://github.com/JiahuiYu/neuralgym
- CD-Mamba paper: https://arxiv.org/abs/2509.04729
- DVPNet paper: IEEE TGRS 2025
- OmniCloudMask: https://github.com/DPIRD-DMA/OmniCloudMask
- UL-HPC documentation: https://hpc.uni.lu/documentation
- NVIDIA Jetson Orin Nano Developer Guide: https://developer.nvidia.com/embedded/jetpack
- JetPack PyTorch wheels: https://developer.nvidia.com/embedded/jetpack-archive
