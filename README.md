# On-board Cloud Detection and Removal for Edge-AI-enabled Earth Observation Satellites

The **Onboard Edge-AI Cloud Detection and Removal** project is the companion repository of the Master's thesis *"On-board Cloud Detection and Removal Study for Edge-AI-enabled EO Satellites"*, developed at the Interdisciplinary Centre for Security, Reliability and Trust ([SnT](https://www.uni.lu/snt-en/research/)) of the University of Luxembourg, within the [SIGCOM](https://www.uni.lu/snt-en/research-groups/SIGCOM/) research group. The work investigates whether contemporary deep learning models for cloud detection and cloud removal in satellite imagery can be deployed under the strict resource, power, and thermal envelopes that govern on-board processing in CubeSat-class missions, and quantifies the trade-offs that emerge between inference quality and edge-device feasibility.

The study compares four model families across two compute platforms representing opposite ends of the deployment spectrum. On the reference side, the [UL-HPC Iris cluster](https://hpc.uni.lu/) provides high-performance NVIDIA Tesla V100-SXM2 GPUs to establish ground-truth inference quality and unconstrained efficiency baselines. On the edge side, the [NVIDIA Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit) is used as the embedded target, since its 7–15 W power envelope, Ampere-class GPU, and JetPack 6.1 software stack are representative of the avionics class currently flying on platforms such as the D-Orbit ION-SCV satellites and several other commercial CubeSat AI demonstrators. The contrast between the two platforms exposes the engineering effort required to bridge the gap from research code targeting workstation GPUs to functional deployment on ARM64 hardware with shared CPU/GPU memory and the SM 87 compute capability.

The four model families evaluated cover both tasks of interest. **CloudGAN** [1] is a TensorFlow 1.15 pipeline comprising an Autoencoder and a U-Net for cloud detection together with an SN-PatchGAN inpainter for cloud removal. **DVPNet** [2] is a recent dual-view prompting network for cloud removal that combines spatial and frequency-domain information through a Restormer-based transformer backbone. **CD-Mamba** [3] is a hybrid CNN-Mamba state-space model for cloud detection designed for boundary precision and terrain robustness on highly heterogeneous Landsat-8 imagery. **OmniCloudMask** [4] is a deployment-ready open-source library trained for sensor-agnostic cloud and cloud-shadow segmentation across multiple satellite platforms.

For every model-platform combination the same two families of metrics are recorded: quality metrics (F1 and IoU for detection; PSNR, SSIM, and MAE for removal) and efficiency metrics (latency, throughput, peak memory, power, and energy per inference). This repository contains all source code, SLURM job descriptions, conda environment specifications, deployment patches, and raw result files needed to reproduce every experiment reported in the thesis. The full thesis manuscript and the weekly progress report accompanying it are also included at the root of this repository, and they provide the complete narrative behind every script and configuration found in the model-specific subdirectories.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Hardware Platforms](#hardware-platforms)
   - [UL-HPC Iris Cluster](#ul-hpc-iris-cluster)
   - [NVIDIA Jetson Orin Nano Developer Kit](#nvidia-jetson-orin-nano-developer-kit)
   - [Hardware Comparison](#hardware-comparison)
3. [Models Evaluated](#models-evaluated)
4. [Software Environments](#software-environments)
5. [Power Measurement Methodology](#power-measurement-methodology)
6. [Datasets](#datasets)
7. [References](#references)

---

## Repository Structure

```
Onboard-Edge-AI-Cloud-Detection-Removal/
├── README.md                                   ← Main Project Description file
├── DATASETS.md                                 ← Dataset descriptions, download links, and preparation commands
├── Weekly_Report.pdf                           ← Weekly progress report
│
├── CloudGAN/
│   ├── README.md
│   ├── environment/
│   │   └── requirements_cloudgan.txt
│   ├── HPC_Iris/
│   │   ├── scripts/                            ← Training + evaluation Python and SLURM scripts
│   │   │   ├── AE/
│   │   │   ├── UNet/
│   │   │   ├── SN-PatchGAN/
│   │   │   └── Uncertainty_Filter/
│   │   └── results/                            ← Raw JSON & Visuals output files
│   │       ├── AE/
│   │       ├── UNet/
│   │       ├── SN-PatchGAN/
│   │       └── Uncertainty_Filter/
│   └── Jetson_Orin_Nano/
│       ├── scripts/                            ← Evaluation Python scripts
│       │   ├── AE/
│       │   ├── UNet/
│       │   └── SN-PatchGAN/
│       └── results/                            ← Raw JSON output files
│           ├── AE/
│           ├── UNet/
│           └── SN-PatchGAN/
│
├── DVPNet/
│   ├── README.md
│   ├── environment/
│   │   ├── requirements_dvpnet_iris.txt
│   │   └── requirements_dvpnet_jetson.txt
│   ├── HPC_Iris/
│   │   ├── scripts/                            ← Evaluation Python and SLURM scripts
│   │   │   ├── RICE-I/
│   │   │   ├── RICE-II/
│   │   │   └── T-Cloud/
│   │   └── results/                            ← Raw JSON & Visuals output files
│   │       ├── RICE-I/
│   │       ├── RICE-II/
│   │       └── T-Cloud/
│   └── Jetson_Orin_Nano/
│       ├── scripts/                            ← Evaluation Python scripts
│       └── results/                            ← Raw JSON output files
│           ├── RICE-I/
│           ├── RICE-II/
│           └── T-Cloud/
│
├── CD-Mamba/
│   ├── README.md
│   ├── environment/
│   │   ├── requirements_cdmamba_iris.txt
│   │   └── requirements_cdmamba_jetson.txt
│   ├── patches/                                ← Critical: sm_87 patches for causal-conv1d and mamba-ssm
│   │   ├── causal_conv1d_sm87.patch
│   │   └── mamba_ssm_sm87.patch
│   ├── HPC_Iris/
│   │   ├── scripts/                            ← Evaluation Python and SLURM scripts
│   │   └── results/                            ← Raw JSON & Visuals output files
│   └── Jetson_Orin_Nano/
│       ├── scripts/                            ← Evaluation Python scripts
│       └── results/                            ← Raw JSON output files
│
└── OmniCloudMask/
    ├── README.md
    ├── environment/
    │   ├── requirements_omnicloud_iris.txt
    │   └── requirements_omnicloud_jetson.txt
    ├── HPC_Iris/
    │   ├── scripts/                            ← Evaluation Python and SLURM scripts
    │   └── results/                            ← Raw JSON & Visuals output files
    └── Jetson_Orin_Nano/
        ├── scripts/                            ← Evaluation Python scripts
        └── results/                            ← Raw JSON output files
```

Each model directory follows the same template: a model-specific `README.md` with full deployment notes, an `environment/` folder containing the exact conda specifications used on each platform, and two platform subdirectories (`HPC_Iris/` and `Jetson_Orin_Nano/`) holding evaluation scripts and raw results. CD-Mamba additionally includes a `patches/` folder with the source-level modifications required to compile its CUDA extensions for the Jetson's SM 87 architecture.

---

## Hardware Platforms

The choice of two platforms with very different design objectives is central to the methodology of this thesis. The Iris cluster provides an unconstrained reference point where the model can be observed in its most favourable conditions, isolating model behaviour from hardware limitations. The Jetson Orin Nano provides an embedded reality check, where the same model must contend with limited memory, lower compute capacity, a different instruction set architecture, and a more restrictive software stack. Reading the two sets of measurements side by side is what allows the thesis to answer the underlying engineering question: how much of the gap between research-grade results and on-board feasibility is intrinsic to the model and how much is a matter of deployment effort.

### UL-HPC Iris Cluster

The [UL-HPC Iris cluster](https://hpc.uni.lu/) is the high-performance computing facility operated by the University of Luxembourg. Access is managed through the [ULHPC IAM portal](https://hpc-ipa.uni.lu/) and connections use SSH on port 8022 as documented in the [SSH connection guide](https://hpc-docs.uni.lu/connect/ssh/). Full operational documentation is available in the [ULHPC user documentation](https://hpc-docs.uni.lu/).

All jobs in this thesis ran on the `gpu` SLURM partition, on nodes equipped with four NVIDIA Tesla V100-SXM2 GPUs each. The relevant hardware and software specifications are summarised in Table 1.

**Table 1.** UL-HPC Iris reference platform configuration.

| Component | Specification |
|---|---|
| CPU | Intel Xeon (server class, Skylake/Cascade Lake) |
| GPU | NVIDIA Tesla V100-SXM2 |
| GPU memory | 32 GB HBM2 |
| CUDA cores | 5120 per GPU |
| Compute capability | SM 70 |
| Memory bandwidth | ~900 GB/s |
| Interconnect | InfiniBand (HDR) |
| Job scheduler | SLURM (`gpu` partition, QoS `normal`) |
| Storage | GPFS scratch, 10 TB per-user quota |
| CUDA (system) | 11.1 (not used; per-environment CUDA installed via conda) |
| Python | Managed per environment under `/scratch/users/<user>/miniconda3/` |

The system CUDA installation (version 11.1) was never used directly in this thesis. Instead, each conda environment ships its own CUDA toolkit and cuDNN libraries, isolating dependencies and avoiding the well-known compatibility friction between TensorFlow 1.15 and modern cluster CUDA versions. The Lmod environment-module system available on the cluster also turned out to be impractical for the project, since it is only loaded on GPU compute nodes by default and the available CUDA modules are not compatible with the legacy TensorFlow 1.x requirements of CloudGAN.

### NVIDIA Jetson Orin Nano Developer Kit

The [NVIDIA Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit) is a compact embedded board built around the NVIDIA Tegra Orin (T234) system-on-chip, designed specifically for AI workloads at the edge. Three properties make it a sensible representative of CubeSat-class on-board hardware. First, its 7–15 W power envelope sits inside the orbital-average power budget of a 6U or 12U platform, which is typically 5–20 W for the on-board computer alone. Second, the full NVIDIA software stack (CUDA, cuDNN, TensorRT) is available on the device, so academic code targeting workstation GPUs has a realistic path to being ported across rather than requiring a full re-implementation. Third, there is direct flight precedent: the D-Orbit ION-SCV satellites carry Jetson hardware in orbit today, and several other commercial CubeSat demonstrators use Jetson-class devices for on-board AI.

The relevant specifications are summarised in Table 2.

**Table 2.** NVIDIA Jetson Orin Nano edge platform configuration.

| Component | Specification |
|---|---|
| SoC | NVIDIA Tegra Orin (T234) |
| CPU | 6-core Arm Cortex-A78AE @ 1.5 GHz |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 third-generation Tensor cores |
| Compute capability | SM 87 |
| GPU peak compute | 40 TOPS (INT8) |
| Memory | 8 GB LPDDR5, unified CPU/GPU (no separate VRAM) |
| Storage | microSD card slot, M.2 NVMe support |
| Power modes | 7 W and 15 W |
| Connectivity | Gigabit Ethernet, USB 3.2 ×4, DisplayPort, 40-pin GPIO |
| OS | Ubuntu 22.04 (L4T 36.4.0) |
| [JetPack](https://developer.nvidia.com/embedded/jetpack) | 6.1 |
| CUDA | 12.6 |

Three architectural specificities of the Jetson Orin Nano affect both the porting effort and the measurement methodology, and they do not arise on a workstation GPU. First, the unified memory model: there is no separate VRAM, and the 8 GB pool is shared between the operating system, the CPU process, and the GPU. Host-to-device copies are essentially free, but memory pressure during heavy compilation jobs (notably the mamba-ssm CUDA extensions of CD-Mamba) becomes a real constraint and required an 8 GB swap file to be added to the device. Second, the absence of `nvidia-smi`: the familiar command-line utility used on workstation GPUs simply does not exist on Tegra SoCs, and an equivalent role is played by `tegrastats`, which exposes per-rail voltage and current samples through a streaming interface. Third, the ARM64 (aarch64) architecture: standard PyPI wheels for PyTorch and TensorFlow do not work on Jetson, and platform-specific wheels distributed by NVIDIA through the [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) must be used instead.

### Hardware Comparison

**Table 3.** Cross-platform summary highlighting the contrasts between the reference and edge environments.

| Property | UL-HPC Iris (V100) | Jetson Orin Nano | Implication |
|---|---|---|---|
| GPU memory | 32 GB HBM2 | 8 GB unified LPDDR5 | Fits all models on both, but compilation pressure on Jetson |
| Compute capability | SM 70 | SM 87 | Custom CUDA kernels need recompilation with SM 87 in arch list |
| Power envelope | ~300 W (board TDP) | 7–15 W | ~20× difference; energy/inference is the meaningful metric for orbit |
| Instruction set | x86_64 | aarch64 (ARM64) | PyPI wheels often unavailable; vendor wheels required |
| Power monitoring | `nvidia-smi`, `pynvml` | `tegrastats` | Different rails reported; custom `TegraStatsMonitor` written for parity |
| Job submission | SLURM scheduler | Direct shell execution | No queueing on edge; sequential runs |
| Compatible with all 4 models | yes (out of the box with version pins) | yes (with patches and stubs) | Engineering effort concentrated on the Jetson side |

---

## Models Evaluated

Four model families covering six configurations are evaluated. The contrast between them is intentional: each model represents a different operating point on the trade-off curve between detection or removal quality, parameter count, computational cost, and deployment maturity. Table 4 summarises the architectures, and the model-specific READMEs inside each subdirectory provide the full background and deployment notes.

**Table 4.** Architectures evaluated in this thesis.

| Model | Task | Backbone | Parameters | FLOPs (typical) | Framework | Repository / Reference |
|---|---|---|---|---|---|---|
| CloudGAN-AE | Detection | Convolutional encoder-decoder | 198,593 | 3.50 G | TensorFlow 1.15 | [JerrySchonenberg/CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) |
| CloudGAN-U-Net | Detection | U-Net (skip connections) | 1,941,381 | 6.92 G | TensorFlow 1.15 | [JerrySchonenberg/CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) |
| CloudGAN-SN-PatchGAN | Removal | SN-PatchGAN inpainter | 4,052,478 | 55.57 G | TensorFlow 1.15 | [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting) |
| DVPNet | Removal | Spatio-frequency prompting U-Net (Restormer backbone) | 9,978,372 | 50–96 G | PyTorch 2.5 | [huangwenwenlili/DVPNet](https://github.com/huangwenwenlili/DVPNet) |
| CD-Mamba | Detection | Hybrid CNN + Cloud-SMB + DA-Block | 111,027 | 0.20 G | PyTorch 2.5 | [kunzhan/CD-Mamba](https://github.com/kunzhan/CD-Mamba) |
| OmniCloudMask v1.7.1 | Detection (4-class, zero-shot) | Ensemble of two SMP U-Nets (EdgeNeXt + RegNetY) | 14,370,000 | 31.86 G | PyTorch (SMP) | [DPIRD-DMA/OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) · [GeoAI](https://geoai.gishub.org/) |

CloudGAN is the legacy academic baseline of the study, originally published with its source code, weights, and accompanying paper on the [CloudGAN repository](https://github.com/JerrySchonenberg/CloudGAN). Its cloud removal stage builds on the [Generative Inpainting framework](https://github.com/JiahuiYu/generative_inpainting) and uses the [NeuralGym toolkit](https://github.com/JiahuiYu/neuralgym) as its training infrastructure. DVPNet is the most recent cloud removal architecture in the comparison and represents the state of the art in frequency-domain cloud handling, with its [reference implementation](https://github.com/huangwenwenlili/DVPNet) built on the [Restormer](https://github.com/swz30/Restormer) high-resolution image restoration backbone. CD-Mamba targets the lightweight modern detection literature, and its 111,027-parameter footprint makes it a natural candidate for the most constrained on-board scenarios; pretrained weights were obtained directly from the corresponding author and subsequently uploaded to the [CD-Mamba repository](https://github.com/kunzhan/CD-Mamba) on May 18 2026. OmniCloudMask completes the comparison from the deployment side: it is not a research codebase but an open-source Python library, distributed on PyPI and documented through the [project repository](https://github.com/DPIRD-DMA/OmniCloudMask), with an integration into the broader [GeoAI ecosystem](https://geoai.gishub.org/) that supports sensor-agnostic cloud and shadow segmentation across Sentinel-2, Landsat, and PlanetScope data.

---

## Software Environments

All experiments rely on per-model conda environments rather than a single shared one. The reason is that the four models target incompatible framework stacks: CloudGAN is built on TensorFlow 1.15 (which itself requires CUDA 10.0 and is restricted to Python 3.6.13), DVPNet was originally written against PyTorch 1.13 with CUDA 11.7, CD-Mamba requires PyTorch 2.1 with custom CUDA extensions for the Mamba state-space kernels, and OmniCloudMask is distributed through the modern PyTorch 2.x stack via the `geoai-py` package. Combining all of these into a single environment is not possible: the resulting dependency graph is unsatisfiable, and several of the version constraints (notably the TensorFlow 1.15 dependency chain) are mutually exclusive with the modern PyTorch ecosystem used by the other three.

On the Iris cluster, every environment is installed under a per-user Miniconda prefix at `/scratch/users/<user>/miniconda3/`, isolating dependencies and providing the per-environment CUDA toolkits that the system installation cannot supply. Where NVCC is needed to compile CUDA extensions (CD-Mamba), it is pulled in from the official `nvidia` conda channel and lives entirely inside the environment prefix. On the Jetson side, the same per-model isolation pattern is followed using Anaconda for aarch64, with the additional constraint that PyTorch and TensorFlow must come from the JetPack-specific wheels distributed by NVIDIA. The result is a set of four environments per platform, each containing only what its model needs, and each fully reproducible from the requirements files included in this repository under `<model>/environment/`. Tables 5 and 6 summarise these environments at a glance; the per-model READMEs document the full installation procedure, including the version pins, patches, and known issues encountered when bringing each one to a working state.

**Table 5.** Conda environments on the UL-HPC Iris cluster.

| Environment | Python | Framework | CUDA | Models served |
|---|---|---|---|---|
| `cloudgan` | 3.6.13 | TensorFlow-GPU 1.15 | 10.0 (conda-forge) | CloudGAN AE, U-Net, SN-PatchGAN |
| `DVPNet` | 3.10 | PyTorch 1.13.1 | 11.7 | DVPNet |
| `cdmamba` | 3.10 | PyTorch 2.1.0 (+ custom mamba-ssm) | 11.8 (nvidia channel) | CD-Mamba |
| `GeoAI` | 3.12 | PyTorch (cu118) + `geoai-py` | 11.8 | OmniCloudMask v1.7.1 |

**Table 6.** Conda environments on the NVIDIA Jetson Orin Nano.

| Environment | Python | Framework | CUDA | Models served |
|---|---|---|---|---|
| `cloudgan_jetson` | 3.10 | TensorFlow 2.16.1 (NVIDIA aarch64 wheel) + TF1→TF2 shim | 12.6 | CloudGAN AE, U-Net, SN-PatchGAN |
| `DVPNet` | 3.10 | PyTorch 2.5.0 (NVIDIA aarch64 wheel) | 12.6 | DVPNet |
| `cdmamba` | 3.10 | PyTorch 2.5.0 + custom mamba-ssm (SM 87 patched) | 12.6 | CD-Mamba |
| `omnicloud` | 3.10 | PyTorch 2.5.0 + `omnicloudmask==1.7.1` | 12.6 | OmniCloudMask v1.7.1 |

---

## Power Measurement Methodology

Energy per inference is the central efficiency metric of this thesis, and obtaining a meaningful value requires reading the right power source on each platform with the right sampling interval. Because the two platforms expose power information through different interfaces, the measurement procedure also differs between them, but the underlying logic is identical: sample the relevant rail at high frequency during a timed inference loop, compute the mean power, and combine it with the measured latency to obtain a per-inference energy figure.

On the Iris cluster, the NVIDIA Tesla V100-SXM2 exposes board-level power draw through `nvidia-smi`. The Python library `pynvml` provides programmatic access to the same information via the NVML interface, and is preferred here over invoking `nvidia-smi` as a subprocess because it removes the overhead of process creation and parsing on each sample. Samples are taken every 100 ms during the timed loop, accumulated, and averaged after the loop completes. The reported value is total board power, encompassing both the GPU die and the HBM2 memory subsystem. Across the 100-iteration measurements used in this thesis, the relative standard deviation of the samples consistently stayed below 5%, suggesting that the mean is a meaningful representation of the steady-state inference load.

On the Jetson Orin Nano, the absence of `nvidia-smi` is compensated by the proprietary `tegrastats` utility, which streams a sampled status line at configurable intervals. Three rails are parsed from each line and stored separately:

| Rail | Description | Role |
|---|---|---|
| `VDD_IN` | Total SoC input power | Full board power, including peripherals |
| `VDD_CPU_GPU_CV` | CPU + GPU + computer-vision island | **Primary AI inference rail** |
| `VDD_SOC` | Memory controller, fabric, peripherals | Auxiliary platform contribution |

`VDD_CPU_GPU_CV` is the rail used as the primary energy indicator throughout the thesis, because it isolates the compute subsystem performing the inference from the noise contributed by the rest of the platform. The `VDD_IN` and `VDD_SOC` rails are reported alongside it for completeness in the per-model result files. To keep the interface consistent across all four model deployments, the polling logic is encapsulated in a custom `TegraStatsMonitor` class that wraps `tegrastats` as a subprocess, parses each line with a regular expression, accumulates per-rail running totals, and exposes per-rail mean values through a context-manager API:

```python
with TegraStatsMonitor() as mon:
    for _ in range(100):
        _ = model(dummy_input)
power_w  = mon.avg_power('VDD_CPU_GPU_CV')
energy_j = power_w * (latency_ms / 1000.0)
```

The same abstraction is reused identically across the CloudGAN, DVPNet, CD-Mamba, and OmniCloudMask evaluations, which guarantees that the per-model power numbers are directly comparable. The 100 ms sampling interval is matched on both platforms, so the temporal resolution of the measurements is identical despite the underlying tools being different.

---

## Datasets

Five datasets are used in this thesis. The cloud detection task uses **38-Cloud** for the CloudGAN AE and U-Net, and the **Landsat-8 Biome** dataset for CD-Mamba and OmniCloudMask. The cloud removal task uses **RICE-I**, **RICE-II**, and **T-Cloud** for the SN-PatchGAN and DVPNet evaluations. Detailed descriptions of each dataset, including download links, preprocessing commands, and patch-extraction pipelines (notably for the Biome 42,873-patch evaluation), are documented separately in [DATASETS.md](./DATASETS.md).

---

## References

[1] Schonenberg, J., Kluiver, F. (2022). *CloudGAN: Cloud Removal from Satellite Images using Generative Adversarial Networks*. GitHub Repository and Paper. https://github.com/JerrySchonenberg/CloudGAN

[2] Deng, Y., Huang, W., Tang, Z., Duan, J. (2025). *Dual-View Prompting for Cloud Removal*. IEEE Transactions on Geoscience and Remote Sensing, vol. 63, art. 5645913, pp. 1–14. Repository: https://github.com/huangwenwenlili/DVPNet

[3] Xue, Y., Wang, J., Zhan, K. et al. (2025). *CD-Mamba: Cloud Detection via Cloud Spatial-Mamba Block*. arXiv preprint arXiv:2509.04729. Repository: https://github.com/kunzhan/CD-Mamba

[4] Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., George, R. J. (2025). *Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask*. Remote Sensing of Environment, vol. 322, art. 114694. https://doi.org/10.1016/j.rse.2025.114694. Repository: https://github.com/DPIRD-DMA/OmniCloudMask · GeoAI integration: https://geoai.gishub.org/

[5] Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., Huang, T. (2019). *Free-Form Image Inpainting with Gated Convolution*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). Repository: https://github.com/JiahuiYu/generative_inpainting

[6] Yu, J. (2018). *NeuralGym: A Deep Learning Toolkit for Generative Models in TensorFlow*. GitHub Repository. https://github.com/JiahuiYu/neuralgym

[7] Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5728–5739. Repository: https://github.com/swz30/Restormer

[8] Iakubovskii, P. (2019). *Segmentation Models PyTorch (SMP)*. GitHub Repository. https://github.com/qubvel/segmentation_models.pytorch

[9] Mohajerani, S., Saeedi, P. (2019). *Cloud-Net: An End-to-End Cloud Detection Algorithm for Landsat 8 Imagery*. In Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS). Repository: https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection. Dataset: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset · Kaggle: https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images

[10] Foga, S., Scaramuzza, P. L., Guo, S., Zhu, Z., Dilley, R. D., Beckmann, T., Schmidt, G. L., Dwyer, J. L., Hughes, M. J., Laue, B. (2017). *Cloud detection algorithm comparison and validation for operational Landsat data products*. Remote Sensing of Environment, vol. 194, pp. 379–390. Landsat-8 Cloud Cover Assessment Validation Data (Biome): https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

[11] Lin, D., Xu, G., Wang, X., Wang, Y., Sun, X., Fu, K. (2019). *A Remote Sensing Image Dataset for Cloud Removal (RICE)*. arXiv preprint arXiv:1901.00600. Repository: https://github.com/BUPTLdy/RICE_DATASET

[12] Ding, H., Zi, Y., Xie, F. (2022). *Uncertainty-based Thin Cloud Removal Network via Conditional Variational Autoencoders*. In Proceedings of the Asian Conference on Computer Vision (ACCV). T-Cloud dataset reference.

[13] University of Luxembourg High Performance Computing (ULHPC). *ULHPC Documentation*. https://hpc-docs.uni.lu/

[14] University of Luxembourg High Performance Computing (ULHPC). *SSH Connection Guide*. https://hpc-docs.uni.lu/connect/ssh/

[15] University of Luxembourg High Performance Computing (ULHPC). *Identity and Access Management Portal*. https://hpc-ipa.uni.lu/

[16] NVIDIA Corporation. *Jetson Orin Nano Developer Kit*. https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit

[17] NVIDIA Corporation. *NVIDIA JetPack SDK*. https://developer.nvidia.com/embedded/jetpack · Archive of platform-specific Python wheels: https://developer.nvidia.com/embedded/jetpack-archive

[18] Gu, A., Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. In Proceedings of the First Conference on Language Modeling (COLM). Repository: https://github.com/state-spaces/mamba

[19] Dao, T. *causal-conv1d: Lightweight Causal 1D Convolution CUDA Kernels*. GitHub Repository. https://github.com/Dao-AILab/causal-conv1d
