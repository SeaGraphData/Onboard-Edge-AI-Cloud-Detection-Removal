# On-board Cloud Detection and Removal for Edge-AI-enabled Earth Observation Satellites

This is the companion repository of the Master's thesis *"On-board Cloud Detection and Removal Study for Edge-AI-enabled EO Satellites"*, written at the Interdisciplinary Centre for Security, Reliability and Trust ([SnT](https://www.uni.lu/snt-en/research/)), University of Luxembourg, inside the [SIGCOM](https://www.uni.lu/snt-en/research-groups/SIGCOM/) research group. The thesis asks a fairly practical question: can recent deep learning models for cloud detection and cloud removal run on the kind of hardware that would actually fly on a small Earth observation satellite, and what does it cost in inference time, memory, power, and engineering effort to get them there?

To answer that, four model families are run on two platforms with very little in common. The reference platform is the [UL-HPC Iris cluster](https://hpc.uni.lu/), where NVIDIA Tesla V100-SXM2 GPUs handle the heavy lifting and the only real constraint is the SLURM queue. The edge platform is the [NVIDIA Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit), a 7 W to 15 W board with an Ampere GPU, 8 GB of shared CPU/GPU memory, and the JetPack 6.1 stack. The Jetson Orin Nano is not a satellite, but its power budget and software stack are close enough to what flies today on platforms like the D-Orbit ION-SCV that the numbers transfer with reasonable fidelity. Putting the two sets of results side by side is what separates model behaviour from hardware limitations.

The four models cover both tasks. **CloudGAN** [1] is a TensorFlow 1.15 pipeline with an Autoencoder and a U-Net for detection plus an SN-PatchGAN inpainter for removal. It is the legacy academic baseline of the study: small networks, simple ideas, and an end-of-life framework that fights back at every step. **DVPNet** [2] is a 2025 cloud removal network with a Restormer transformer backbone and a frequency-domain prompting block, currently the most expensive removal model in the comparison. **CD-Mamba** [3] is also from 2025 and goes in the opposite direction: a hybrid CNN-Mamba detector with 111 027 parameters that reaches state-of-the-art on Biome while occupying 0.42 MB on disk. **OmniCloudMask** [4] is the odd one out. It is not a research codebase but a pip-installable library distributed through the [GeoAI ecosystem](https://geoai.gishub.org/) and trained to be sensor-agnostic across Sentinel-2, Landsat, and PlanetScope without per-sensor retraining. Putting the four together in the same experimental setup gives a fair view of what the research-to-deployment gap actually costs.

For every model and platform combination, the same two families of numbers are collected. The quality numbers (F1 and IoU for detection, PSNR and SSIM and MAE for removal) are hardware-independent and are reported once. The efficiency numbers (latency, throughput, peak memory, power, energy per inference) are measured separately on each platform. Everything needed to reproduce the runs is here: evaluation scripts, SLURM job descriptions, conda specifications, source-level patches for the Jetson SM 87 architecture, and the raw JSON output files. The full thesis manuscript and the weekly progress report are at the root of this repository and contain the narrative behind every script.

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

Each model folder follows the same template. A `README.md` covers the model and its deployment quirks, an `environment/` folder holds the conda specifications used on each platform, and the two platform subfolders (`HPC_Iris/` and `Jetson_Orin_Nano/`) carry the evaluation scripts and the raw results produced by them. CD-Mamba additionally has a `patches/` folder, because compiling its CUDA extensions on the Jetson Orin Nano needs two small source-level patches that no released version of the packages ships with.

---

## Hardware Platforms

Two platforms were used in the thesis, and they were chosen because they could not be more different. The Iris cluster is the comfortable end of the spectrum: 32 GB of HBM2 per GPU, no real memory or power pressure, and a well-supported software stack. The Jetson Orin Nano is the uncomfortable end: 8 GB of memory shared between CPU and GPU, a tight power budget, ARM64 instead of x86, and a JetPack stack that is not always compatible with the versions of CUDA and Python that the model authors assumed. Most of the engineering work in this thesis happened on the Jetson side. Iris was useful as the reference, the place where the model could be observed running freely and the numbers could be trusted without worrying about a missing CUDA architecture or a swap file overflow.

### UL-HPC Iris Cluster

[Iris](https://hpc.uni.lu/) is the cluster operated by the University of Luxembourg HPC team. Access goes through the [ULHPC IAM portal](https://hpc-ipa.uni.lu/) (registered SSH key per machine) and connections use port 8022, as described in the [SSH connection guide](https://hpc-docs.uni.lu/connect/ssh/). The full operational documentation lives at [hpc-docs.uni.lu](https://hpc-docs.uni.lu/).

Every job in this thesis ran on the `gpu` SLURM partition, on nodes with four Tesla V100-SXM2 GPUs each. One GPU per job was always enough; none of the four models exceeds 1 GB of GPU memory at inference time. The relevant specifications are in Table 1.

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

The system CUDA on Iris is 11.1 and was never used directly. Each conda environment carries its own CUDA toolkit and cuDNN, which is the only way to keep a TensorFlow 1.15 setup alive in 2026. The Lmod module system available on the cluster was also not very helpful here. It is only loaded on GPU compute nodes by default, and the CUDA modules it exposes are not compatible with the legacy TensorFlow stack that CloudGAN requires. Everything ended up living inside the conda prefix instead.

### NVIDIA Jetson Orin Nano Developer Kit

The [Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) is a small embedded board built around the NVIDIA Tegra Orin (T234) system-on-chip. It is roughly the size of a Raspberry Pi and is designed for AI workloads at the edge. Three things make it a sensible stand-in for CubeSat-class hardware. Its 7 W to 15 W power budget sits inside the orbital-average envelope of a 6U or 12U platform (typically 5 W to 20 W for the on-board computer alone). The full NVIDIA software stack (CUDA, cuDNN, TensorRT) is available on the device, so research code targeting workstation GPUs has a real path to running here rather than needing a from-scratch reimplementation. And there is direct flight precedent: D-Orbit ION-SCV satellites already carry Jetson hardware in orbit, and several other commercial CubeSat demonstrators use Jetson-class devices for on-board AI. The relevant specifications are in Table 2.

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

Three Jetson specificities matter for both deployment and measurement, and none of them shows up on a workstation GPU. The first is unified memory. There is no VRAM on the Orin Nano; the 8 GB pool is shared by the operating system, the CPU process, and the GPU. Host-to-device copies are basically free because the data does not move, but heavy compilation jobs (the `mamba-ssm` CUDA extensions of CD-Mamba being the worst case) hit memory limits very quickly. Adding an 8 GB swap file at `/swapfile_extra` was a precondition for that build to even finish. The second is the absence of `nvidia-smi`. The familiar utility used everywhere else simply does not exist on Tegra SoCs. The equivalent role is played by `tegrastats`, which exposes per-rail voltage and current samples and streams them at a configurable interval. The third is the ARM64 instruction set. Standard PyPI wheels for PyTorch and TensorFlow do not work on Jetson; the wheels distributed by NVIDIA through the [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) must be used instead, and they are compiled for specific Python and CUDA versions.

### Hardware Comparison

Table 3 places the two platforms side by side and notes the practical consequence of each difference.

**Table 3.** Cross-platform summary highlighting the contrasts between the reference and edge environments.

| Property | UL-HPC Iris (V100) | Jetson Orin Nano | Practical consequence |
|---|---|---|---|
| GPU memory | 32 GB HBM2 | 8 GB unified LPDDR5 | All models fit on both, but compilation is tight on Jetson |
| Compute capability | SM 70 | SM 87 | Custom CUDA kernels need recompilation with SM 87 in the arch list |
| Power envelope | ~300 W (board TDP) | 7 W to 15 W | About 20x difference; energy per inference is what matters in orbit |
| Instruction set | x86_64 | aarch64 (ARM64) | PyPI wheels often unavailable; vendor wheels required |
| Power monitoring | `nvidia-smi`, `pynvml` | `tegrastats` | Different rails reported; a custom `TegraStatsMonitor` was written for parity |
| Job submission | SLURM scheduler | Direct shell execution | No queueing on the edge; runs are sequential |
| Compatible with all 4 models | yes (with version pins) | yes (with patches and stubs) | The engineering effort lives on the Jetson side |

---

## Models Evaluated

Four model families covering six configurations are evaluated. The set was chosen on purpose so that each model sits at a different point on the trade-off curve between detection or removal quality, parameter count, compute cost, and how ready it is for deployment. Table 4 summarises the architectures; the per-model READMEs inside each subdirectory go into the deployment details.

**Table 4.** Architectures evaluated in this thesis.

| Model | Task | Backbone | Parameters | FLOPs (typical) | Framework | Repository / Reference |
|---|---|---|---|---|---|---|
| CloudGAN-AE | Detection | Convolutional encoder-decoder | 198,593 | 3.50 G | TensorFlow 1.15 | [JerrySchonenberg/CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) |
| CloudGAN-U-Net | Detection | U-Net (skip connections) | 1,941,381 | 6.92 G | TensorFlow 1.15 | [JerrySchonenberg/CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) |
| CloudGAN-SN-PatchGAN | Removal | SN-PatchGAN inpainter | 4,052,478 | 55.57 G | TensorFlow 1.15 | [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting) |
| DVPNet | Removal | Spatio-frequency prompting U-Net (Restormer backbone) | 9,978,372 | 50-96 G | PyTorch 2.5 | [huangwenwenlili/DVPNet](https://github.com/huangwenwenlili/DVPNet) |
| CD-Mamba | Detection | Hybrid CNN + Cloud-SMB + DA-Block | 111,027 | 0.20 G | PyTorch 2.5 | [kunzhan/CD-Mamba](https://github.com/kunzhan/CD-Mamba) |
| OmniCloudMask v1.7.1 | Detection (4-class, zero-shot) | Ensemble of two SMP U-Nets (EdgeNeXt + RegNetY) | 14,370,000 | 31.86 G | PyTorch (SMP) | [DPIRD-DMA/OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) · [GeoAI](https://geoai.gishub.org/) |

CloudGAN is the legacy baseline of the study. The full pipeline (source code, weights, paper) is published in the [CloudGAN repository](https://github.com/JerrySchonenberg/CloudGAN). Its removal stage builds on the [Generative Inpainting framework](https://github.com/JiahuiYu/generative_inpainting) by Yu et al. and uses the [NeuralGym toolkit](https://github.com/JiahuiYu/neuralgym) as its training infrastructure. DVPNet is the most recent cloud removal architecture in the comparison and the most expensive. Its [reference implementation](https://github.com/huangwenwenlili/DVPNet) is built on the [Restormer](https://github.com/swz30/Restormer) high-resolution image restoration backbone and exploits frequency-domain information through a dedicated prompting block. CD-Mamba is interesting for the opposite reason: at 111 027 parameters it is the smallest model in this thesis by a wide margin, and it is the natural candidate for the most constrained on-board scenarios. The pre-trained weights were not initially in the public [CD-Mamba repository](https://github.com/kunzhan/CD-Mamba); they were obtained directly from the corresponding author, Prof. Kun Zhan, and subsequently uploaded to the repository on 18 May 2026. OmniCloudMask completes the comparison from the deployment side. It is not a research codebase but a pip-installable Python library, documented through its [project repository](https://github.com/DPIRD-DMA/OmniCloudMask) and integrated into the broader [GeoAI ecosystem](https://geoai.gishub.org/). It targets sensor-agnostic cloud and shadow segmentation across Sentinel-2, Landsat, and PlanetScope without any per-sensor retraining.

---

## Software Environments

The four models do not share a software stack. CloudGAN runs on TensorFlow 1.15, which needs CUDA 10.0 and is restricted to Python 3.6.13. DVPNet was originally written against PyTorch 1.13 with CUDA 11.7. CD-Mamba needs PyTorch 2.1 plus custom CUDA extensions for the Mamba state-space kernels. OmniCloudMask uses the modern PyTorch 2.x stack through the `geoai-py` package. Trying to fit all of this into a single environment is not realistic; the dependency graph is unsatisfiable, and the TensorFlow 1.15 chain is in particular mutually exclusive with anything PyTorch 2.x ships today. So every model gets its own conda environment, and the two platforms each maintain four of them.

On Iris, the environments live under a per-user Miniconda prefix at `/scratch/users/<user>/miniconda3/`. Where NVCC is needed to compile CUDA extensions (the CD-Mamba case), it is installed from the official `nvidia` conda channel and stays inside the environment prefix; the cluster's system NVCC is never used. On the Jetson, the same per-model isolation is followed using the Anaconda distribution for aarch64. The catch on the Jetson is that PyTorch and TensorFlow have to come from the JetPack-specific wheels distributed by NVIDIA. Tables 5 and 6 summarise both sets; the per-model READMEs document the full installation procedure, including the version pins, the patches, and the issues encountered along the way.

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

Energy per inference is the metric that matters most in this thesis, and getting a meaningful value out of either platform required reading the right power source at the right sampling rate. The two platforms expose power through different interfaces, so the measurement code differs, but the principle is identical on both: poll the relevant rail at high frequency during a timed inference loop, average the samples, and multiply by the measured latency to get a per-inference figure.

On Iris, the Tesla V100-SXM2 reports board-level power through `nvidia-smi`. The Python library `pynvml` gives programmatic access to the same data over the NVML interface, which is faster than calling `nvidia-smi` as a subprocess (no process creation, no string parsing per sample). Polling is done every 100 ms during the timed loop. The value reported is total board power and includes both the GPU die and the HBM2 memory subsystem. Across the 100-iteration measurements used in this thesis, the relative standard deviation of the samples stayed below 5%, so the mean is a fair representation of the steady-state inference load.

On the Jetson, the absence of `nvidia-smi` is filled by `tegrastats`. The utility streams one line per sampling interval; three rails are parsed from each line and stored separately:

| Rail | Description | Role |
|---|---|---|
| `VDD_IN` | Total SoC input power | Full board power, including peripherals |
| `VDD_CPU_GPU_CV` | CPU + GPU + computer-vision island | **Primary AI inference rail** |
| `VDD_SOC` | Memory controller, fabric, peripherals | Auxiliary platform contribution |

`VDD_CPU_GPU_CV` is the primary energy indicator throughout the thesis. It isolates the compute subsystem that actually runs the inference from the noise contributed by the rest of the board. `VDD_IN` and `VDD_SOC` are reported alongside it in the per-model result files for completeness. To keep the interface consistent across the four model deployments, the polling logic is wrapped in a small `TegraStatsMonitor` class that launches `tegrastats` as a subprocess, parses each line with a regular expression, accumulates per-rail running totals, and exposes per-rail mean values through a context-manager API:

```python
with TegraStatsMonitor() as mon:
    for _ in range(100):
        _ = model(dummy_input)
power_w  = mon.avg_power('VDD_CPU_GPU_CV')
energy_j = power_w * (latency_ms / 1000.0)
```

The same class is reused without modification across the CloudGAN, DVPNet, CD-Mamba, and OmniCloudMask Jetson evaluations. The 100 ms sampling interval is the same on both platforms, so the temporal resolution of the measurements is identical even though the tools below it are not.

---

## Datasets

Five datasets are used in this thesis. Cloud detection uses **38-Cloud** for the CloudGAN AE and U-Net, and the **Landsat-8 Biome** dataset for CD-Mamba and OmniCloudMask. Cloud removal uses **RICE-I**, **RICE-II**, and **T-Cloud** for the SN-PatchGAN and DVPNet evaluations. Full descriptions, download links, and the preprocessing commands (in particular the patching pipeline that turns the 96 Biome scenes into the 42 873 patches used by CD-Mamba and OmniCloudMask) are in [DATASETS.md](./DATASETS.md).

---

## References

[1] Schonenberg, J., Kluiver, F. (2022). *CloudGAN: Cloud Removal from Satellite Images using Generative Adversarial Networks*. GitHub Repository and Paper. https://github.com/JerrySchonenberg/CloudGAN

[2] Deng, Y., Huang, W., Tang, Z., Duan, J. (2025). *Dual-View Prompting for Cloud Removal*. IEEE Transactions on Geoscience and Remote Sensing, vol. 63, art. 5645913, pp. 1-14. Repository: https://github.com/huangwenwenlili/DVPNet

[3] Xue, Y., Wang, J., Zhan, K. et al. (2025). *CD-Mamba: Cloud Detection via Cloud Spatial-Mamba Block*. arXiv preprint arXiv:2509.04729. Repository: https://github.com/kunzhan/CD-Mamba

[4] Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., George, R. J. (2025). *Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask*. Remote Sensing of Environment, vol. 322, art. 114694. https://doi.org/10.1016/j.rse.2025.114694. Repository: https://github.com/DPIRD-DMA/OmniCloudMask · GeoAI integration: https://geoai.gishub.org/

[5] Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., Huang, T. (2019). *Free-Form Image Inpainting with Gated Convolution*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). Repository: https://github.com/JiahuiYu/generative_inpainting

[6] Yu, J. (2018). *NeuralGym: A Deep Learning Toolkit for Generative Models in TensorFlow*. GitHub Repository. https://github.com/JiahuiYu/neuralgym

[7] Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5728-5739. Repository: https://github.com/swz30/Restormer

[8] Iakubovskii, P. (2019). *Segmentation Models PyTorch (SMP)*. GitHub Repository. https://github.com/qubvel/segmentation_models.pytorch

[9] Mohajerani, S., Saeedi, P. (2019). *Cloud-Net: An End-to-End Cloud Detection Algorithm for Landsat 8 Imagery*. In Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS). Repository: https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection. Dataset: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset · Kaggle: https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images

[10] Foga, S., Scaramuzza, P. L., Guo, S., Zhu, Z., Dilley, R. D., Beckmann, T., Schmidt, G. L., Dwyer, J. L., Hughes, M. J., Laue, B. (2017). *Cloud detection algorithm comparison and validation for operational Landsat data products*. Remote Sensing of Environment, vol. 194, pp. 379-390. Landsat-8 Cloud Cover Assessment Validation Data (Biome): https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

[11] Lin, D., Xu, G., Wang, X., Wang, Y., Sun, X., Fu, K. (2019). *A Remote Sensing Image Dataset for Cloud Removal (RICE)*. arXiv preprint arXiv:1901.00600. Repository: https://github.com/BUPTLdy/RICE_DATASET

[12] Ding, H., Zi, Y., Xie, F. (2022). *Uncertainty-based Thin Cloud Removal Network via Conditional Variational Autoencoders*. In Proceedings of the Asian Conference on Computer Vision (ACCV). T-Cloud dataset reference.

[13] University of Luxembourg High Performance Computing (ULHPC). *ULHPC Documentation*. https://hpc-docs.uni.lu/

[14] University of Luxembourg High Performance Computing (ULHPC). *SSH Connection Guide*. https://hpc-docs.uni.lu/connect/ssh/

[15] University of Luxembourg High Performance Computing (ULHPC). *Identity and Access Management Portal*. https://hpc-ipa.uni.lu/

[16] NVIDIA Corporation. *Jetson Orin Nano Developer Kit*. https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit

[17] NVIDIA Corporation. *NVIDIA JetPack SDK*. https://developer.nvidia.com/embedded/jetpack · Archive of platform-specific Python wheels: https://developer.nvidia.com/embedded/jetpack-archive

[18] Gu, A., Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. In Proceedings of the First Conference on Language Modeling (COLM). Repository: https://github.com/state-spaces/mamba

[19] Dao, T. *causal-conv1d: Lightweight Causal 1D Convolution CUDA Kernels*. GitHub Repository. https://github.com/Dao-AILab/causal-conv1d
