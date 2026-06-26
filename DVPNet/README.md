# DVPNet

**Task:** Cloud removal  
**Paper:** IEEE TGRS 2025  
**Framework:** PyTorch + basicsr  
**Datasets:** RICE-I, RICE-II, T-Cloud

---

## Model Overview

DVPNet is a cloud removal network based on the basicsr image restoration framework. It provides three separate pretrained checkpoints, one per target dataset. All three checkpoints share the same architecture; only the weights differ.

| Checkpoint | Dataset | PSNR | SSIM |
|---|---|---|---|
| `rice1/net_g_best.pth` | RICE-I | 38.16 dB | 0.9656 |
| `rice2/net_g_best.pth` | RICE-II | — | — |
| `t-cloud/net_g_best.pth` | T-Cloud | — | — |

**Checkpoint distinctness verification:** An early question was whether the three checkpoint files were actually distinct or just copies under different names. MD5 hashes all differed. Pairwise L2 norms of the flattened parameter vectors confirmed they are genuinely different models:
- RICE-I vs RICE-II: 123.63
- RICE-I vs T-Cloud: 611.52
- RICE-II vs T-Cloud: 616.31

The RICE-I and RICE-II models are closer to each other than either is to T-Cloud, which makes sense given that T-Cloud contains a different distribution of cloud types.

---

## Environment Setup

### Iris Cluster

DVPNet provides a `DVPNet-env.yml` file, but it cannot be imported directly on Iris. Two fixes are required before creating the environment:

1. **Tsinghua mirror channels**: the `.yml` references `https://mirrors.tuna.tsinghua.edu.cn/...` channels that are not accessible outside China. Comment them out or replace with standard channels.
2. **`causal-conv1d==1.1.1`**: listed as a dependency but not used by DVPNet's Restormer-based architecture. This appears to be a copy-paste artefact. Comment it out to avoid a multi-minute unnecessary compilation step.

After those two edits:

```bash
conda env create -f DVPNet-env.yml
conda activate DVPNet
pip install pynvml   # not declared as a dependency but required for power measurement
```

Alternatively, create from scratch:

```bash
conda create -n DVPNet python=3.10 -y
conda activate DVPNet
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
pip install basicsr
pip install fvcore   # FLOPs counting on Iris
pip install pynvml   # GPU power measurement
pip install numpy scipy scikit-image tqdm PyYAML lmdb
```

See `environment/requirements_dvpnet_iris.txt` for the full pinned list.

### Jetson Orin Nano

DVPNet is built on basicsr and PyTorch. There is no legacy TF issue here, but three Jetson-specific problems must be resolved:

**1. NVIDIA aarch64 PyTorch wheel**

The standard PyPI PyTorch wheel does not work on Jetson. Use the NVIDIA-provided wheel for JetPack 6.1 / Python 3.10 / CUDA 12.6:

```bash
pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl
```

The wheel file is available from https://developer.nvidia.com/embedded/jetpack-archive (JetPack 6.1 → Python packages).

torchvision 0.20.0 must be compiled from source against this PyTorch installation (takes ~40 minutes on the Jetson):

```bash
git clone --branch v0.20.0 https://github.com/pytorch/vision.git
cd vision
python setup.py install
```

Additionally, `cuSPARSELt 0.7.1` must be installed because the NVIDIA PyTorch wheel depends on it at import time but it is not part of the default JetPack package set.

**2. basicsr patch for PyTorch 2.5**

The `basicsr` package as released on PyPI assumes an older PyTorch API. Its registry import mechanism in `basicsr/utils/registry.py` raises a `TypeError` on PyTorch 2.5 due to a change in how `torch.nn.Module` handles reserved attribute names. The fix is a single-line patch; refer to the thesis appendix (Appendix: Known Issues and Patches) for the exact diff.

**3. FLOPs library swap**

`fvcore` triggers a CUDA allocator failure on the Jetson the first time it is invoked. Replace with `ptflops`:

```bash
pip install ptflops
```

FLOPs are computed as `2 × MACs` from `ptflops.get_model_complexity_info()`.

**4. pynvml not available on Tegra**

`pynvml` does not function on Tegra SoCs. Remove all `pynvml` calls and replace with `TegraStatsMonitor` (integrated in the Jetson evaluation script). See `Jetson_Orin_Nano/scripts/dvpnet_jetson_eval.py`.

**ABI fixes:**

```bash
# NumPy: downgrade to 1.x branch
pip install "numpy<2"
# OpenCV
pip install opencv-python==4.9.0.80
# lmdb (lazy import in basicsr utilities, not declared as dependency)
pip install lmdb
```

See `environment/requirements_dvpnet_jetson.txt` for the full list.

---

## HPC Iris — Evaluation Workflow

Three SLURM scripts were written, one per checkpoint. The RICE-I and RICE-II scripts run full evaluation (efficiency metrics + image quality). The T-Cloud script runs efficiency-only evaluation with a synthetic dummy input, because the T-Cloud dataset was not available on the cluster at the time of the initial run; a full T-Cloud evaluation script was added later.

The evaluation scripts handle the hardcoded absolute paths present in the original DVPNet test code by replacing them with paths relative to the repository root.

Scripts: `HPC_Iris/scripts/`  
Results: `HPC_Iris/results/`

---

## Jetson Orin Nano — Evaluation

A single script `dvpnet_jetson_eval.py` evaluates all three pretrained checkpoints sequentially. It uses synthetic dummy inputs of shape `(1, 3, 256, 256)` — the full RICE/T-Cloud datasets are not required on the Jetson because quality metrics are hardware-independent and were already obtained on Iris.

The script measures: parameters, model size, FLOPs (via ptflops), latency, FPS, peak GPU memory, power (via TegraStatsMonitor, VDD_CPU_GPU_CV rail), and energy per inference.

Scripts: `Jetson_Orin_Nano/scripts/dvpnet_jetson_eval.py`  
Results: `Jetson_Orin_Nano/results/dvpnet_jetson_results.json`

---

## Results Summary

| Checkpoint | Platform | Params | Size | FLOPs | Latency (ms) | FPS | PSNR | SSIM |
|---|---|---|---|---|---|---|---|---|
| RICE-I | Iris V100 | — | ~38 MB | ~50.32B | ~40 | ~25 | 38.16 dB | 0.9656 |
| RICE-I | Jetson | — | ~38 MB | — | — | — | — | — |
| RICE-II | Iris V100 | — | ~38 MB | — | — | — | — | — |
| T-Cloud | Iris V100 | — | ~38 MB | — | — | — | — | — |

For full results including all efficiency metrics and image quality metrics (MAE, MSE, RMSE, BRMSE, SAM), see `HPC_Iris/results/` and `Jetson_Orin_Nano/results/`.
