# Datasets

This document describes all datasets used in the thesis, including download links, split details, and the exact preparation commands needed to reproduce the experiments. Where a dataset required custom preprocessing (patching, filtering, reorganisation), the full command sequence is given.

---

## 1. 38-Cloud

**Task:** Cloud detection  
**Models:** CloudGAN (AE, U-Net), CD-Mamba  
**Reference:** Mohajerani & Saeedi (2019)

| Split | Images | Notes |
|---|---|---|
| Train | 20,023 | Used to train AE and U-Net from scratch on Iris |
| Test | 2,352 | Used for all segmentation metric evaluation |

**Download:**  
https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images

**Structure expected by CloudGAN:**
```
38-Cloud/
    train/
        gt/          ← binary cloud masks (PNG)
        red/         ← red band (PNG)
        green/
        blue/
        nir/
    test/
        gt/
        red/
        green/
        blue/
        nir/
```

**Important note on the test set:** The first 20 images when sorted alphabetically are cloud-free ocean scenes with near-zero cloud coverage. When selecting representative sample images for visualisation, filter for patches with 20–70% cloud coverage to get meaningful examples.

---

## 2. RICE Dataset (RICE-I and RICE-II)

**Task:** Cloud removal  
**Models:** DVPNet, CloudGAN SN-PatchGAN (qualitative only)  
**Reference:** Zhang et al. (2019)

| Dataset | Pairs | Notes |
|---|---|---|
| RICE-I | 500 | Thin/medium cloud cover |
| RICE-II | 736 | Thick cloud cover |

**Download:**  
https://github.com/BUPTLdy/RICE_DATASET

**Structure:**
```
RICE_DATASET/
    RICE1/
        cloud/       ← cloudy input images
        label/       ← cloud-free reference images
    RICE2/
        cloud/
        label/
```

---

## 3. T-Cloud

**Task:** Cloud removal  
**Models:** DVPNet  

| Split | Images | Notes |
|---|---|---|
| Train | — | Not used (pretrained weights only) |
| Test | — | Used for image quality evaluation |

**Download:**  
https://github.com/zhiqiangdon/CU-Net (T-Cloud dataset link in the repository)

**Structure:**
```
T-Cloud/
    train/
        cloud/
        reference/
    test/
        cloud/
        reference/
```

---

## 4. Landsat-8 Biome Dataset

**Task:** Cloud detection  
**Models:** CD-Mamba, OmniCloudMask  
**Reference:** Foga et al. (2017)  
**Scale:** 42,873 patches of 384×384 px after preprocessing

**Download:**  
https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

The full dataset consists of 96 Landsat-8 scenes across 8 biome types (Bare, Forest, Grass, Shrubland, Snow, Urban, Water, Wetlands).

**Preprocessing — patching pipeline:**

The raw Landsat-8 scenes must be converted into 384×384 px patches before running CD-Mamba or OmniCloudMask. The patching script used in this thesis produces the 4-fold cross-validation split required by CD-Mamba's training configuration.

The complete patching pipeline is documented in the thesis (Chapter: Datasets) and in the CD-Mamba README. The commands below reproduce the patch creation:

```bash
# On Iris, from the CDMamba repository root
conda activate cdmamba

# Step 1: Download and organise raw scenes into the expected directory structure
# (scenes go under data/Biome/<biome_name>/<scene_id>/)

# Step 2: Run the patching script (produces 384x384 patches with 4-fold splits)
python tools/prepare_biome.py \
    --data_root /scratch/users/jfernandezmartinez/CDMamba/data/Biome \
    --output_root /scratch/users/jfernandezmartinez/CDMamba/CDMamba_patches \
    --patch_size 384 \
    --stride 384 \
    --n_folds 4

# This produces:
# CDMamba_patches/
#     fold_01/ ... fold_04/
#         train/ test/
#             images/  masks/
```

**Total patch count after preprocessing:** 42,873 patches  
**Storage requirement:** approximately 18 GB for the full patch dataset

**Bands used:**
- CD-Mamba: all Landsat-8 bands as configured by the repository (bands 1–7 + QA)
- OmniCloudMask: Red (Band 4), Green (Band 3), NIR (Band 5) only

---

## Dataset Storage on Iris Scratch

During this thesis, datasets were stored under:

```
/scratch/users/jfernandezmartinez/
├── CloudGAN/
│   └── 38-Cloud/
├── DVPNet/
│   └── datasets/
│       ├── RICE_DATASET/
│       │   ├── RICE1/
│       │   └── RICE2/
│       └── T-Cloud/
└── CDMamba/
    └── CDMamba_patches/   ← 42,873 pre-patched tiles
```

The scratch filesystem has a 10 TB per-user quota and no backup guarantee. All processed datasets were also kept on a local machine.

---

## File Transfer to/from Iris

From macOS (primary workstation):
```bash
rsync -avz -e "ssh -p 8022 -i ~/.ssh/id_ed25519" \
    ./dataset_folder/ \
    jfernandezmartinez@access-iris.uni.lu:/scratch/users/jfernandezmartinez/dataset_folder/
```

From Windows (Git Bash, office laptop):
```bash
scp -r -P 8022 -i ~/.ssh/id_ed25519 \
    ./dataset_folder/ \
    jfernandezmartinez@access-iris.uni.lu:/scratch/users/jfernandezmartinez/dataset_folder/
```

Note: `rsync` is not bundled with Git Bash by default, so `scp` was used on Windows.
