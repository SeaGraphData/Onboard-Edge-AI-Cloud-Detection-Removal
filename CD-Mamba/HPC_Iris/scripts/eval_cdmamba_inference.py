"""
CD-Mamba — Inference, Segmentation Metrics & Visualization Script
==================================================================
Runs 4-fold cross-validation evaluation using the pretrained checkpoints:

    cdm_01_0.82835.pth  →  tested on split/test/patch_in_123.txt
    cdm_04_0.86710.pth  →  tested on split/test/patch_in_456.txt
    cdm_07_0.89102.pth  →  tested on split/test/patch_in_789.txt
    cdm_10_0.89449.pth  →  tested on split/test/patch_in_101112.txt

Segmentation metrics computed:
    Accuracy, F1 / Dice, IoU (Jaccard), Precision, Recall
    Confusion Matrix (TP, TN, FP, FN)

Visualizations saved per fold (VIS_PER_FOLD random patches):
    output/cdmamba-vis/{fold}/{patch_id}/1_rgb_input.png
    output/cdmamba-vis/{fold}/{patch_id}/2_pred_mask.png
    output/cdmamba-vis/{fold}/{patch_id}/3_gt_mask.png
    output/cdmamba-vis/{fold}/{patch_id}/4_overlay.png
    output/cdmamba-vis/summary_grid_{fold}.png
    output/cdmamba-vis/confusion_matrix_{fold}.png

Global outputs:
    output/cdmamba-vis/confusion_matrix_all.png
    output/cdmamba-vis/summary_grid_all.png
    output/cdmamba-vis/cdmamba_metrics_{timestamp}.json
    output/cdmamba-vis/cdmamba_metrics_{timestamp}.txt

Usage:
    python eval_cdmamba_inference.py
"""

import os
import sys
import json
import random
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imread

# ── Model import ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'f01'))
from models.cloud import cdMamba

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(SCRIPT_DIR, 'dataset', 'biome')
SPLIT_DIR    = os.path.join(SCRIPT_DIR, 'split', 'test')
PT_DIR       = os.path.join(SCRIPT_DIR, 'pt_models')
VIS_DIR      = os.path.join(SCRIPT_DIR, 'output', 'cdmamba-vis')
os.makedirs(VIS_DIR, exist_ok=True)

THRESHOLD    = 0.5
VIS_PER_FOLD = 15     # patches to visualize per fold
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

MODEL_CONFIG = dict(
    num_classes    = 1,
    input_channels = 4,
    c_list         = [8, 16, 24, 32, 48, 64],
    split_att      = 'fc',
    bridge         = True,
)

# 4-fold setup: each checkpoint tests on its corresponding split file
FOLDS = [
    {'name': 'fold_01', 'ckpt': 'cdm_01_0.82835.pth',  'split': 'patch_in_123.txt'},
    {'name': 'fold_04', 'ckpt': 'cdm_04_0.86710.pth',  'split': 'patch_in_456.txt'},
    {'name': 'fold_07', 'ckpt': 'cdm_07_0.89102.pth',  'split': 'patch_in_789.txt'},
    {'name': 'fold_10', 'ckpt': 'cdm_10_0.89449.pth',  'split': 'patch_in_101112.txt'},
]

device = torch.device('cuda')


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_model(ckpt_name: str) -> torch.nn.Module:
    model = cdMamba(**MODEL_CONFIG).to(device).eval()
    state_dict = torch.load(os.path.join(PT_DIR, ckpt_name), map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def read_patch(patch_id: str) -> np.ndarray:
    """Read 4-channel patch (R, G, B, NIR) → float32 array (4, H, W)."""
    bands = []
    for folder in ['red', 'green', 'blue', 'nir']:
        path = os.path.join(DATASET_ROOT, folder, patch_id + '.TIF')
        band = imread(path).astype(np.float32)
        bands.append(band)
    return np.stack(bands, axis=0)          # (4, H, W)


def read_gt_mask(patch_id: str) -> np.ndarray:
    """Read ground-truth mask → float32 (H, W) in [0, 1]."""
    path = os.path.join(DATASET_ROOT, 'mask', patch_id + '.png')
    return imread(path).astype(np.float32) / 255.0


# ── Visualization helpers ─────────────────────────────────────────────────────
def percentile_stretch(band: np.ndarray, p_lo=2, p_hi=98) -> np.ndarray:
    """Stretch a float band to uint8 using percentile clipping."""
    lo = np.percentile(band, p_lo)
    hi = np.percentile(band, p_hi)
    out = np.clip((band - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def make_rgb(img_4ch: np.ndarray) -> np.ndarray:
    """Build a uint8 (H, W, 3) RGB image from channels [R, G, B, NIR]."""
    r = percentile_stretch(img_4ch[0])
    g = percentile_stretch(img_4ch[1])
    b = percentile_stretch(img_4ch[2])
    return np.stack([r, g, b], axis=-1)


def make_overlay(rgb: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
    """Highlight predicted cloud pixels in red on top of the RGB image."""
    overlay = rgb.copy()
    cloud   = pred_bin.astype(bool)
    overlay[cloud, 0] = 220   # R channel → full red
    overlay[cloud, 1] = 30    # G channel → dark
    overlay[cloud, 2] = 30    # B channel → dark
    return overlay


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(TP: int, TN: int, FP: int, FN: int) -> dict:
    eps = 1e-8
    accuracy  = (TP + TN)                    / (TP + TN + FP + FN + eps)
    precision = TP                           / (TP + FP + eps)
    recall    = TP                           / (TP + FN + eps)
    f1        = 2 * precision * recall       / (precision + recall + eps)
    iou       = TP                           / (TP + FP + FN + eps)
    return dict(
        accuracy  = float(accuracy),
        precision = float(precision),
        recall    = float(recall),
        f1        = float(f1),
        iou       = float(iou),
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────
def save_confusion_matrix(TP: int, TN: int, FP: int, FN: int,
                           title: str, save_path: str) -> None:
    cm  = np.array([[TN, FP],
                    [FN, TP]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    labels = ['No Cloud', 'Cloud']
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color=color, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def save_summary_grid(records: list, title: str, save_path: str,
                      max_rows: int = 20) -> None:
    """
    Each record: dict with keys patch_id, rgb, pred, gt, overlay.
    Grid layout: rows = patches, cols = [RGB, Predicted, GT, Overlay].
    """
    records = records[:max_rows]
    n = len(records)
    if n == 0:
        return

    col_titles = ['RGB Input (B4/B3/B2)', 'Predicted Mask',
                  'Ground Truth', 'Overlay (pred in red)']
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]       # ensure 2-D indexing

    for col_idx, ct in enumerate(col_titles):
        axes[0, col_idx].set_title(ct, fontsize=10, fontweight='bold')

    for row_idx, rec in enumerate(records):
        # Shorten patch ID for the y-label
        label = rec['patch_id']
        label = label[:label.find('_LC8')] + '\n' + label[label.find('_LC8') + 1:]

        axes[row_idx, 0].imshow(rec['rgb'])
        axes[row_idx, 0].set_ylabel(label, fontsize=6, rotation=0,
                                     labelpad=80, va='center')
        axes[row_idx, 1].imshow(rec['pred'],   cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].imshow(rec['gt'],     cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 3].imshow(rec['overlay'])

        for ax in axes[row_idx]:
            ax.axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Summary grid saved:     {save_path}")


# ── Main evaluation loop ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CD-Mamba — INFERENCE, METRICS & VISUALIZATION")
print("=" * 65)
print(f"  Dataset:   {DATASET_ROOT}")
print(f"  Device:    {torch.cuda.get_device_name(0)}")
print(f"  Threshold: {THRESHOLD}")
print("=" * 65)

global_TP = global_TN = global_FP = global_FN = 0
all_fold_results  = []
all_vis_records   = []     # up to VIS_PER_FOLD × 4 records for global grid

for fold in FOLDS:
    fold_name  = fold['name']
    split_file = fold['split']
    ckpt_name  = fold['ckpt']
    split_path = os.path.join(SPLIT_DIR, split_file)

    print(f"\n{'─' * 65}")
    print(f"  Fold      : {fold_name}")
    print(f"  Checkpoint: {ckpt_name}")
    print(f"  Split     : {split_file}")
    print(f"{'─' * 65}")

    # ── Load patch list ────────────────────────────────────────────────────
    with open(split_path, 'r') as f:
        patch_ids = [line.strip() for line in f if line.strip()]
    print(f"  Patches in split : {len(patch_ids)}")

    # Filter to patches that actually exist on disk (band TIFs + mask PNG)
    valid_ids = [
        pid for pid in patch_ids
        if (os.path.exists(os.path.join(DATASET_ROOT, 'red',  pid + '.TIF')) and
            os.path.exists(os.path.join(DATASET_ROOT, 'mask', pid + '.png')))
    ]
    print(f"  Patches on disk  : {len(valid_ids)} / {len(patch_ids)}")

    if len(valid_ids) == 0:
        print("  [SKIP] No valid patches found — run patch_tif_bands.py first.")
        continue

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model(ckpt_name)
    print(f"  Model loaded.")

    # ── Select visualization subset ────────────────────────────────────────
    vis_ids    = set(random.sample(valid_ids, min(VIS_PER_FOLD, len(valid_ids))))
    fold_vis   = []
    fold_dir   = os.path.join(VIS_DIR, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    # ── Inference loop ─────────────────────────────────────────────────────
    TP = TN = FP = FN = 0
    log_interval = max(1, len(valid_ids) // 10)

    for idx, patch_id in enumerate(valid_ids):
        if (idx + 1) % log_interval == 0 or (idx + 1) == len(valid_ids):
            print(f"    {idx + 1:5d} / {len(valid_ids)}  "
                  f"(F1 so far: {2*TP/(2*TP+FP+FN+1e-8):.4f})", end='\r')

        try:
            img_np = read_patch(patch_id)      # (4, H, W) float32
            gt_np  = read_gt_mask(patch_id)    # (H, W) float32 [0,1]
        except Exception as e:
            print(f"\n    [WARN] Skipping {patch_id}: {e}")
            continue

        # Forward pass
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)   # (1,4,H,W)
        with torch.no_grad():
            out = model(img_t)                                       # (1,1,H,W)
        pred_prob = out.squeeze().cpu().numpy()                      # (H,W) [0,1]
        pred_bin  = (pred_prob >= THRESHOLD).astype(np.uint8)        # (H,W) binary
        gt_bin    = (gt_np     >= 0.5      ).astype(np.uint8)

        # Accumulate confusion matrix components
        TP += int((pred_bin * gt_bin                 ).sum())
        TN += int(((1 - pred_bin) * (1 - gt_bin)    ).sum())
        FP += int((pred_bin * (1 - gt_bin)           ).sum())
        FN += int(((1 - pred_bin) * gt_bin           ).sum())

        # ── Visualization (selected patches only) ──────────────────────────
        if patch_id in vis_ids:
            rgb     = make_rgb(img_np)
            overlay = make_overlay(rgb, pred_bin)

            # Save individual 4-panel images
            patch_out_dir = os.path.join(fold_dir, patch_id)
            os.makedirs(patch_out_dir, exist_ok=True)
            plt.imsave(os.path.join(patch_out_dir, '1_rgb_input.png'),  rgb)
            plt.imsave(os.path.join(patch_out_dir, '2_pred_mask.png'),  pred_bin,
                       cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(patch_out_dir, '3_gt_mask.png'),    gt_bin,
                       cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(patch_out_dir, '4_overlay.png'),    overlay)

            fold_vis.append(dict(
                patch_id = patch_id,
                rgb      = rgb,
                pred     = pred_bin,
                gt       = gt_bin,
                overlay  = overlay,
            ))

    print()   # newline after \r progress line

    # ── Fold metrics ───────────────────────────────────────────────────────
    fold_metrics = compute_metrics(TP, TN, FP, FN)
    fold_result  = {
        'fold':       fold_name,
        'checkpoint': ckpt_name,
        'n_patches':  len(valid_ids),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        **fold_metrics,
    }
    all_fold_results.append(fold_result)
    all_vis_records.extend(fold_vis)

    # Accumulate global confusion matrix
    global_TP += TP; global_TN += TN; global_FP += FP; global_FN += FN

    print(f"\n  ── {fold_name} Results ──────────────────────────────────")
    print(f"  Accuracy  : {fold_metrics['accuracy']:.4f}")
    print(f"  F1 / Dice : {fold_metrics['f1']:.4f}")
    print(f"  IoU       : {fold_metrics['iou']:.4f}")
    print(f"  Precision : {fold_metrics['precision']:.4f}")
    print(f"  Recall    : {fold_metrics['recall']:.4f}")
    print(f"  TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")

    # Per-fold plots
    save_confusion_matrix(
        TP, TN, FP, FN,
        title     = f'Confusion Matrix — {fold_name}',
        save_path = os.path.join(VIS_DIR, f'confusion_matrix_{fold_name}.png'),
    )
    save_summary_grid(
        fold_vis,
        title     = f'CD-Mamba — {fold_name}  ({split_file})',
        save_path = os.path.join(VIS_DIR, f'summary_grid_{fold_name}.png'),
    )

    del model
    torch.cuda.empty_cache()


# ── Global / overall results ──────────────────────────────────────────────────
global_metrics = compute_metrics(global_TP, global_TN, global_FP, global_FN)

sep = "=" * 65
print(f"\n{sep}")
print("  CD-Mamba — 4-FOLD CROSS-VALIDATION OVERALL SUMMARY")
print(sep)
print(f"  {'Fold':<12}  {'F1':>8}  {'IoU':>8}  {'Acc':>8}  "
      f"{'Prec':>8}  {'Recall':>8}  {'#Patches':>9}")
print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
for r in all_fold_results:
    print(f"  {r['fold']:<12}  {r['f1']:>8.4f}  {r['iou']:>8.4f}  "
          f"{r['accuracy']:>8.4f}  {r['precision']:>8.4f}  "
          f"{r['recall']:>8.4f}  {r['n_patches']:>9,}")
print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
total_patches = sum(r['n_patches'] for r in all_fold_results)
print(f"  {'OVERALL':<12}  {global_metrics['f1']:>8.4f}  "
      f"{global_metrics['iou']:>8.4f}  {global_metrics['accuracy']:>8.4f}  "
      f"{global_metrics['precision']:>8.4f}  {global_metrics['recall']:>8.4f}  "
      f"{total_patches:>9,}")
print(sep)

# Global plots — cap the summary grid at 20 patches to keep file manageable
save_confusion_matrix(
    global_TP, global_TN, global_FP, global_FN,
    title     = 'Confusion Matrix — All Folds (CD-Mamba)',
    save_path = os.path.join(VIS_DIR, 'confusion_matrix_all.png'),
)
save_summary_grid(
    all_vis_records,
    title     = 'CD-Mamba Predictions — All Folds',
    save_path = os.path.join(VIS_DIR, 'summary_grid_all.png'),
    max_rows  = 20,
)

# ── Save JSON + TXT ───────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
now_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

output_data = {
    'model':     'CD-Mamba',
    'timestamp': now_str,
    'threshold': THRESHOLD,
    'per_fold':  all_fold_results,
    'overall':   {**global_metrics,
                  'TP': global_TP, 'TN': global_TN,
                  'FP': global_FP, 'FN': global_FN},
}

json_path = os.path.join(VIS_DIR, f'cdmamba_metrics_{timestamp}.json')
txt_path  = os.path.join(VIS_DIR, f'cdmamba_metrics_{timestamp}.txt')

with open(json_path, 'w') as f:
    json.dump(output_data, f, indent=4)

with open(txt_path, 'w') as f:
    f.write("CD-Mamba SEGMENTATION METRICS — 4-FOLD CROSS-VALIDATION\n")
    f.write(f"Timestamp : {now_str}\n")
    f.write(f"Threshold : {THRESHOLD}\n\n")
    f.write("--- Per-Fold Results ---\n")
    for r in all_fold_results:
        f.write(f"\n  {r['fold']}  ({r['checkpoint']})\n")
        f.write(f"  Patches evaluated : {r['n_patches']:,}\n")
        f.write(f"  Accuracy          : {r['accuracy']:.4f}\n")
        f.write(f"  F1 / Dice         : {r['f1']:.4f}\n")
        f.write(f"  IoU (Jaccard)     : {r['iou']:.4f}\n")
        f.write(f"  Precision         : {r['precision']:.4f}\n")
        f.write(f"  Recall            : {r['recall']:.4f}\n")
        f.write(f"  TP={r['TP']:,}  TN={r['TN']:,}  FP={r['FP']:,}  FN={r['FN']:,}\n")
    f.write("\n--- Overall (All Folds) ---\n")
    f.write(f"  Accuracy          : {global_metrics['accuracy']:.4f}\n")
    f.write(f"  F1 / Dice         : {global_metrics['f1']:.4f}\n")
    f.write(f"  IoU (Jaccard)     : {global_metrics['iou']:.4f}\n")
    f.write(f"  Precision         : {global_metrics['precision']:.4f}\n")
    f.write(f"  Recall            : {global_metrics['recall']:.4f}\n")
    f.write(f"  TP={global_TP:,}  TN={global_TN:,}  FP={global_FP:,}  FN={global_FN:,}\n")

print(f"\n  Results saved:")
print(f"    {json_path}")
print(f"    {txt_path}")
print(f"\n  All visualizations in: {VIS_DIR}")
