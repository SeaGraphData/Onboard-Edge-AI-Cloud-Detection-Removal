"""
CD-Mamba — Segmentation Metrics Script (Jetson)
=================================================
Computes: Accuracy, F1/Dice, IoU, Precision, Recall, Confusion Matrix
4-fold cross-validation using the 4 pre-trained checkpoints.
Also saves per-patch visualisations (RGB, predicted mask, GT, overlay).

Usage:
    conda activate cdmamba
    cd ~/Desktop/CDMamba
    python eval_cdmamba_seg_jetson.py
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
DATASET_ROOT = os.path.join(os.path.expanduser('~'), 'Desktop',
                             'CDMamba', 'CDMamba_patches')
SPLIT_DIR    = os.path.join(SCRIPT_DIR, 'split', 'test')
PT_DIR       = os.path.join(SCRIPT_DIR, 'pt_models')
OUTPUT_DIR   = os.path.join(os.path.expanduser('~'), 'Desktop',
                             'cdmamba_jetson_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD    = 0.5
VIS_PER_FOLD = 10
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

MODEL_CONFIG = dict(
    num_classes    = 1,
    input_channels = 4,
    c_list         = [8, 16, 24, 32, 48, 64],
    split_att      = 'fc',
    bridge         = True,
)

FOLDS = [
    {'name': 'fold_01', 'ckpt': 'cdm_01_0.82835.pth',  'split': 'patch_in_123.txt'},
    {'name': 'fold_04', 'ckpt': 'cdm_04_0.86710.pth',  'split': 'patch_in_456.txt'},
    {'name': 'fold_07', 'ckpt': 'cdm_07_0.89102.pth',  'split': 'patch_in_789.txt'},
    {'name': 'fold_10', 'ckpt': 'cdm_10_0.89449.pth',  'split': 'patch_in_101112.txt'},
]

device = torch.device('cuda')


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(ckpt_name):
    model = cdMamba(**MODEL_CONFIG).to(device).eval()
    state_dict = torch.load(os.path.join(PT_DIR, ckpt_name), map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def read_patch(pid):
    bands = [imread(os.path.join(DATASET_ROOT, f, pid + '.TIF')).astype(np.float32)
             for f in ['red', 'green', 'blue', 'nir']]
    return np.stack(bands, axis=0)   # (4, H, W)


def read_gt(pid):
    return (imread(os.path.join(DATASET_ROOT, 'mask', pid + '.png'))
            .astype(np.float32) / 255.0)


def stretch(band, p_lo=2, p_hi=98):
    lo, hi = np.percentile(band, p_lo), np.percentile(band, p_hi)
    return (np.clip((band - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)


def make_rgb(img):
    return np.stack([stretch(img[0]), stretch(img[1]), stretch(img[2])], axis=-1)


def make_overlay(rgb, pred):
    ov = rgb.copy()
    ov[pred.astype(bool), 0] = 220
    ov[pred.astype(bool), 1] = 30
    ov[pred.astype(bool), 2] = 30
    return ov


def compute_metrics(TP, TN, FP, FN):
    e = 1e-8
    return dict(
        accuracy  = float((TP + TN) / (TP + TN + FP + FN + e)),
        precision = float(TP / (TP + FP + e)),
        recall    = float(TP / (TP + FN + e)),
        f1        = float(2 * TP / (2 * TP + FP + FN + e)),
        iou       = float(TP / (TP + FP + FN + e)),
    )


def save_confusion_matrix(TP, TN, FP, FN, title, path):
    cm = np.array([[TN, FP], [FN, TP]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color=color, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Cloud', 'Cloud'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No Cloud', 'Cloud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_grid(records, title, path, max_rows=20):
    records = records[:max_rows]
    if not records:
        return
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for ci, ct in enumerate(['RGB Input', 'Predicted Mask',
                              'Ground Truth', 'Overlay']):
        axes[0, ci].set_title(ct, fontsize=9, fontweight='bold')
    for ri, rec in enumerate(records):
        label = rec['patch_id']
        axes[ri, 0].imshow(rec['rgb'])
        axes[ri, 0].set_ylabel(label[:label.find('_LC8')] + '\n' +
                                label[label.find('_LC8')+1:],
                                fontsize=5, rotation=0, labelpad=80, va='center')
        axes[ri, 1].imshow(rec['pred'],    cmap='gray', vmin=0, vmax=1)
        axes[ri, 2].imshow(rec['gt'],      cmap='gray', vmin=0, vmax=1)
        axes[ri, 3].imshow(rec['overlay'])
        for ax in axes[ri]:
            ax.axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Grid saved: {path}")


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CD-Mamba — SEGMENTATION EVALUATION (Jetson)")
print("=" * 60)
print(f"  Dataset : {DATASET_ROOT}")
print(f"  Output  : {OUTPUT_DIR}")
print("=" * 60)

global_TP = global_TN = global_FP = global_FN = 0
all_fold_results = []
all_vis          = []

for fold in FOLDS:
    fold_name  = fold['name']
    split_path = os.path.join(SPLIT_DIR, fold['split'])

    with open(split_path, 'r') as f:
        patch_ids = [l.strip() for l in f if l.strip()]

    valid_ids = [
        pid for pid in patch_ids
        if (os.path.exists(os.path.join(DATASET_ROOT, 'red',  pid + '.TIF')) and
            os.path.exists(os.path.join(DATASET_ROOT, 'mask', pid + '.png')))
    ]

    print(f"\n  {'─'*58}")
    print(f"  {fold_name}  —  {fold['ckpt']}")
    print(f"  Patches in split: {len(patch_ids)}  |  Found on disk: {len(valid_ids)}")

    if not valid_ids:
        print("  [SKIP] No valid patches found.")
        continue

    model   = load_model(fold['ckpt'])
    vis_ids = set(random.sample(valid_ids, min(VIS_PER_FOLD, len(valid_ids))))
    fold_dir = os.path.join(OUTPUT_DIR, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    TP = TN = FP = FN = 0
    fold_vis  = []
    log_every = max(1, len(valid_ids) // 10)

    for idx, pid in enumerate(valid_ids):
        if (idx + 1) % log_every == 0 or (idx + 1) == len(valid_ids):
            f1_now = 2 * TP / (2 * TP + FP + FN + 1e-8)
            print(f"    {idx+1:5d}/{len(valid_ids)}  F1: {f1_now:.4f}", end='\r')

        try:
            img_np = read_patch(pid)
            gt_np  = read_gt(pid)
        except Exception as exc:
            print(f"\n    [WARN] {pid}: {exc}")
            continue

        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
        pred_bin = (out.squeeze().cpu().numpy() >= THRESHOLD).astype(np.uint8)
        gt_bin   = (gt_np >= 0.5).astype(np.uint8)

        TP += int((pred_bin *       gt_bin ).sum())
        TN += int(((1-pred_bin) * (1-gt_bin)).sum())
        FP += int((pred_bin *   (1-gt_bin) ).sum())
        FN += int(((1-pred_bin) *   gt_bin ).sum())

        if pid in vis_ids:
            rgb = make_rgb(img_np)
            ov  = make_overlay(rgb, pred_bin)
            patch_out = os.path.join(fold_dir, pid)
            os.makedirs(patch_out, exist_ok=True)
            plt.imsave(os.path.join(patch_out, '1_rgb_input.png'),  rgb)
            plt.imsave(os.path.join(patch_out, '2_pred_mask.png'),  pred_bin,
                       cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(patch_out, '3_gt_mask.png'),    gt_bin,
                       cmap='gray', vmin=0, vmax=1)
            plt.imsave(os.path.join(patch_out, '4_overlay.png'),    ov)
            fold_vis.append(dict(patch_id=pid, rgb=rgb,
                                 pred=pred_bin, gt=gt_bin, overlay=ov))
    print()

    seg = compute_metrics(TP, TN, FP, FN)
    all_fold_results.append({
        'fold': fold_name, 'checkpoint': fold['ckpt'],
        'n_patches': len(valid_ids),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        **seg,
    })
    global_TP += TP; global_TN += TN
    global_FP += FP; global_FN += FN
    all_vis.extend(fold_vis)

    print(f"  Accuracy : {seg['accuracy']:.4f}  |  F1: {seg['f1']:.4f}"
          f"  |  IoU: {seg['iou']:.4f}")
    print(f"  Precision: {seg['precision']:.4f}  |  Recall: {seg['recall']:.4f}")

    save_confusion_matrix(TP, TN, FP, FN,
        title=f'Confusion Matrix — {fold_name} (Jetson)',
        path=os.path.join(OUTPUT_DIR, f'confusion_matrix_{fold_name}.png'))
    save_summary_grid(fold_vis,
        title=f'CD-Mamba — {fold_name} (Jetson)',
        path=os.path.join(OUTPUT_DIR, f'summary_grid_{fold_name}.png'))

    del model
    torch.cuda.empty_cache()

# ── Global metrics ────────────────────────────────────────────────────────────
global_seg = compute_metrics(global_TP, global_TN, global_FP, global_FN)
save_confusion_matrix(global_TP, global_TN, global_FP, global_FN,
    title='Confusion Matrix — All Folds (Jetson)',
    path=os.path.join(OUTPUT_DIR, 'confusion_matrix_all.png'))
save_summary_grid(all_vis,
    title='CD-Mamba — All Folds (Jetson)',
    path=os.path.join(OUTPUT_DIR, 'summary_grid_all.png'), max_rows=20)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  OVERALL RESULTS — 4-FOLD CROSS-VALIDATION")
print("=" * 60)
print(f"  {'Fold':<12} {'F1':>8} {'IoU':>8} {'Acc':>8} "
      f"{'Prec':>8} {'Recall':>8}")
print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for r in all_fold_results:
    print(f"  {r['fold']:<12} {r['f1']:>8.4f} {r['iou']:>8.4f} "
          f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}")
print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
print(f"  {'OVERALL':<12} {global_seg['f1']:>8.4f} {global_seg['iou']:>8.4f} "
      f"{global_seg['accuracy']:>8.4f} {global_seg['precision']:>8.4f} "
      f"{global_seg['recall']:>8.4f}")
print("=" * 60)

# ── Save results ──────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
now_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data = {
    'model': 'CD-Mamba', 'device': 'Jetson Orin Nano',
    'timestamp': now_str, 'threshold': THRESHOLD,
    'per_fold': all_fold_results,
    'overall': {**global_seg,
                'TP': global_TP, 'TN': global_TN,
                'FP': global_FP, 'FN': global_FN},
}

json_path = os.path.join(OUTPUT_DIR, f'cdmamba_seg_jetson_{timestamp}.json')
txt_path  = os.path.join(OUTPUT_DIR, f'cdmamba_seg_jetson_{timestamp}.txt')

with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

with open(txt_path, 'w') as f:
    f.write("CD-Mamba SEGMENTATION METRICS — Jetson Orin Nano\n")
    f.write(f"Timestamp : {now_str}\n")
    f.write(f"Threshold : {THRESHOLD}\n\n")
    f.write("--- Per-Fold Results ---\n")
    for r in all_fold_results:
        f.write(f"\n  {r['fold']} ({r['checkpoint']})\n")
        f.write(f"  Patches   : {r['n_patches']:,}\n")
        f.write(f"  Accuracy  : {r['accuracy']:.4f}\n")
        f.write(f"  F1 / Dice : {r['f1']:.4f}\n")
        f.write(f"  IoU       : {r['iou']:.4f}\n")
        f.write(f"  Precision : {r['precision']:.4f}\n")
        f.write(f"  Recall    : {r['recall']:.4f}\n")
        f.write(f"  TP={r['TP']:,}  TN={r['TN']:,}  "
                f"FP={r['FP']:,}  FN={r['FN']:,}\n")
    f.write("\n--- Overall ---\n")
    f.write(f"  Accuracy  : {global_seg['accuracy']:.4f}\n")
    f.write(f"  F1 / Dice : {global_seg['f1']:.4f}\n")
    f.write(f"  IoU       : {global_seg['iou']:.4f}\n")
    f.write(f"  Precision : {global_seg['precision']:.4f}\n")
    f.write(f"  Recall    : {global_seg['recall']:.4f}\n")
    f.write(f"  TP={global_TP:,}  TN={global_TN:,}  "
            f"FP={global_FP:,}  FN={global_FN:,}\n")

print(f"\n  Results : {json_path}")
print(f"           {txt_path}")
print(f"  Visuals : {OUTPUT_DIR}")
