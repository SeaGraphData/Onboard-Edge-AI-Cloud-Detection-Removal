import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score

MASK_DIR   = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/mask")
BINARY_DIR = Path("/scratch/users/jfernandezmartinez/GeoAI/results/masks_binary")
OUT_DIR    = Path("/scratch/users/jfernandezmartinez/GeoAI/results")

with open(OUT_DIR / "efficiency_omnicloudmask.json") as f:
    eff = json.load(f)

binary_files = sorted(BINARY_DIR.glob("*.png"))
print(f"[INFO] Binary masks found: {len(binary_files)}")

per_patch = []

# Acumuladores globales — sin concatenar arrays en memoria
tp_global = 0
tn_global = 0
fp_global = 0
fn_global = 0

for pred_path in tqdm(binary_files, desc="Computing metrics"):
    patch_name = pred_path.stem
    gt_path    = MASK_DIR / (patch_name + ".png")

    if not gt_path.exists():
        print(f"[WARN] GT not found for {patch_name}")
        continue

    gt   = (np.array(Image.open(gt_path)) > 127).astype(np.uint8).flatten()
    pred = (np.array(Image.open(pred_path)) > 127).astype(np.uint8).flatten()

    # Acumuladores globales
    tp_global += int(np.sum((pred == 1) & (gt == 1)))
    tn_global += int(np.sum((pred == 0) & (gt == 0)))
    fp_global += int(np.sum((pred == 1) & (gt == 0)))
    fn_global += int(np.sum((pred == 0) & (gt == 1)))

    # Métricas por patch
    f1  = float(f1_score(gt, pred, zero_division=0))
    pr  = float(precision_score(gt, pred, zero_division=0))
    re  = float(recall_score(gt, pred, zero_division=0))
    acc = float(accuracy_score(gt, pred))
    iou = float(jaccard_score(gt, pred, zero_division=0))

    per_patch.append({
        "patch":     patch_name,
        "f1":        round(f1,  4),
        "precision": round(pr,  4),
        "recall":    round(re,  4),
        "accuracy":  round(acc, 4),
        "iou":       round(iou, 4),
    })

# Métricas globales desde acumuladores — sin concatenar nada
eps = 1e-8
global_f1  = 2*tp_global / (2*tp_global + fp_global + fn_global + eps)
global_pr  = tp_global   / (tp_global + fp_global + eps)
global_re  = tp_global   / (tp_global + fn_global + eps)
global_acc = (tp_global + tn_global) / (tp_global + tn_global + fp_global + fn_global + eps)
global_iou = tp_global   / (tp_global + fp_global + fn_global + eps)

n = len(per_patch)
print(f"\n[INFO] Global metrics computed from accumulators (no memory issue)")
print(f"  TP={tp_global}  TN={tn_global}  FP={fp_global}  FN={fn_global}")

results = {
    "model":   "OmniCloudMask v1.7.1",
    "dataset": "Biome (Landsat-8)",
    "device":  eff.get("device", "cuda"),
    "gpu":     eff.get("gpu", "Tesla V100"),
    "n_patches": n,
    "efficiency_metrics": {
        "gflops":          eff["gflops_total"],
        "params_M":        eff["params_M_total"],
        "avg_latency_ms":  eff["avg_latency_ms"],
        "fps":             eff["fps"],
        "avg_power_w":     eff["avg_power_w"],
        "avg_energy_mj":   eff["avg_energy_mj"],
        "avg_gpu_mem_mb":  eff["avg_gpu_mem_mb"],
        "peak_gpu_mem_mb": eff["peak_gpu_mem_mb"],
    },
    "global_metrics": {
        "f1":        round(float(global_f1),  4),
        "precision": round(float(global_pr),  4),
        "recall":    round(float(global_re),  4),
        "accuracy":  round(float(global_acc), 4),
        "iou":       round(float(global_iou), 4),
    },
    "confusion_matrix": {
        "tp": tp_global, "tn": tn_global,
        "fp": fp_global, "fn": fn_global,
    },
    "per_patch": per_patch,
}

out_json = OUT_DIR / "results_omnicloudmask.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

out_txt = OUT_DIR / "results_omnicloudmask.txt"
with open(out_txt, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("OmniCloudMask — Biome Dataset Evaluation\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Device         : {results['device']}\n")
    f.write(f"GPU            : {results['gpu']}\n")
    f.write(f"Patches        : {n}\n\n")
    f.write("Efficiency Metrics:\n")
    f.write(f"  GFLOPs           : {eff['gflops_total']}\n")
    f.write(f"  Parameters       : {eff['params_M_total']} M\n")
    f.write(f"  Avg latency      : {eff['avg_latency_ms']} ms/patch\n")
    f.write(f"  FPS              : {eff['fps']}\n")
    f.write(f"  Avg power        : {eff['avg_power_w']} W\n")
    f.write(f"  Avg energy       : {eff['avg_energy_mj']} mJ/inference\n")
    f.write(f"  Avg GPU memory   : {eff['avg_gpu_mem_mb']} MB\n")
    f.write(f"  Peak GPU memory  : {eff['peak_gpu_mem_mb']} MB\n\n")
    f.write("Global Metrics (pixel-level, all patches):\n")
    f.write(f"  F1         : {global_f1:.4f}\n")
    f.write(f"  Precision  : {global_pr:.4f}\n")
    f.write(f"  Recall     : {global_re:.4f}\n")
    f.write(f"  Accuracy   : {global_acc:.4f}\n")
    f.write(f"  IoU        : {global_iou:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  TP={tp_global}  TN={tn_global}\n")
    f.write(f"  FP={fp_global}  FN={fn_global}\n")

print(f"\n[DONE]")
print(f"  Patches   : {n}")
print(f"  F1        : {global_f1:.4f}")
print(f"  Precision : {global_pr:.4f}")
print(f"  Recall    : {global_re:.4f}")
print(f"  Accuracy  : {global_acc:.4f}")
print(f"  IoU       : {global_iou:.4f}")
print(f"  Saved to  : {out_json}")
