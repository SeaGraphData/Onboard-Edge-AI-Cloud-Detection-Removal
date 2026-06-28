import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, jaccard_score, cohen_kappa_score,
    confusion_matrix
)
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
CDMAMBA_VIS  = Path("/scratch/users/jfernandezmartinez/CDMamba/output/cdmamba-vis")
GEOAI_4CLASS = Path("/scratch/users/jfernandezmartinez/GeoAI/results/masks_4class")
GEOAI_BINARY = Path("/scratch/users/jfernandezmartinez/GeoAI/results/masks_binary")
BIOME_MASK   = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/mask")
BIOME_RED    = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/red")
BIOME_GREEN  = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/green")
BIOME_BLUE   = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/blue")
BIOME_NIR    = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome/nir")

OUT_DIR      = Path("/scratch/users/jfernandezmartinez/GeoAI/results/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for fold in ["fold_01", "fold_04", "fold_07", "fold_10"]:
    (OUT_DIR / fold).mkdir(exist_ok=True)

FOLDS = ["fold_01", "fold_04", "fold_07", "fold_10"]

# ── OmniCloudMask 4-class color → index ───────────────────────────────────────
COLOR_TO_IDX = {
    (0,   180,   0): 0,
    (255, 255, 255): 1,
    (180, 180, 180): 2,
    (30,   30, 100): 3,
}
CLASS_NAMES  = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]
tab10        = plt.get_cmap("tab10")
ocm_legend   = [mpatches.Patch(color=tab10(i/9), label=CLASS_NAMES[i]) for i in range(4)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_binary_png(path):
    """Carga PNG binario (0/255) y devuelve array 0/1."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 127).astype(np.uint8)

def load_rgba_as_rgb(path):
    """Carga PNG RGBA y devuelve float RGB [0,1]."""
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return arr

def load_band(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def stretch(arr):
    p2, p98 = np.percentile(arr, (2, 98))
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

def make_true_color(red, green, blue):
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, None)
    rgb = np.power(rgb, 0.5)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)

def make_false_color(nir, red, green):
    return np.clip(np.stack([stretch(nir), stretch(red), stretch(green)], axis=-1), 0, 1)

def load_ocm_4class(path):
    """Carga PNG de 4 clases OmniCloudMask y devuelve índices 0-3."""
    mask_rgb = np.array(Image.open(path).convert("RGB"))
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, idx in COLOR_TO_IDX.items():
        mask_idx[np.all(mask_rgb == np.array(color), axis=-1)] = idx
    return mask_idx

def make_ocm_overlay(true_color, mask_idx):
    rgba = np.zeros((*mask_idx.shape, 4), dtype=float)
    for cls in range(4):
        colour = tab10(cls / 9)
        where  = mask_idx == cls
        rgba[where, :3] = colour[:3]
        rgba[where,  3] = 0.0 if cls == 0 else 0.4
    return rgba

def make_error_map(pred, gt):
    """
    Mapa de error CD-Mamba o OmniCloud vs GT.
    Verde=TP, Azul claro=TN, Rojo=FP, Naranja=FN
    """
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[(pred == 1) & (gt == 1)] = [0.0, 0.8, 0.0]   # TP verde
    rgb[(pred == 0) & (gt == 0)] = [0.8, 0.9, 1.0]   # TN azul claro
    rgb[(pred == 1) & (gt == 0)] = [0.9, 0.1, 0.1]   # FP rojo
    rgb[(pred == 0) & (gt == 1)] = [1.0, 0.6, 0.0]   # FN naranja
    return rgb

def make_diff_map(pred_cd, pred_ocm, gt):
    """
    Mapa de diferencias entre modelos.
    Verde:   ambos aciertan
    Azul:    CD-Mamba detecta, OmniCloud no (y GT=nube)
    Naranja: OmniCloud detecta, CD-Mamba no (y GT=nube)
    Rojo:    ambos fallan
    Gris:    ambos aciertan TN
    """
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    both_correct_cloud = (pred_cd == 1) & (pred_ocm == 1) & (gt == 1)
    both_correct_clear = (pred_cd == 0) & (pred_ocm == 0) & (gt == 0)
    cd_only            = (pred_cd == 1) & (pred_ocm == 0) & (gt == 1)
    ocm_only           = (pred_cd == 0) & (pred_ocm == 1) & (gt == 1)
    both_fail          = ((pred_cd != gt) & (pred_ocm != gt))
    rgb[both_correct_cloud] = [0.0, 0.8, 0.0]   # verde
    rgb[both_correct_clear] = [0.85, 0.9, 0.95] # gris claro
    rgb[cd_only]            = [0.2, 0.4, 0.9]   # azul
    rgb[ocm_only]           = [1.0, 0.6, 0.0]   # naranja
    rgb[both_fail]          = [0.9, 0.1, 0.1]   # rojo
    return rgb

def compute_metrics(pred, gt):
    p = pred.flatten()
    g = gt.flatten()
    tp = int(np.sum((p == 1) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    f1  = float(f1_score(g, p, zero_division=0))
    pr  = float(precision_score(g, p, zero_division=0))
    re  = float(recall_score(g, p, zero_division=0))
    acc = float(accuracy_score(g, p))
    iou = float(jaccard_score(g, p, zero_division=0))
    kap = float(cohen_kappa_score(g, p)) if len(np.unique(g)) > 1 else 0.0
    far = fp / (fp + tn + 1e-8)   # False Alarm Rate
    mis = fn / (fn + tp + 1e-8)   # Miss Rate
    csi = tp / (tp + fn + fp + 1e-8)  # Critical Success Index
    dic = 2*tp / (2*tp + fp + fn + 1e-8)  # Dice
    return {
        "f1": round(f1, 4), "precision": round(pr, 4),
        "recall": round(re, 4), "accuracy": round(acc, 4),
        "iou": round(iou, 4), "kappa": round(kap, 4),
        "false_alarm_rate": round(far, 4), "miss_rate": round(mis, 4),
        "csi": round(csi, 4), "dice": round(dic, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def pixel_agreement(pred_cd, pred_ocm):
    return float(np.mean(pred_cd == pred_ocm))

# ── Leyendas para mapas ────────────────────────────────────────────────────────
error_legend = [
    mpatches.Patch(color=[0.0,0.8,0.0], label="TP (correct cloud)"),
    mpatches.Patch(color=[0.8,0.9,1.0], label="TN (correct clear)"),
    mpatches.Patch(color=[0.9,0.1,0.1], label="FP (false alarm)"),
    mpatches.Patch(color=[1.0,0.6,0.0], label="FN (missed cloud)"),
]
diff_legend = [
    mpatches.Patch(color=[0.0,0.8,0.0],  label="Both correct (cloud)"),
    mpatches.Patch(color=[0.85,0.9,0.95],label="Both correct (clear)"),
    mpatches.Patch(color=[0.2,0.4,0.9],  label="CD-Mamba only"),
    mpatches.Patch(color=[1.0,0.6,0.0],  label="OmniCloud only"),
    mpatches.Patch(color=[0.9,0.1,0.1],  label="Both fail"),
]

# ── Loop principal ─────────────────────────────────────────────────────────────
all_results = []
missing_ocm = []

for fold in FOLDS:
    fold_dir = CDMAMBA_VIS / fold
    patches  = sorted([p for p in fold_dir.iterdir() if p.is_dir()])
    print(f"\n[{fold}] {len(patches)} patches")

    for patch_dir in tqdm(patches, desc=fold):
        patch_name = patch_dir.name

        # ── Ficheros CD-Mamba ──────────────────────────────────────────────────
        cd_rgb_path     = patch_dir / "1_rgb_input.png"
        cd_pred_path    = patch_dir / "2_pred_mask.png"
        cd_gt_path      = patch_dir / "3_gt_mask.png"
        cd_overlay_path = patch_dir / "4_overlay.png"

        # ── Ficheros OmniCloudMask ─────────────────────────────────────────────
        ocm_4class_path = GEOAI_4CLASS / (patch_name + ".png")
        ocm_binary_path = GEOAI_BINARY / (patch_name + ".png")

        if not ocm_4class_path.exists():
            print(f"  [WARN] OmniCloud mask not found: {patch_name}")
            missing_ocm.append(patch_name)
            continue

        # ── Cargar CD-Mamba ────────────────────────────────────────────────────
        cd_rgb     = load_rgba_as_rgb(cd_rgb_path)
        cd_pred    = load_binary_png(cd_pred_path)
        cd_gt      = load_binary_png(cd_gt_path)
        cd_overlay = load_rgba_as_rgb(cd_overlay_path)

        # ── Cargar OmniCloudMask ───────────────────────────────────────────────
        ocm_4class = load_ocm_4class(ocm_4class_path)
        ocm_binary = load_binary_png(ocm_binary_path)

        # ── Cargar bandas para True/False color ───────────────────────────────
        red_path  = BIOME_RED   / (patch_name + ".TIF")
        grn_path  = BIOME_GREEN / (patch_name + ".TIF")
        blu_path  = BIOME_BLUE  / (patch_name + ".TIF")
        nir_path  = BIOME_NIR   / (patch_name + ".TIF")

        if all(p.exists() for p in [red_path, grn_path, blu_path, nir_path]):
            red = load_band(red_path)
            grn = load_band(grn_path)
            blu = load_band(blu_path)
            nir = load_band(nir_path)
            true_color  = make_true_color(red, grn, blu)
            false_color = make_false_color(nir, red, grn)
        else:
            true_color  = cd_rgb
            false_color = cd_rgb

        # ── Mapas derivados ────────────────────────────────────────────────────
        cd_error_map  = make_error_map(cd_pred,  cd_gt)
        ocm_error_map = make_error_map(ocm_binary, cd_gt)
        diff_map      = make_diff_map(cd_pred, ocm_binary, cd_gt)
        ocm_overlay_arr = make_ocm_overlay(true_color, ocm_4class)

        # ── Métricas ───────────────────────────────────────────────────────────
        cd_metrics  = compute_metrics(cd_pred,   cd_gt)
        ocm_metrics = compute_metrics(ocm_binary, cd_gt)
        agreement   = pixel_agreement(cd_pred, ocm_binary)

        # ── Figura comparativa (3 filas × 5 columnas) ─────────────────────────
        fig, axes = plt.subplots(3, 5, figsize=(25, 16))

        # Fila 1 — CD-Mamba
        axes[0,0].imshow(cd_rgb);          axes[0,0].set_title("CD-Mamba: RGB input",        fontsize=9)
        axes[0,1].imshow(cd_gt, cmap='gray', vmin=0, vmax=1); axes[0,1].set_title("GT mask (binary)",  fontsize=9)
        axes[0,2].imshow(cd_pred, cmap='gray', vmin=0, vmax=1); axes[0,2].set_title("CD-Mamba prediction", fontsize=9)
        axes[0,3].imshow(cd_overlay);      axes[0,3].set_title("CD-Mamba overlay",           fontsize=9)
        axes[0,4].imshow(cd_error_map);    axes[0,4].legend(handles=error_legend, loc='lower right', fontsize=6)
        axes[0,4].set_title("CD-Mamba error map",             fontsize=9)

        # Fila 2 — OmniCloudMask
        axes[1,0].imshow(true_color);      axes[1,0].set_title("True Color (R/G/B)",          fontsize=9)
        axes[1,1].imshow(cd_gt, cmap='gray', vmin=0, vmax=1); axes[1,1].set_title("GT mask (binary)",  fontsize=9)
        axes[1,2].imshow(ocm_binary, cmap='gray', vmin=0, vmax=1)
        axes[1,2].set_title("OmniCloud binarized\n(from 4-class)", fontsize=9)
        axes[1,3].imshow(ocm_4class, cmap='tab10', vmin=0, vmax=3)
        axes[1,3].legend(handles=ocm_legend, loc='lower right', fontsize=6)
        axes[1,3].set_title("OmniCloud 4-class mask",         fontsize=9)
        axes[1,4].imshow(true_color); axes[1,4].imshow(ocm_overlay_arr)
        axes[1,4].set_title("OmniCloud overlay",              fontsize=9)

        # Fila 3 — Diferencias
        axes[2,0].imshow(diff_map);        axes[2,0].legend(handles=diff_legend, loc='lower right', fontsize=6)
        axes[2,0].set_title("Difference map\n(CD-Mamba vs OmniCloud)", fontsize=9)
        axes[2,1].imshow(false_color);     axes[2,1].set_title("False Color (NIR/R/G)",       fontsize=9)
        axes[2,2].imshow(ocm_error_map);   axes[2,2].legend(handles=error_legend, loc='lower right', fontsize=6)
        axes[2,2].set_title("OmniCloud error map",            fontsize=9)

        # Panel métricas comparativas
        axes[2,3].axis('off')
        metrics_text = (
            f"{'Metric':<18} {'CD-Mamba':>10} {'OmniCloud':>10}\n"
            f"{'-'*40}\n"
            f"{'F1':<18} {cd_metrics['f1']:>10.4f} {ocm_metrics['f1']:>10.4f}\n"
            f"{'Precision':<18} {cd_metrics['precision']:>10.4f} {ocm_metrics['precision']:>10.4f}\n"
            f"{'Recall':<18} {cd_metrics['recall']:>10.4f} {ocm_metrics['recall']:>10.4f}\n"
            f"{'Accuracy':<18} {cd_metrics['accuracy']:>10.4f} {ocm_metrics['accuracy']:>10.4f}\n"
            f"{'IoU':<18} {cd_metrics['iou']:>10.4f} {ocm_metrics['iou']:>10.4f}\n"
            f"{'Kappa':<18} {cd_metrics['kappa']:>10.4f} {ocm_metrics['kappa']:>10.4f}\n"
            f"{'CSI':<18} {cd_metrics['csi']:>10.4f} {ocm_metrics['csi']:>10.4f}\n"
            f"{'Dice':<18} {cd_metrics['dice']:>10.4f} {ocm_metrics['dice']:>10.4f}\n"
            f"{'FAR':<18} {cd_metrics['false_alarm_rate']:>10.4f} {ocm_metrics['false_alarm_rate']:>10.4f}\n"
            f"{'Miss Rate':<18} {cd_metrics['miss_rate']:>10.4f} {ocm_metrics['miss_rate']:>10.4f}\n"
            f"{'-'*40}\n"
            f"{'Pixel agreement':<18} {agreement:>10.4f}"
        )
        axes[2,3].text(0.05, 0.95, metrics_text, transform=axes[2,3].transAxes,
                       fontsize=8, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[2,3].set_title("Metrics comparison", fontsize=9)

        # Panel barras comparativas
        axes[2,4].axis('off')
        metric_names = ['F1', 'Precision', 'Recall', 'IoU', 'Kappa', 'CSI', 'Dice']
        cd_vals  = [cd_metrics['f1'],  cd_metrics['precision'],  cd_metrics['recall'],
                    cd_metrics['iou'], cd_metrics['kappa'],       cd_metrics['csi'], cd_metrics['dice']]
        ocm_vals = [ocm_metrics['f1'], ocm_metrics['precision'], ocm_metrics['recall'],
                    ocm_metrics['iou'],ocm_metrics['kappa'],      ocm_metrics['csi'], ocm_metrics['dice']]

        ax_bar = axes[2,4]
        ax_bar.set_visible(True)
        x    = np.arange(len(metric_names))
        w    = 0.35
        ax_bar.bar(x - w/2, cd_vals,  w, label='CD-Mamba',    color='steelblue',  alpha=0.85)
        ax_bar.bar(x + w/2, ocm_vals, w, label='OmniCloud',   color='darkorange', alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_names, fontsize=7, rotation=30)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.legend(fontsize=7)
        ax_bar.set_title("Metrics bar chart", fontsize=9)
        ax_bar.grid(axis='y', alpha=0.3)

        for ax in axes.flat:
            ax.axis('off') if not ax.has_data() else ax.set_xticks([])
            if ax.has_data():
                ax.set_yticks([])

        fig.suptitle(f"{fold} — {patch_name}  |  Pixel agreement: {agreement:.4f}",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        out_path = OUT_DIR / fold / (patch_name + "_comparison.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # ── Guardar métricas ───────────────────────────────────────────────────
        all_results.append({
            "fold":            fold,
            "patch":           patch_name,
            "pixel_agreement": round(agreement, 4),
            "cdmamba":         cd_metrics,
            "omnicloud":       ocm_metrics,
        })

# ── Figura resumen global ──────────────────────────────────────────────────────
if all_results:
    metric_keys = ['f1','precision','recall','iou','kappa','csi','dice',
                   'false_alarm_rate','miss_rate']
    cd_means  = {k: np.mean([r['cdmamba'][k]  for r in all_results]) for k in metric_keys}
    ocm_means = {k: np.mean([r['omnicloud'][k] for r in all_results]) for k in metric_keys}
    mean_agr  = np.mean([r['pixel_agreement'] for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Barras globales
    x = np.arange(len(metric_keys))
    w = 0.35
    axes[0].bar(x - w/2, [cd_means[k]  for k in metric_keys], w,
                label='CD-Mamba',  color='steelblue',  alpha=0.85)
    axes[0].bar(x + w/2, [ocm_means[k] for k in metric_keys], w,
                label='OmniCloud', color='darkorange', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([k.replace('_','\n') for k in metric_keys], fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Global mean metrics (60 patches)\nPixel agreement: {mean_agr:.4f}", fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Scatter F1 patch a patch
    cd_f1s  = [r['cdmamba']['f1']  for r in all_results]
    ocm_f1s = [r['omnicloud']['f1'] for r in all_results]
    axes[1].scatter(cd_f1s, ocm_f1s, alpha=0.7, color='purple', s=60)
    axes[1].plot([0,1],[0,1], 'k--', alpha=0.4, label='y=x')
    axes[1].set_xlabel("CD-Mamba F1", fontsize=10)
    axes[1].set_ylabel("OmniCloud F1", fontsize=10)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_title("F1 per patch: CD-Mamba vs OmniCloud", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_comparison.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[INFO] Summary figure saved.")

# ── JSON final ─────────────────────────────────────────────────────────────────
out_json = OUT_DIR / "comparison_results.json"
with open(out_json, "w") as f:
    json.dump({
        "n_patches":       len(all_results),
        "missing_ocm":     missing_ocm,
        "mean_agreement":  round(float(mean_agr), 4) if all_results else 0,
        "global_means": {
            "cdmamba":   {k: round(float(v),4) for k,v in cd_means.items()},
            "omnicloud": {k: round(float(v),4) for k,v in ocm_means.items()},
        },
        "per_patch": all_results,
    }, f, indent=2)

print(f"\n[DONE]")
print(f"  Patches compared : {len(all_results)}")
print(f"  Missing OmniCloud: {len(missing_ocm)}")
if all_results:
    print(f"  Mean pixel agr.  : {mean_agr:.4f}")
    print(f"  CD-Mamba F1 mean : {cd_means['f1']:.4f}")
    print(f"  OmniCloud F1 mean: {ocm_means['f1']:.4f}")
print(f"  Saved to         : {OUT_DIR}")
