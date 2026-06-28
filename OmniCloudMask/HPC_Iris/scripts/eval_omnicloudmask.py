import os
import json
import time
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omnicloudmask import predict_from_array
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, jaccard_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import threading

# ── pynvml ─────────────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_OK  = True
    print("[INFO] pynvml OK")
except Exception as e:
    PYNVML_OK  = False
    GPU_HANDLE = None
    print(f"[WARN] pynvml not available: {e}")

# ── thop ───────────────────────────────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    THOP_OK = True
    print("[INFO] thop OK")
except Exception as e:
    THOP_OK = False
    print(f"[WARN] thop not available: {e}")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome")
RED_DIR   = BASE / "red"
GREEN_DIR = BASE / "green"
NIR_DIR   = BASE / "nir"
BLUE_DIR  = BASE / "blue"
MASK_DIR  = BASE / "mask"
OUT_DIR   = Path("/scratch/users/jfernandezmartinez/GeoAI/results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "masks_4class").mkdir(exist_ok=True)
(OUT_DIR / "masks_binary").mkdir(exist_ok=True)
(OUT_DIR / "figures_red").mkdir(exist_ok=True)
(OUT_DIR / "figures_green").mkdir(exist_ok=True)
(OUT_DIR / "figures_blue").mkdir(exist_ok=True)
(OUT_DIR / "figures_nir").mkdir(exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device : {device}")
print(f"[INFO] torch  : {torch.__version__}")
if device == "cuda":
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

# ── Clase colors ───────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0,   180,   0),
    1: (255, 255, 255),
    2: (180, 180, 180),
    3: (30,   30, 100),
}
CLASS_NAMES  = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]
tab10        = plt.get_cmap("tab10")
legend_patches = [
    mpatches.Patch(color=tab10(i / 9), label=CLASS_NAMES[i])
    for i in range(4)
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def load_gt_mask(path):
    return (np.array(Image.open(path)) > 127).astype(np.uint8)

def stretch(arr):
    p2, p98 = np.percentile(arr, (2, 98))
    arr = (arr - p2) / (p98 - p2 + 1e-8)
    return np.clip(arr, 0, 1)

def make_true_color(red, green, blue):
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, None)
    rgb = np.power(rgb, 0.5)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)

def make_false_color(nir, red, green):
    return np.clip(
        np.stack([stretch(nir), stretch(red), stretch(green)], axis=-1), 0, 1
    )

def make_overlay(rgb_display, mask_idx):
    rgba = np.zeros((*mask_idx.shape, 4), dtype=float)
    for cls in range(4):
        colour = tab10(cls / 9)
        where  = mask_idx == cls
        rgba[where, :3] = colour[:3]
        rgba[where,  3] = 0.0 if cls == 0 else 0.4
    return rgba

def save_4class_png(pred_4class, out_path):
    h, w = pred_4class.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[pred_4class == cls] = color
    Image.fromarray(rgb).save(out_path)

def save_binary_png(pred_binary, out_path):
    Image.fromarray((pred_binary * 255).astype(np.uint8)).save(out_path)

def save_figure(band_arr, band_name, gt, mask_idx, true_color,
                false_color, overlay, patch_name, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Fila 1
    axes[0, 0].imshow(band_arr, cmap='gray')
    axes[0, 0].set_title(f"{band_name} band", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(true_color)
    axes[0, 1].set_title("True Color (R/G/B)", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(false_color)
    axes[0, 2].set_title("False Color (NIR/R/G)", fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(true_color)
    axes[0, 3].imshow(overlay)
    axes[0, 3].legend(handles=legend_patches, loc='lower right', fontsize=7)
    axes[0, 3].set_title("Overlay", fontsize=10)
    axes[0, 3].axis('off')

    # Fila 2
    axes[1, 0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("GT mask (binary)", fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_idx, cmap='tab10', vmin=0, vmax=3)
    axes[1, 1].legend(handles=legend_patches, loc='lower right', fontsize=7)
    axes[1, 1].set_title("OmniCloudMask (4 classes)", fontsize=10)
    axes[1, 1].axis('off')

    pred_binary = (mask_idx >= 1).astype(np.uint8)
    axes[1, 2].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("Binarized prediction", fontsize=10)
    axes[1, 2].axis('off')

    # Panel vacío
    axes[1, 3].axis('off')

    fig.suptitle(patch_name, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# ── Power sampler ──────────────────────────────────────────────────────────────
class PowerSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples  = []
        self._stop    = False
        self._thread  = None

    def start(self):
        self.samples = []
        self._stop   = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join()

    def _run(self):
        while not self._stop:
            if PYNVML_OK:
                try:
                    self.samples.append(
                        pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
                    )
                except:
                    pass
            time.sleep(self.interval)

    def avg_power_w(self):
        return float(np.mean(self.samples)) if self.samples else 0.0

def get_gpu_memory_mb():
    if PYNVML_OK:
        try:
            return round(pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE).used / 1024**2, 1)
        except:
            pass
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**2, 1)
    return 0.0

# ── FLOPs con collect_models ───────────────────────────────────────────────────
gflops  = None
mparams = None

if THOP_OK and torch.cuda.is_available():
    print("[INFO] Measuring FLOPs via collect_models...")
    try:
        import omnicloudmask.cloud_mask as cm
        device_obj = torch.device("cuda")
        models = cm.collect_models(
            custom_models=None,
            inference_device=device_obj,
            inference_dtype=torch.float32,
            source="hugging_face",
        )
        dummy_t     = torch.randn(1, 3, 384, 384).cuda()
        total_macs  = 0
        total_params = 0
        for i, model in enumerate(models):
            model.eval()
            macs, params = thop_profile(model, inputs=(dummy_t,), verbose=False)
            total_macs   += macs
            total_params += params
            print(f"  Model {i+1}: {round(macs*2/1e9,3)} GFLOPs, {round(params/1e6,3)} M params")
        gflops  = round(total_macs   * 2 / 1e9, 3)
        mparams = round(total_params / 1e6,      3)
        print(f"[INFO] Total GFLOPs : {gflops}")
        print(f"[INFO] Total Params : {mparams} M")
    except Exception as e:
        print(f"[WARN] FLOPs failed: {e}")

# ── Listar patches y detectar cuáles faltan (resume) ──────────────────────────
mask_files = sorted([f for f in MASK_DIR.iterdir() if f.suffix == '.png'])
print(f"[INFO] Total patches en dataset : {len(mask_files)}")

already_done = set(f.stem for f in (OUT_DIR / "masks_4class").iterdir())
pending      = [f for f in mask_files if f.stem not in already_done]
print(f"[INFO] Ya procesados            : {len(already_done)}")
print(f"[INFO] Pendientes               : {len(pending)}")

# Patches para figuras (primeros 20 de cada banda, de los ya procesados)
N_FIGURES   = 20
done_sorted = sorted(already_done)
fig_patches = done_sorted[:N_FIGURES]

# ── Cargar resultados existentes si los hay ────────────────────────────────────
existing_json = OUT_DIR / "results_omnicloudmask.json"
if existing_json.exists():
    with open(existing_json) as f:
        existing = json.load(f)
    per_patch = existing.get("per_patch", [])
    print(f"[INFO] Cargados {len(per_patch)} resultados previos del JSON")
else:
    per_patch = []

existing_patches = set(p["patch"] for p in per_patch)

# ── Evaluación de patches pendientes ──────────────────────────────────────────
all_gt   = []
all_pred = []

# Reconstruir arrays globales desde per_patch existente no es posible,
# así que recalculamos métricas globales al final solo con los patches procesados
power_sampler = PowerSampler(interval=0.05)
t_start = time.time()

for i, mask_path in enumerate(tqdm(pending, desc="Evaluating pending patches")):
    patch_name = mask_path.stem

    if patch_name in existing_patches:
        continue

    red_path   = RED_DIR   / (patch_name + ".TIF")
    green_path = GREEN_DIR / (patch_name + ".TIF")
    nir_path   = NIR_DIR   / (patch_name + ".TIF")

    if not (red_path.exists() and green_path.exists() and nir_path.exists()):
        print(f"[WARN] Missing bands for {patch_name}, skipping")
        continue

    red   = load_band(red_path)
    green = load_band(green_path)
    nir   = load_band(nir_path)
    gt    = load_gt_mask(mask_path)

    input_arr = np.stack([red, green, nir], axis=0).astype(np.float32)

    power_sampler.start()
    t0 = time.time()
    pred_mask = predict_from_array(input_arr, inference_dtype="float32", no_data_value=None)
    latency   = time.time() - t0
    power_sampler.stop()

    avg_power  = power_sampler.avg_power_w()
    energy_mj  = avg_power * latency * 1000
    gpu_mem_mb = get_gpu_memory_mb()

    pred_4class = pred_mask[0].astype(np.uint8)
    pred_binary = (pred_4class >= 1).astype(np.uint8)

    save_4class_png(pred_4class, OUT_DIR / "masks_4class" / (patch_name + ".png"))
    save_binary_png(pred_binary, OUT_DIR / "masks_binary" / (patch_name + ".png"))

    gt_flat   = gt.flatten()
    pred_flat = pred_binary.flatten()

    f1  = f1_score(gt_flat, pred_flat, zero_division=0)
    pr  = precision_score(gt_flat, pred_flat, zero_division=0)
    re  = recall_score(gt_flat, pred_flat, zero_division=0)
    acc = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)

    per_patch.append({
        "patch":      patch_name,
        "f1":         round(float(f1),  4),
        "precision":  round(float(pr),  4),
        "recall":     round(float(re),  4),
        "accuracy":   round(float(acc), 4),
        "iou":        round(float(iou), 4),
        "latency_s":  round(latency,    4),
        "power_w":    round(avg_power,  2),
        "energy_mj":  round(energy_mj,  3),
        "gpu_mem_mb": round(gpu_mem_mb, 1),
    })

t_pending = time.time() - t_start
print(f"[INFO] Pending patches done in {t_pending:.1f}s")

# ── Generar figuras para las 4 bandas ─────────────────────────────────────────
print(f"[INFO] Generating figures for {len(fig_patches)} patches x 4 bands...")

for patch_name in tqdm(fig_patches, desc="Generating figures"):
    red_path   = RED_DIR   / (patch_name + ".TIF")
    green_path = GREEN_DIR / (patch_name + ".TIF")
    blue_path  = BLUE_DIR  / (patch_name + ".TIF")
    nir_path   = NIR_DIR   / (patch_name + ".TIF")
    gt_path    = MASK_DIR  / (patch_name + ".png")
    mask_4c    = OUT_DIR / "masks_4class" / (patch_name + ".png")

    missing = [p for p in [red_path, green_path, blue_path, nir_path, gt_path, mask_4c]
               if not p.exists()]
    if missing:
        print(f"[WARN] Skipping {patch_name} — missing: {[p.name for p in missing]}")
        continue

    red   = load_band(red_path)
    green = load_band(green_path)
    blue  = load_band(blue_path)
    nir   = load_band(nir_path)
    gt    = load_gt_mask(gt_path)

    mask_rgb = np.array(Image.open(mask_4c))
    COLOR_TO_IDX = {
        (0,   180,   0): 0,
        (255, 255, 255): 1,
        (180, 180, 180): 2,
        (30,   30, 100): 3,
    }
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, idx in COLOR_TO_IDX.items():
        mask_idx[np.all(mask_rgb[:, :, :3] == np.array(color), axis=-1)] = idx

    true_color  = make_true_color(red, green, blue)
    false_color = make_false_color(nir, red, green)
    overlay     = make_overlay(true_color, mask_idx)

    for band_arr, band_name, fig_dir in [
        (red,   "Red",   OUT_DIR / "figures_red"),
        (green, "Green", OUT_DIR / "figures_green"),
        (blue,  "Blue",  OUT_DIR / "figures_blue"),
        (nir,   "NIR",   OUT_DIR / "figures_nir"),
    ]:
        save_figure(
            band_arr, band_name, gt, mask_idx,
            true_color, false_color, overlay,
            patch_name,
            fig_dir / (patch_name + ".png")
        )

# ── Métricas globales desde per_patch ─────────────────────────────────────────
print("[INFO] Computing global metrics from all per_patch entries...")

all_f1  = [p["f1"]        for p in per_patch]
all_pr  = [p["precision"]  for p in per_patch]
all_re  = [p["recall"]     for p in per_patch]
all_acc = [p["accuracy"]   for p in per_patch]
all_iou = [p["iou"]        for p in per_patch]
all_lat = [p["latency_s"]  for p in per_patch]
all_pw  = [p["power_w"]    for p in per_patch]
all_en  = [p["energy_mj"]  for p in per_patch]
all_mem = [p["gpu_mem_mb"] for p in per_patch]

n_patches   = len(per_patch)
avg_latency = float(np.mean(all_lat)) if all_lat else 0
fps         = 1.0 / avg_latency if avg_latency > 0 else 0

results = {
    "model":   "OmniCloudMask v1.7.1",
    "dataset": "Biome (Landsat-8)",
    "device":  device,
    "gpu":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "n_patches": n_patches,
    "efficiency_metrics": {
        "gflops":          gflops,
        "params_M":        mparams,
        "avg_latency_ms":  round(avg_latency * 1000, 2),
        "fps":             round(fps, 2),
        "avg_power_w":     round(float(np.mean(all_pw)),  2) if all_pw  else 0,
        "avg_energy_mj":   round(float(np.mean(all_en)),  3) if all_en  else 0,
        "avg_gpu_mem_mb":  round(float(np.mean(all_mem)), 1) if all_mem else 0,
        "peak_gpu_mem_mb": round(float(np.max(all_mem)),  1) if all_mem else 0,
    },
    "global_metrics": {
        "f1":        round(float(np.mean(all_f1)),  4),
        "precision": round(float(np.mean(all_pr)),  4),
        "recall":    round(float(np.mean(all_re)),  4),
        "accuracy":  round(float(np.mean(all_acc)), 4),
        "iou":       round(float(np.mean(all_iou)), 4),
    },
    "per_patch": per_patch,
}

with open(existing_json, "w") as f:
    json.dump(results, f, indent=2)

# ── TXT summary ───────────────────────────────────────────────────────────────
with open(OUT_DIR / "results_omnicloudmask.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("OmniCloudMask — Biome Dataset Evaluation\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Device         : {device}\n")
    f.write(f"GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")
    f.write(f"Patches        : {n_patches}\n\n")
    f.write("Efficiency Metrics:\n")
    f.write(f"  GFLOPs           : {gflops}\n")
    f.write(f"  Parameters       : {mparams} M\n")
    f.write(f"  Avg latency      : {avg_latency*1000:.2f} ms/patch\n")
    f.write(f"  FPS              : {fps:.2f}\n")
    f.write(f"  Avg power        : {results['efficiency_metrics']['avg_power_w']:.2f} W\n")
    f.write(f"  Avg energy       : {results['efficiency_metrics']['avg_energy_mj']:.3f} mJ\n")
    f.write(f"  Avg GPU memory   : {results['efficiency_metrics']['avg_gpu_mem_mb']:.1f} MB\n")
    f.write(f"  Peak GPU memory  : {results['efficiency_metrics']['peak_gpu_mem_mb']:.1f} MB\n\n")
    f.write("Global Metrics (mean over patches):\n")
    f.write(f"  F1         : {results['global_metrics']['f1']:.4f}\n")
    f.write(f"  Precision  : {results['global_metrics']['precision']:.4f}\n")
    f.write(f"  Recall     : {results['global_metrics']['recall']:.4f}\n")
    f.write(f"  Accuracy   : {results['global_metrics']['accuracy']:.4f}\n")
    f.write(f"  IoU        : {results['global_metrics']['iou']:.4f}\n")

print(f"\n[DONE]")
print(f"  Patches total   : {n_patches}")
print(f"  FPS             : {fps:.2f}")
print(f"  F1 (mean)       : {results['global_metrics']['f1']:.4f}")
print(f"  IoU (mean)      : {results['global_metrics']['iou']:.4f}")
print(f"  Results saved to: {OUT_DIR}")
