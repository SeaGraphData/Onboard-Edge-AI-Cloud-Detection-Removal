import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/scratch/users/jfernandezmartinez/CDMamba/dataset/biome")
RED_DIR   = BASE / "red"
GREEN_DIR = BASE / "green"
BLUE_DIR  = BASE / "blue"
NIR_DIR   = BASE / "nir"
MASK_DIR  = BASE / "mask"

RESULTS_DIR = Path("/scratch/users/jfernandezmartinez/GeoAI/results")
MASKS_4CLASS = RESULTS_DIR / "masks_4class"
OUT_DIR      = RESULTS_DIR / "figures_rgb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FIGURES = 20

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def load_gt_mask(path):
    return (np.array(Image.open(path)) > 127).astype(np.uint8)

def stretch(arr):
    """Percentile stretch para visualización."""
    p2, p98 = np.percentile(arr, (2, 98))
    arr = (arr - p2) / (p98 - p2 + 1e-8)
    return np.clip(arr, 0, 1)

def make_true_color(red, green, blue):
    """RGB true color: Red/Green/Blue con gamma 0.5 como en el notebook HLS."""
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, None)
    rgb = np.power(rgb, 0.5)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)

def make_false_color(nir, red, green):
    """False color NIR/Red/Green con stretch percentil."""
    fc = np.stack([stretch(nir), stretch(red), stretch(green)], axis=-1)
    return np.clip(fc, 0, 1)

def make_overlay(rgb_display, mask_4class):
    """Overlay semitransparente de la máscara sobre la imagen, igual que en el notebook HLS."""
    tab10   = plt.get_cmap("tab10")
    mask_sq = mask_4class.astype(np.uint8)
    rgba    = np.zeros((*mask_sq.shape, 4), dtype=float)
    for cls in range(4):
        colour = tab10(cls / 9)
        where  = mask_sq == cls
        rgba[where, :3] = colour[:3]
        rgba[where,  3] = 0.0 if cls == 0 else 0.4  # clear=transparente
    return rgba

# ── Clase labels ───────────────────────────────────────────────────────────────
tab10      = plt.get_cmap("tab10")
CLASS_NAMES = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]
legend_patches = [
    mpatches.Patch(color=tab10(i / 9), label=CLASS_NAMES[i])
    for i in range(4)
]

# ── Listar patches con máscara 4 clases ya generada ───────────────────────────
available = sorted([f.stem for f in MASKS_4CLASS.glob("*.png")])
print(f"[INFO] Máscaras 4 clases disponibles: {len(available)}")

patches_to_plot = available[:N_FIGURES]
print(f"[INFO] Generando figuras para los primeros {len(patches_to_plot)} patches...")

# ── Generar figuras ────────────────────────────────────────────────────────────
for patch_name in tqdm(patches_to_plot, desc="Generating figures"):

    red_path   = RED_DIR   / (patch_name + ".TIF")
    green_path = GREEN_DIR / (patch_name + ".TIF")
    blue_path  = BLUE_DIR  / (patch_name + ".TIF")
    nir_path   = NIR_DIR   / (patch_name + ".TIF")
    gt_path    = MASK_DIR  / (patch_name + ".png")
    mask_path  = MASKS_4CLASS / (patch_name + ".png")

    # Verificar que existen todos los ficheros
    missing = [p for p in [red_path, green_path, blue_path, nir_path, gt_path, mask_path]
               if not p.exists()]
    if missing:
        print(f"[WARN] Skipping {patch_name} — missing: {[p.name for p in missing]}")
        continue

    # Cargar bandas
    red   = load_band(red_path)
    green = load_band(green_path)
    blue  = load_band(blue_path)
    nir   = load_band(nir_path)
    gt    = load_gt_mask(gt_path)

    # Cargar máscara 4 clases ya generada por el job principal
    mask_4class = np.array(Image.open(mask_path).convert("L"))
    # Convertir PNG RGB de colores a índices 0-3
    mask_rgb = np.array(Image.open(mask_path))
    CLASS_COLORS = {
        (0,   180,   0): 0,  # Clear
        (255, 255, 255): 1,  # Thick Cloud
        (180, 180, 180): 2,  # Thin Cloud
        (30,   30, 100): 3,  # Cloud Shadow
    }
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, idx in CLASS_COLORS.items():
        match = np.all(mask_rgb[:, :, :3] == np.array(color), axis=-1)
        mask_idx[match] = idx

    # Construir visualizaciones
    true_color  = make_true_color(red, green, blue)
    false_color = make_false_color(nir, red, green)
    overlay     = make_overlay(true_color, mask_idx)

    # ── Figura 6 paneles ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Fila 1
    axes[0, 0].imshow(true_color)
    axes[0, 0].set_title("True Color (R/G/B)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title("False Color (NIR/R/G)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(true_color)
    axes[0, 2].imshow(overlay)
    axes[0, 2].legend(handles=legend_patches, loc="lower right", fontsize=8)
    axes[0, 2].set_title("Overlay (True Color + mask)", fontsize=11)
    axes[0, 2].axis("off")

    # Fila 2
    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT mask (binary)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mask_idx, cmap="tab10", vmin=0, vmax=3)
    axes[1, 1].legend(handles=legend_patches, loc="lower right", fontsize=8)
    axes[1, 1].set_title("OmniCloudMask (4 classes)", fontsize=11)
    axes[1, 1].axis("off")

    # Binarizada
    pred_binary = (mask_idx >= 1).astype(np.uint8)
    axes[1, 2].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Binarized prediction", fontsize=11)
    axes[1, 2].axis("off")

    fig.suptitle(patch_name, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / (patch_name + ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

print(f"\n[DONE] Figuras guardadas en: {OUT_DIR}")
