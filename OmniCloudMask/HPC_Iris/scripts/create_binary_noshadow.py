import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

MASKS_4CLASS = Path("/scratch/users/jfernandezmartinez/GeoAI/results/masks_4class")
OUT_DIR      = Path("/scratch/users/jfernandezmartinez/GeoAI/results/masks_binary_noshadow")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color → class index mapping
COLOR_TO_IDX = {
    (0,   180,   0): 0,  # Clear
    (255, 255, 255): 1,  # Thick Cloud
    (180, 180, 180): 2,  # Thin Cloud
    (30,   30, 100): 3,  # Cloud Shadow
}

files = sorted(MASKS_4CLASS.glob("*.png"))
print(f"[INFO] Total 4-class masks: {len(files)}")

for f in tqdm(files, desc="Creating binary_noshadow masks"):
    mask_rgb = np.array(Image.open(f).convert("RGB"))
    mask_idx = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, idx in COLOR_TO_IDX.items():
        mask_idx[np.all(mask_rgb == np.array(color), axis=-1)] = idx

    # Binarisation: only thick cloud (1) + thin cloud (2) = cloud
    # Clear (0) and Cloud Shadow (3) both treated as non-cloud
    pred_binary = ((mask_idx == 1) | (mask_idx == 2)).astype(np.uint8)
    Image.fromarray(pred_binary * 255).save(OUT_DIR / f.name)

print(f"\n[DONE] Saved to: {OUT_DIR}")
print(f"  Total masks: {len(files)}")
