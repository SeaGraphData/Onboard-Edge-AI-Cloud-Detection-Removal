"""
CD-Mamba — TIF Band Patching Script
====================================
Patches full-scene Landsat TIF band files into 384×384 tiles,
replicating the EXACT same logic as split/patch.py (which was used
to create the ground-truth mask patches already in dataset/biome/mask/).

Padding applied before slicing (identical to patch.py):
    top=167, bottom=168, left=61, right=62  (zero padding)

Patch naming convention (identical to the mask filenames):
    patch_{k}_{row}_by_{col}_{scene_id}.TIF

Input  (per band folder, full scenes):
    dataset/biome/red/LC8XXXXXXXXXXXXXXXX_B4.TIF
    dataset/biome/green/LC8XXXXXXXXXXXXXXXX_B3.TIF
    dataset/biome/blue/LC8XXXXXXXXXXXXXXXX_B2.TIF
    dataset/biome/nir/LC8XXXXXXXXXXXXXXXX_B5.TIF

Output (saved in the same folder alongside the full scene):
    dataset/biome/red/patch_{k}_{r}_by_{c}_{scene_id}.TIF
    ...

Usage:
    python patch_tif_bands.py
"""

import os
import glob
import sys
import numpy as np

# ── TIF I/O: prefer tifffile, fall back to skimage ───────────────────────────
try:
    import tifffile
    def read_tif(path):
        return tifffile.imread(path)
    def write_tif(path, arr):
        tifffile.imwrite(path, arr)
    print("  [I/O backend] tifffile")
except ImportError:
    from skimage.io import imread as _sk_read, imsave as _sk_save
    def read_tif(path):
        return _sk_read(path)
    def write_tif(path, arr):
        _sk_save(path, arr)
    print("  [I/O backend] skimage (tifffile not found)")

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, 'dataset', 'biome')

# Band folder → band suffix in filename
BAND_CONFIG = {
    'red':   '_B4',
    'green': '_B3',
    'blue':  '_B2',
    'nir':   '_B5',
}

PATCH_SIZE = 384
PAD_TOP    = 167
PAD_BOTTOM = 168
PAD_LEFT   = 61
PAD_RIGHT  = 62


# ── Core patching function ────────────────────────────────────────────────────
def patch_scene(tif_path: str, output_dir: str, scene_id: str):
    """
    Read one full-scene TIF, apply zero-padding, slice into PATCH_SIZE×PATCH_SIZE
    tiles, and write each tile as a separate TIF.

    Returns (n_saved, n_skipped, grid_total).
    """
    img = read_tif(tif_path)

    # Ensure 2-D (single band) — some TIFs may be read as (1, H, W)
    if img.ndim == 3:
        if img.shape[0] <= 4:
            img = img[0]        # (bands, H, W) → take first band
        else:
            img = img[:, :, 0]  # (H, W, bands) → take first band

    # Zero-pad exactly as patch.py does
    img = np.pad(img,
                 ((PAD_TOP, PAD_BOTTOM), (PAD_LEFT, PAD_RIGHT)),
                 mode='constant', constant_values=0)

    H, W    = img.shape
    n_rows  = H // PATCH_SIZE
    n_cols  = W // PATCH_SIZE

    k       = 0
    saved   = 0
    skipped = 0

    for i in range(n_rows):
        for j in range(n_cols):
            k += 1
            patch = img[i * PATCH_SIZE:(i + 1) * PATCH_SIZE,
                        j * PATCH_SIZE:(j + 1) * PATCH_SIZE]

            # Name matches the mask naming: patch_{k}_{row}_by_{col}_{scene_id}
            out_name = f'patch_{k}_{i + 1}_by_{j + 1}_{scene_id}.TIF'
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            write_tif(out_path, patch)
            saved += 1

    return saved, skipped, k


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  CD-Mamba — TIF Band Patching")
    print("=" * 65)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Patch size   : {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"  Padding      : top={PAD_TOP}, bottom={PAD_BOTTOM}, "
          f"left={PAD_LEFT}, right={PAD_RIGHT}")
    print("=" * 65)

    grand_saved   = 0
    grand_skipped = 0

    for band_folder, band_suffix in BAND_CONFIG.items():
        band_dir = os.path.join(DATASET_ROOT, band_folder)

        if not os.path.isdir(band_dir):
            print(f"\n  [WARN] Folder not found, skipping: {band_dir}")
            continue

        # Only process full-scene TIFs (exclude already-patched files)
        all_tifs    = sorted(glob.glob(os.path.join(band_dir, 'LC8*.TIF')))
        scene_files = [f for f in all_tifs
                       if 'patch_' not in os.path.basename(f)]

        print(f"\n  [{band_folder.upper():5s}]  {len(scene_files)} scene TIFs to patch")

        band_saved   = 0
        band_skipped = 0

        for tif_path in scene_files:
            basename = os.path.basename(tif_path)
            # Strip band suffix + '.TIF' to get clean scene ID
            # e.g. 'LC80010112014080LGN00_B4.TIF' → 'LC80010112014080LGN00'
            scene_id = basename.replace(band_suffix + '.TIF', '')

            try:
                saved, skipped, n_total = patch_scene(tif_path, band_dir, scene_id)
            except Exception as e:
                print(f"    [ERROR] {basename}: {e}")
                continue

            band_saved   += saved
            band_skipped += skipped
            print(f"    {scene_id}  →  "
                  f"saved={saved:4d}, skipped={skipped:4d}, grid={n_total}")

        grand_saved   += band_saved
        grand_skipped += band_skipped
        print(f"  [{band_folder.upper():5s}]  Done — "
              f"new patches: {band_saved:,}, "
              f"already existed: {band_skipped:,}")

    print("\n" + "=" * 65)
    print("  Patching complete.")
    print(f"  New patches written   : {grand_saved:,}")
    print(f"  Patches already exist : {grand_skipped:,}")
    print("=" * 65)


if __name__ == '__main__':
    main()
