#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_snpatchgan_quality_v2.py
Cloud Removal Quality Metrics for SN-PatchGAN (full CloudGAN pipeline)
Part 1: RICE examples — PSNR, SSIM, MAE, MSE, RMSE, BRMSE, SAM
Part 2: 38-cloud test images — visual outputs only (no target available)
"""
import os, sys, json, datetime, math, glob
import numpy as np
import cv2
import tensorflow as tf
import neuralgym as ng

SCRATCH      = "/scratch/users/jfernandezmartinez"
PROJECT      = os.path.join(SCRATCH, "CloudGAN")
CONFIG_GAN   = os.path.join(PROJECT, "config", "cloud_removal_config.yml")
WEIGHTS_GAN  = os.path.join(PROJECT, "weights", "SN_PatchGAN")
TEST_IMG     = os.path.join(PROJECT, "datasets", "38-cloud", "test", "img", "data")
TEST_MASK    = os.path.join(PROJECT, "datasets", "38-cloud", "test", "mask", "data")
RICE_DIR     = os.path.join(PROJECT, "cloud_removal", "examples", "RICE")
OUTPUT_DIR   = os.path.join(PROJECT, "evaluation", "results", "snpatchgan_quality_v2_images")
OUTPUT_38    = os.path.join(PROJECT, "evaluation", "results", "snpatchgan_38cloud_visual")
RESULT_JSON  = os.path.join(PROJECT, "evaluation", "results", "snpatchgan_quality_v2.json")
RESULT_TXT   = os.path.join(PROJECT, "evaluation", "results", "snpatchgan_quality_v2.txt")

NUM_38CLOUD_SAMPLES = 10  # visual-only samples from 38-cloud

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, PROJECT)
from cloud_removal.inpaint_model import InpaintCAModel
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_38, exist_ok=True)

# ── Metric helpers ──
def compute_sam(img1, img2):
    a = img1.astype(np.float64).reshape(-1, 3)
    b = img2.astype(np.float64).reshape(-1, 3)
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    valid = (norm_a > 0) & (norm_b > 0)
    cos_angle = np.clip(dot[valid] / (norm_a[valid] * norm_b[valid]), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))
    return float(np.mean(angles)) if len(angles) > 0 else 0.0

def compute_metrics(target, output):
    t = target.astype(np.float64) / 255.0
    o = output.astype(np.float64) / 255.0
    diff = o - t
    mae  = float(np.mean(np.abs(diff)))
    mse  = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)
    brmse = float(np.mean(np.sqrt(np.mean(diff ** 2, axis=(0, 1)))))
    psnr = peak_signal_noise_ratio(target, output)
    ssim = structural_similarity(target, output, multichannel=True)
    sam  = compute_sam(target, output)
    return {
        "PSNR": round(psnr, 4), "SSIM": round(ssim, 4),
        "MAE": round(mae, 6), "MSE": round(mse, 6),
        "RMSE": round(rmse, 6), "BRMSE": round(brmse, 6),
        "SAM": round(sam, 4),
    }

def run_snpatchgan(img_bgr, mask_3ch_norm):
    """Run SN-PatchGAN. img_bgr: uint8, mask_3ch_norm: float [0,1]."""
    tf.compat.v1.reset_default_graph()
    FLAGS_GAN = ng.Config(CONFIG_GAN)
    img  = np.expand_dims(img_bgr.astype(np.float32), axis=0)
    mask = np.expand_dims(mask_3ch_norm.astype(np.float32), axis=0)
    input_img = np.concatenate([img, mask * 255], axis=2)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    model = InpaintCAModel()
    with tf.compat.v1.Session(config=sess_config) as sess:
        input_tensor = tf.constant(input_img, dtype=tf.float32)
        output = model.build_server_graph(FLAGS_GAN, input_tensor)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            var_value = tf.contrib.framework.load_variable(WEIGHTS_GAN, var.name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        result = sess.run(output)[0][:, :, ::-1]
    return result

print("=" * 60)
print("  SN-PatchGAN — Cloud Removal Quality Evaluation v2")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# PART 1: RICE examples — full metrics
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 1: RICE examples (with quality metrics)")
print("=" * 60)

examples = []
for d in sorted(os.listdir(RICE_DIR)):
    ex_dir = os.path.join(RICE_DIR, d)
    if not os.path.isdir(ex_dir):
        continue
    s = os.path.join(ex_dir, "sample.png")
    t = os.path.join(ex_dir, "target.png")
    m = os.path.join(ex_dir, "mask.png")
    if os.path.exists(s) and os.path.exists(t) and os.path.exists(m):
        examples.append({"id": d, "sample": s, "target": t, "mask": m})
print("  Found {} RICE examples".format(len(examples)))

per_image_results = []
for idx, ex in enumerate(examples):
    print("\n  --- RICE Example {} (id={}) ---".format(idx + 1, ex["id"]))
    img_bgr    = cv2.imread(ex["sample"])
    target_bgr = cv2.imread(ex["target"])
    mask_bgr   = cv2.imread(ex["mask"])
    mask_norm  = mask_bgr.astype(np.float32) / 255.0

    output_img = run_snpatchgan(img_bgr, mask_norm)
    m = compute_metrics(target_bgr, output_img)
    m["id"] = ex["id"]
    per_image_results.append(m)

    print("    PSNR={:.4f} SSIM={:.4f} MAE={:.6f} SAM={:.4f}".format(
        m["PSNR"], m["SSIM"], m["MAE"], m["SAM"]))

    composite = np.concatenate([img_bgr, mask_bgr, output_img, target_bgr], axis=1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "rice_{}.png".format(ex["id"])), composite)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "rice_{}_output.png".format(ex["id"])), output_img)

# Averages
metric_keys = ["PSNR", "SSIM", "MAE", "MSE", "RMSE", "BRMSE", "SAM"]
averages = {}
for k in metric_keys:
    vals = [r[k] for r in per_image_results]
    averages["avg_" + k] = round(float(np.mean(vals)), 4 if k in ["PSNR","SSIM","SAM"] else 6)

# ══════════════════════════════════════════════════════════════
# PART 2: 38-cloud test images — visual output only (no target)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 2: 38-cloud visual outputs (no metrics, thesis figures)")
print("=" * 60)

# Select images with 15-70% cloud coverage for interesting visuals
print("  Scanning 38-cloud test masks for good candidates...")
img_files = sorted(glob.glob(os.path.join(TEST_IMG, "*.png")))
candidates = []
for img_path in img_files:
    basename = os.path.basename(img_path)
    mask_path = os.path.join(TEST_MASK, basename)
    if not os.path.exists(mask_path):
        continue
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        continue
    coverage = float(np.mean(mask_gray > 127))
    if 0.15 <= coverage <= 0.70:
        candidates.append((img_path, mask_path, coverage))

# Sort by coverage (mid-range first) and pick top N
candidates.sort(key=lambda x: abs(x[2] - 0.40))
selected = candidates[:NUM_38CLOUD_SAMPLES]
print("  Selected {} images with 15-70% cloud coverage".format(len(selected)))

visual_info = []
for idx, (img_path, mask_path, coverage) in enumerate(selected):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    print("\n  --- 38-cloud [{}/{}] coverage={:.1f}% file={} ---".format(
        idx + 1, len(selected), coverage * 100, basename[:30]))

    img_bgr  = cv2.imread(img_path)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask_gray > 127).astype(np.float32)
    mask_3ch = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=2)

    output_img = run_snpatchgan(img_bgr, mask_3ch)

    # Composite: Input | GT Mask | SN-PatchGAN Output
    mask_vis = cv2.cvtColor((mask_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    composite = np.concatenate([img_bgr, mask_vis, output_img], axis=1)
    out_name = "38cloud_{:02d}_{}.png".format(idx, basename[:40])
    cv2.imwrite(os.path.join(OUTPUT_38, out_name), composite)
    cv2.imwrite(os.path.join(OUTPUT_38, "38cloud_{:02d}_output.png".format(idx)), output_img)

    visual_info.append({
        "index": idx,
        "filename": os.path.basename(img_path),
        "cloud_coverage_pct": round(coverage * 100, 1),
    })
    print("    Saved: {}".format(out_name))

print("\n  38-cloud visual outputs saved: {}".format(OUTPUT_38))

# ══════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════
results = {
    "model": "SN-PatchGAN (CloudGAN pipeline)",
    "task": "Cloud Removal",
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "weights_GAN": WEIGHTS_GAN,
    "config_GAN": CONFIG_GAN,
    "part1_rice": {
        "dataset": "RICE examples",
        "num_examples": len(examples),
        "averages": averages,
        "per_image": per_image_results,
    },
    "part2_38cloud_visual": {
        "dataset": "38-cloud test set (visual only, no target)",
        "num_images": len(visual_info),
        "images": visual_info,
    },
}

with open(RESULT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

sep = "=" * 60
summary = """
{0}
  SN-PatchGAN — Cloud Removal Quality Metrics v2
{0}

  PART 1: RICE (with metrics, {1} images)
  ----------------------------------------
    PSNR:   {2} dB
    SSIM:   {3}
    MAE:    {4}
    MSE:    {5}
    RMSE:   {6}
    BRMSE:  {7}
    SAM:    {8} deg
""".format(sep, len(examples),
           averages["avg_PSNR"], averages["avg_SSIM"],
           averages["avg_MAE"], averages["avg_MSE"],
           averages["avg_RMSE"], averages["avg_BRMSE"],
           averages["avg_SAM"])

for r in per_image_results:
    summary += "    Ex {}: PSNR={:.4f} SSIM={:.4f} MAE={:.6f} SAM={:.4f}\n".format(
        r["id"], r["PSNR"], r["SSIM"], r["MAE"], r["SAM"])

summary += """
  PART 2: 38-cloud (visual only, {0} images)
  ----------------------------------------
    Output dir: {1}
""".format(len(visual_info), OUTPUT_38)

for v in visual_info:
    summary += "    [{0}] {1} (coverage: {2}%)\n".format(
        v["index"], v["filename"][:40], v["cloud_coverage_pct"])

summary += sep + "\n"

with open(RESULT_TXT, 'w') as f:
    f.write(summary)
print(summary)
print("JSON: {}".format(RESULT_JSON))
print("TXT: {}".format(RESULT_TXT))
print("Done!")
