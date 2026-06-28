#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_uncertainty_filter.py
Demonstrates the uncertainty filter for the CloudGAN pipeline.
For each 38-cloud test image, computes:
  - Cloud coverage (from predicted hard mask)
  - AE uncertainty (fraction of pixels near 0.5 in soft mask)
  - A combined confidence flag: ACCEPT or REJECT
Saves results as JSON + CSV + visual examples of accepted vs rejected.
"""
import os, sys, glob, json, datetime
import numpy as np
import cv2
import tensorflow as tf

SCRATCH    = "/scratch/users/jfernandezmartinez"
PROJECT    = os.path.join(SCRATCH, "CloudGAN")
WEIGHTS_AE = os.path.join(PROJECT, "weights", "ae_checkpoint.h5")
TEST_IMG   = os.path.join(PROJECT, "datasets", "38-cloud", "test", "img", "data")
TEST_MASK  = os.path.join(PROJECT, "datasets", "38-cloud", "test", "mask", "data")
OUTPUT_DIR = os.path.join(PROJECT, "evaluation", "results", "uncertainty_filter")
RESULT_JSON = os.path.join(PROJECT, "evaluation", "results", "uncertainty_filter.json")
RESULT_CSV  = os.path.join(PROJECT, "evaluation", "results", "uncertainty_filter.csv")

# ── Filter thresholds (from paper: ~40% cloud = unreliable) ──
COVERAGE_THRESHOLD    = 0.40   # reject if predicted cloud coverage > 40%
UNCERTAINTY_THRESHOLD = 0.15   # reject if > 15% of pixels are in [0.3, 0.7]
NUM_VISUAL_SAMPLES    = 10     # save N examples of ACCEPT and REJECT each

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "cloud_detection", "networks"))
from autoencoder import AE

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  CloudGAN Uncertainty Filter Evaluation")
print("=" * 60)
print("  Coverage threshold:    > {:.0f}% -> REJECT".format(COVERAGE_THRESHOLD * 100))
print("  Uncertainty threshold: > {:.0f}% uncertain pixels -> REJECT".format(
    UNCERTAINTY_THRESHOLD * 100))

# ── Load AE ──
print("\n[1/3] Loading AE model...")
model = AE(256, "relu", "sigmoid")
model.build(input_shape=(1, 256, 256, 3))
model.load_weights(WEIGHTS_AE)
sess = tf.compat.v1.keras.backend.get_session()
input_ph  = tf.compat.v1.placeholder(tf.float32, shape=(1, 256, 256, 3))
output_op = model(input_ph, training=False)
print("  AE loaded.")

# ── Scan test images ──
print("\n[2/3] Running filter on test set...")
img_files = sorted(glob.glob(os.path.join(TEST_IMG, "*.png")))
print("  Found {} test images".format(len(img_files)))

records = []
accepted_saved, rejected_saved = 0, 0

for i, img_path in enumerate(img_files):
    basename = os.path.basename(img_path)
    gt_mask_path = os.path.join(TEST_MASK, basename)
    if not os.path.exists(gt_mask_path):
        continue

    img_bgr   = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_batch = np.expand_dims(img_bgr.astype(np.float32) / 255.0, axis=0)

    # Run AE
    soft = sess.run(output_op, feed_dict={input_ph: img_batch})
    soft_1ch = soft[0, :, :, 0]  # (256, 256) values in [0, 1]

    # ── Metric 1: predicted cloud coverage ──
    pred_mask  = (soft_1ch > 0.5).astype(np.uint8)
    coverage   = float(np.mean(pred_mask))

    # ── Metric 2: AE uncertainty (pixels near 0.5) ──
    uncertain_pixels = float(np.mean((soft_1ch > 0.3) & (soft_1ch < 0.7)))

    # ── Metric 3: pixel-level entropy (optional, richer signal) ──
    eps     = 1e-7
    entropy = -soft_1ch * np.log(soft_1ch + eps) - (1 - soft_1ch) * np.log(1 - soft_1ch + eps)
    mean_entropy = float(np.mean(entropy))  # max is log(2)~0.693 at p=0.5

    # ── Filter decision ──
    reject_coverage    = coverage > COVERAGE_THRESHOLD
    reject_uncertainty = uncertain_pixels > UNCERTAINTY_THRESHOLD
    decision = "REJECT" if (reject_coverage or reject_uncertainty) else "ACCEPT"

    # ── GT mask stats for reference ──
    gt_gray = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_coverage = float(np.mean(gt_gray > 127))

    rec = {
        "filename": basename,
        "predicted_coverage": round(coverage, 4),
        "gt_coverage": round(gt_coverage, 4),
        "uncertain_pixels": round(uncertain_pixels, 4),
        "mean_entropy": round(mean_entropy, 6),
        "reject_coverage": reject_coverage,
        "reject_uncertainty": reject_uncertainty,
        "decision": decision,
    }
    records.append(rec)

    # ── Save visual examples ──
    save_this = (
        (decision == "ACCEPT" and accepted_saved < NUM_VISUAL_SAMPLES) or
        (decision == "REJECT" and rejected_saved < NUM_VISUAL_SAMPLES)
    )
    if save_this:
        soft_heat = cv2.applyColorMap((soft_1ch * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_3ch  = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        gt_3ch    = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2BGR)

        # Uncertainty map: bright where uncertain (0.3-0.7)
        unc_map = ((soft_1ch > 0.3) & (soft_1ch < 0.7)).astype(np.uint8) * 255
        unc_3ch = cv2.cvtColor(unc_map, cv2.COLOR_GRAY2BGR)
        unc_3ch[:, :, 1] = 0  # make uncertain pixels cyan/blue

        label = "{} | cov={:.0f}% | unc={:.0f}%".format(
            decision, coverage * 100, uncertain_pixels * 100)

        composite = np.concatenate([img_bgr, soft_heat, unc_3ch, pred_3ch, gt_3ch], axis=1)
        # Add label bar
        label_bar = np.zeros((30, composite.shape[1], 3), dtype=np.uint8)
        color = (0, 200, 0) if decision == "ACCEPT" else (0, 0, 200)
        cv2.putText(label_bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        composite = np.vstack([label_bar, composite])

        tag = "ACCEPT" if decision == "ACCEPT" else "REJECT"
        idx = accepted_saved if decision == "ACCEPT" else rejected_saved
        out_name = "{}_{:02d}_{:.0f}pct_{}.png".format(
            tag, idx, coverage * 100, os.path.splitext(basename)[0][:30])
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), composite)

        if decision == "ACCEPT":
            accepted_saved += 1
        else:
            rejected_saved += 1

    if (i + 1) % 200 == 0:
        print("  Processed {}/{}...".format(i + 1, len(img_files)))

print("  Total processed: {}".format(len(records)))

# ── Statistics ──
print("\n[3/3] Computing statistics...")
n_total    = len(records)
n_accept   = sum(1 for r in records if r["decision"] == "ACCEPT")
n_reject   = sum(1 for r in records if r["decision"] == "REJECT")
n_rej_cov  = sum(1 for r in records if r["reject_coverage"])
n_rej_unc  = sum(1 for r in records if r["reject_uncertainty"])

avg_cov_accept = np.mean([r["predicted_coverage"] for r in records if r["decision"] == "ACCEPT"])
avg_cov_reject = np.mean([r["predicted_coverage"] for r in records if r["decision"] == "REJECT"])
avg_unc_accept = np.mean([r["uncertain_pixels"] for r in records if r["decision"] == "ACCEPT"])
avg_unc_reject = np.mean([r["uncertain_pixels"] for r in records if r["decision"] == "REJECT"])

# Coverage distribution per bin
bins = [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
        (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
bin_stats = []
for lo, hi in bins:
    in_bin = [r for r in records if lo <= r["predicted_coverage"] < hi]
    accepted = sum(1 for r in in_bin if r["decision"] == "ACCEPT")
    bin_stats.append({
        "bin": "{:.0f}-{:.0f}%".format(lo*100, hi*100),
        "total": len(in_bin),
        "accepted": accepted,
        "rejected": len(in_bin) - accepted,
    })

summary = {
    "thresholds": {
        "coverage": COVERAGE_THRESHOLD,
        "uncertainty": UNCERTAINTY_THRESHOLD,
    },
    "total_images": n_total,
    "accepted": n_accept,
    "rejected": n_reject,
    "rejected_by_coverage": n_rej_cov,
    "rejected_by_uncertainty": n_rej_unc,
    "avg_coverage_accepted": round(float(avg_cov_accept), 4),
    "avg_coverage_rejected": round(float(avg_cov_reject), 4),
    "avg_uncertainty_accepted": round(float(avg_unc_accept), 4),
    "avg_uncertainty_rejected": round(float(avg_unc_reject), 4),
    "bin_distribution": bin_stats,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(RESULT_JSON, 'w') as f:
    json.dump({"summary": summary, "per_image": records}, f, indent=2)

# CSV for easy analysis
with open(RESULT_CSV, 'w') as f:
    f.write("filename,predicted_coverage,gt_coverage,uncertain_pixels,mean_entropy,decision\n")
    for r in records:
        f.write("{},{},{},{},{},{}\n".format(
            r["filename"], r["predicted_coverage"], r["gt_coverage"],
            r["uncertain_pixels"], r["mean_entropy"], r["decision"]))

# Print summary
sep = "=" * 60
print("""
{0}
  UNCERTAINTY FILTER RESULTS
{0}
  Total images:    {1}
  ACCEPTED:        {2} ({3:.1f}%)
  REJECTED:        {4} ({5:.1f}%)
    - by coverage: {6}
    - by uncert.:  {7}

  Avg coverage ACCEPTED: {8:.1f}%
  Avg coverage REJECTED: {9:.1f}%
  Avg uncert.  ACCEPTED: {10:.1f}%
  Avg uncert.  REJECTED: {11:.1f}%
{0}
  Coverage bin breakdown:
""".format(sep, n_total,
           n_accept, n_accept/n_total*100,
           n_reject, n_reject/n_total*100,
           n_rej_cov, n_rej_unc,
           avg_cov_accept*100, avg_cov_reject*100,
           avg_unc_accept*100, avg_unc_reject*100))

for b in bin_stats:
    print("  {:10s} total={:4d} accept={:4d} reject={:4d}".format(
        b["bin"], b["total"], b["accepted"], b["rejected"]))

print("\n  JSON: {}".format(RESULT_JSON))
print("  CSV:  {}".format(RESULT_CSV))
print("  Visuals: {}".format(OUTPUT_DIR))
print(sep)
print("Done!")
