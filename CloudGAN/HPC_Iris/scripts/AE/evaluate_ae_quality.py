#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_ae_quality.py — Cloud Detection Quality Metrics for AE
"""
import os, sys, json, glob, datetime
import numpy as np
import cv2
import tensorflow as tf

SCRATCH    = "/scratch/users/jfernandezmartinez"
PROJECT    = os.path.join(SCRATCH, "CloudGAN")
WEIGHTS    = os.path.join(PROJECT, "weights", "ae_checkpoint.h5")
TEST_IMG   = os.path.join(PROJECT, "datasets", "38-cloud", "test", "img", "data")
TEST_MASK  = os.path.join(PROJECT, "datasets", "38-cloud", "test", "mask", "data")
OUTPUT_DIR = os.path.join(PROJECT, "evaluation", "results", "ae_quality_images")
RESULT_JSON = os.path.join(PROJECT, "evaluation", "results", "ae_quality.json")
RESULT_TXT  = os.path.join(PROJECT, "evaluation", "results", "ae_quality.txt")
NUM_SAMPLES_TO_SAVE = 20

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "cloud_detection", "networks"))
from autoencoder import AE

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)

print("=" * 60)
print("  AUTOENCODER — Cloud Detection Quality Evaluation")
print("=" * 60)

# ── Load model and get Keras session ──
print("\n[1/4] Loading AE model...")
assert os.path.exists(WEIGHTS), "AE weights not found: {}".format(WEIGHTS)
model = AE(256, "relu", "sigmoid")
model.build(input_shape=(1, 256, 256, 3))
model.load_weights(WEIGHTS)
print("  Model loaded from: {}".format(WEIGHTS))

sess = tf.compat.v1.keras.backend.get_session()
input_ph  = tf.compat.v1.placeholder(tf.float32, shape=(1, 256, 256, 3))
output_op = model(input_ph, training=False)

# ── Discover test images ──
print("\n[2/4] Discovering test images...")
img_files = sorted(glob.glob(os.path.join(TEST_IMG, "*.png")))
print("  Found {} test images".format(len(img_files)))
assert len(img_files) > 0, "No test images found!"

# ── Inference + Metrics ──
print("\n[3/4] Running inference on test set...")
total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
per_image_acc, per_image_prec, per_image_rec, per_image_f1 = [], [], [], []
per_image_iou_c, per_image_iou_nc = [], []
num_processed, num_saved = 0, 0

for i, img_path in enumerate(img_files):
    basename = os.path.basename(img_path)
    mask_gt_path = os.path.join(TEST_MASK, basename)
    if not os.path.exists(mask_gt_path):
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_norm = img_bgr.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    mask_gt_raw = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
    if mask_gt_raw is None:
        continue
    mask_gt_bin = (mask_gt_raw > 127).astype(np.uint8)

    soft_mask = sess.run(output_op, feed_dict={input_ph: img_batch})
    pred_mask = (soft_mask[0, :, :, 0] > 0.5).astype(np.uint8)

    TP = np.sum((pred_mask == 1) & (mask_gt_bin == 1))
    TN = np.sum((pred_mask == 0) & (mask_gt_bin == 0))
    FP = np.sum((pred_mask == 1) & (mask_gt_bin == 0))
    FN = np.sum((pred_mask == 0) & (mask_gt_bin == 1))
    total_TP += TP; total_TN += TN; total_FP += FP; total_FN += FN

    acc  = (TP + TN) / float(TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    prec = TP / float(TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou_c  = TP / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    iou_nc = TN / float(TN + FN + FP) if (TN + FN + FP) > 0 else 0.0

    per_image_acc.append(acc); per_image_prec.append(prec)
    per_image_rec.append(rec); per_image_f1.append(f1)
    per_image_iou_c.append(iou_c); per_image_iou_nc.append(iou_nc)
    num_processed += 1

    if num_saved < NUM_SAMPLES_TO_SAVE:
        pred_vis_3ch = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        gt_vis_3ch   = cv2.cvtColor((mask_gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        soft_vis_3ch = cv2.applyColorMap((soft_mask[0,:,:,0]*255).astype(np.uint8), cv2.COLORMAP_JET)
        composite = np.concatenate([img_bgr, soft_vis_3ch, pred_vis_3ch, gt_vis_3ch], axis=1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "ae_{:04d}_{}.png".format(num_saved, os.path.splitext(basename)[0])), composite)
        num_saved += 1

    if (i + 1) % 100 == 0 or (i + 1) == len(img_files):
        print("  Processed {}/{}...".format(i + 1, len(img_files)))

print("  Total processed: {}  |  Samples saved: {}".format(num_processed, num_saved))

# ── Compute metrics ──
print("\n[4/4] Computing final metrics...")
g_acc  = (total_TP+total_TN) / float(total_TP+total_TN+total_FP+total_FN)
g_prec = total_TP/float(total_TP+total_FP) if (total_TP+total_FP)>0 else 0.0
g_rec  = total_TP/float(total_TP+total_FN) if (total_TP+total_FN)>0 else 0.0
g_f1   = 2.0*g_prec*g_rec/(g_prec+g_rec) if (g_prec+g_rec)>0 else 0.0
g_iou_c  = total_TP/float(total_TP+total_FP+total_FN) if (total_TP+total_FP+total_FN)>0 else 0.0
g_iou_nc = total_TN/float(total_TN+total_FN+total_FP) if (total_TN+total_FN+total_FP)>0 else 0.0
g_miou = (g_iou_c + g_iou_nc) / 2.0

m_acc=float(np.mean(per_image_acc)); m_prec=float(np.mean(per_image_prec))
m_rec=float(np.mean(per_image_rec)); m_f1=float(np.mean(per_image_f1))
m_iou_c=float(np.mean(per_image_iou_c)); m_iou_nc=float(np.mean(per_image_iou_nc))
m_miou = (m_iou_c + m_iou_nc) / 2.0

results = {
    "model": "Autoencoder (AE)", "task": "Cloud Detection",
    "dataset": "38-cloud (test set)", "num_images": num_processed,
    "weights": WEIGHTS,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "threshold": 0.5,
    "global_metrics": {
        "accuracy": round(g_acc,6), "precision": round(g_prec,6),
        "recall": round(g_rec,6), "f1_score": round(g_f1,6),
        "iou_cloud": round(g_iou_c,6), "iou_non_cloud": round(g_iou_nc,6),
        "mIoU": round(g_miou,6),
    },
    "macro_avg_metrics": {
        "accuracy": round(m_acc,6), "precision": round(m_prec,6),
        "recall": round(m_rec,6), "f1_score": round(m_f1,6),
        "iou_cloud": round(m_iou_c,6), "iou_non_cloud": round(m_iou_nc,6),
        "mIoU": round(m_miou,6),
    },
    "confusion_matrix": {"TP":int(total_TP),"TN":int(total_TN),"FP":int(total_FP),"FN":int(total_FN)},
}

with open(RESULT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

sep = "=" * 60
summary = """
{0}
  AUTOENCODER — RESULTS SUMMARY
{0}
  Images evaluated: {1}
  --- Global (Micro-Avg) ---
  Accuracy:      {2:.4f}
  Precision:     {3:.4f}
  Recall:        {4:.4f}
  F1-Score:      {5:.4f}
  IoU (cloud):   {6:.4f}
  mIoU:          {7:.4f}
  --- Macro-Average ---
  Accuracy:      {8:.4f}
  Precision:     {9:.4f}
  Recall:        {10:.4f}
  F1-Score:      {11:.4f}
  IoU (cloud):   {12:.4f}
  mIoU:          {13:.4f}
{0}
  Confusion Matrix: TP={14:,} TN={15:,} FP={16:,} FN={17:,}
  Sample images: {18}
{0}
""".format(sep, num_processed, g_acc, g_prec, g_rec, g_f1, g_iou_c, g_miou,
           m_acc, m_prec, m_rec, m_f1, m_iou_c, m_miou,
           int(total_TP), int(total_TN), int(total_FP), int(total_FN), OUTPUT_DIR)

with open(RESULT_TXT, 'w') as f:
    f.write(summary)
print(summary)
print("JSON saved: {}".format(RESULT_JSON))
print("TXT saved: {}".format(RESULT_TXT))
print("Done!")
