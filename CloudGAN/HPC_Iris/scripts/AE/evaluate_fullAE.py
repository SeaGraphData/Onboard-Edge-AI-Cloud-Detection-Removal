#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import numpy as np
import tensorflow as tf

# Path to your AE model
sys.path.insert(0, '/scratch/users/jfernandezmartinez/CloudGAN/cloud_detection/networks')
from autoencoder import AE

# Paths
WEIGHTS = "/scratch/users/jfernandezmartinez/CloudGAN/output/AE/checkpoint.h5"
OUT_DIR = "/scratch/users/jfernandezmartinez/CloudGAN/evaluation/results"

# ---------------- UTILS ---------------- #

def get_gpu_power():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        )
        return float(out.decode().strip().split("\n")[0])
    except:
        return None


def get_gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return float(out.decode().strip().split("\n")[0])
    except:
        return None


def count_params(model):
    return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))


def compute_flops(input_shape=(1, 256, 256, 3)):
    """
    Compute FLOPs using an isolated TF1 graph.
    Builds the model inside the graph context BEFORE opening the session,
    so we never touch set_session and never corrupt the global Keras state.
    """
    graph = tf.Graph()
    with graph.as_default():
        # Build and trace the graph BEFORE opening any session
        fresh_model = AE(
            input_res=256,
            activation="relu",
            activation_out="sigmoid",
            seed=42
        )
        inputs = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        _ = fresh_model(inputs)

        # Session is only used for the profiler — no set_session call
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            from tensorflow.python.profiler import model_analyzer
            from tensorflow.python.profiler import option_builder

            opts = option_builder.ProfileOptionBuilder.float_operation()
            flops = model_analyzer.profile(graph, options=opts)

    return flops.total_float_ops if flops is not None else 0


# ---------------- MAIN ---------------- #

def evaluate_ae():
    print("=" * 60)
    print("AUTOENCODER FULL EVALUATION")
    print("=" * 60)

    # 1) Size
    size_mb = os.path.getsize(WEIGHTS) / (1024 ** 2)
    print(f"[1/6] Size: {size_mb:.2f} MB")

    # 2) Build model
    print("[2/6] Building model...")
    model = AE(input_res=256, activation="relu", activation_out="sigmoid", seed=42)
    model.build((None, 256, 256, 3))
    model.load_weights(WEIGHTS)
    print("  Model loaded")

    # 3) Parameters
    total_params = count_params(model)
    print(f"[3/6] Parameters: {total_params:,}")

    # 4) FLOPs
    print("[4/6] Computing FLOPs...")
    total_flops = compute_flops()
    print(f"  FLOPs: {total_flops:,}")

    # 5) Inference + GPU metrics
    print("[5/6] Measuring performance...")

    dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)

    # Warm-up
    for _ in range(10):
        _ = model.predict(dummy, verbose=0)

    mem_before = get_gpu_memory()
    power_samples = []

    times = []
    for i in range(100):
        p = get_gpu_power()
        if p:
            power_samples.append(p)

        start = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - start) * 1000)

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/100")

    mem_after = get_gpu_memory()

    mean_ms = float(np.mean(times))
    fps = float(1000.0 / mean_ms)

    avg_power = np.mean(power_samples) if power_samples else None
    energy = None
    if avg_power:
        energy = avg_power * (mean_ms / 1000.0)

    print(f"  Latency: {mean_ms:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Avg Power: {avg_power:.2f} W" if avg_power else "  Power: N/A")
    print(f"  Energy: {energy:.4f} J" if energy else "  Energy: N/A")

    # 6) Save
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {
        "model": "Autoencoder",
        "parameters": total_params,
        "size_mb": size_mb,
        "flops": int(total_flops),
        "latency_ms": mean_ms,
        "fps": fps,
        "gpu_memory_before_mb": mem_before,
        "gpu_memory_after_mb": mem_after,
        "avg_power_w": avg_power,
        "energy_per_inference_j": energy,
    }

    with open(os.path.join(OUT_DIR, "ae_full.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUT_DIR, "ae_full.txt"), "w") as f:
        f.write("AUTOENCODER FULL EVALUATION\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Size: {size_mb:.2f} MB\n")
        f.write(f"FLOPs: {int(total_flops):,}\n")
        f.write(f"Latency: {mean_ms:.2f} ms\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"GPU Memory Before: {mem_before} MB\n")
        f.write(f"GPU Memory After: {mem_after} MB\n")
        f.write(f"Avg Power: {avg_power} W\n")
        f.write(f"Energy per inference: {energy} J\n")

    print("[6/6] Results saved")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_ae()