#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Paths
PROJECT_ROOT = "/scratch/users/jfernandezmartinez/CloudGAN"
CHECKPOINT_DIR = f"{PROJECT_ROOT}/weights/SN_PatchGAN"
CHECKPOINT_PREFIX = "snap-1132000"
OUT_DIR = f"{PROJECT_ROOT}/evaluation/results"

sys.path.insert(0, PROJECT_ROOT)
from cloud_removal.inpaint_model import InpaintCAModel

# ---------------- FLAGS ---------------- #
class FLAGS:
    guided = False
    edge_threshold = 0.6
    padding = "SAME"
    viz_max_out = 1

    height = 256
    width = 256
    img_shapes = [256, 256, 3]

    max_delta_height = 32
    max_delta_width = 32
    vertical_margin = 0
    horizontal_margin = 0

    batch_size = 1

    l1_loss_alpha = 1.0
    gan_loss_alpha = 1.0
    ae_loss = True
    gan = "sngan"
    gan_with_mask = True

    random_seed = False


# ---------------- UTILS ---------------- #

def count_params():
    total = 0
    for v in tf.compat.v1.trainable_variables():
        shape = v.get_shape().as_list()
        n = 1
        for d in shape:
            if d is not None:
                n *= d
        total += n
    return total


def get_checkpoint():
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        return latest
    return os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)


def get_gpu_power():
    """Returns power draw in Watts using nvidia-smi"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        )
        return float(out.decode().strip().split("\n")[0])
    except:
        return None


def get_gpu_memory():
    """Returns used memory in MB"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return float(out.decode().strip().split("\n")[0])
    except:
        return None


# ---------------- MAIN ---------------- #

def evaluate():
    print("=" * 60)
    print("SN-PATCHGAN FULL EVALUATION")
    print("=" * 60)

    # 1) Size
    total_size = 0
    for f in os.listdir(CHECKPOINT_DIR):
        if CHECKPOINT_PREFIX in f:
            total_size += os.path.getsize(os.path.join(CHECKPOINT_DIR, f))
    size_mb = total_size / (1024 ** 2)
    print(f"[1/6] Size: {size_mb:.2f} MB")

    # 2) Build graph
    print("[2/6] Building graph...")
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[1, 256, 256, 3], name="input"
        )

        model = InpaintCAModel()
        output = model.build_static_infer_graph(FLAGS, input_ph, name="val")

        total_params = count_params()
        print(f"  Parameters: {total_params:,}")

        # FLOPs
        print("[3/6] Computing FLOPs...")
        from tensorflow.python.profiler import model_analyzer
        from tensorflow.python.profiler import option_builder

        opts = option_builder.ProfileOptionBuilder.float_operation()
        flops = model_analyzer.profile(graph, options=opts)
        total_flops = flops.total_float_ops if flops is not None else 0
        print(f"  FLOPs: {total_flops:,}")

        # Session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        checkpoint = get_checkpoint()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, checkpoint)
            print("[4/6] Checkpoint restored")

            dummy = np.random.randint(0, 256, (1, 256, 256, 3)).astype(np.float32)
            feed = {input_ph: dummy}

            # Warmup
            for _ in range(10):
                sess.run(output, feed_dict=feed)

            # Measure GPU memory
            mem_before = get_gpu_memory()

            # Measure power baseline
            power_before = get_gpu_power()

            # Timing
            print("[5/6] Measuring inference...")
            times = []
            power_samples = []

            for i in range(100):
                p = get_gpu_power()
                if p:
                    power_samples.append(p)

                start = time.perf_counter()
                sess.run(output, feed_dict=feed)
                times.append((time.perf_counter() - start) * 1000)

            mem_after = get_gpu_memory()

            mean_ms = float(np.mean(times))
            fps = float(1000.0 / mean_ms)

            avg_power = np.mean(power_samples) if power_samples else None

            # Energy per inference (Joules)
            energy = None
            if avg_power:
                energy = avg_power * (mean_ms / 1000.0)

            print(f"  Latency: {mean_ms:.2f} ms")
            print(f"  FPS: {fps:.2f}")
            print(f"  Avg Power: {avg_power:.2f} W" if avg_power else "  Power: N/A")
            print(f"  Energy per inference: {energy:.4f} J" if energy else "  Energy: N/A")

    # 6) Save
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {
        "model": "SN-PatchGAN",
        "parameters": int(total_params),
        "size_mb": size_mb,
        "flops": int(total_flops),
        "latency_ms": mean_ms,
        "fps": fps,
        "gpu_memory_before_mb": mem_before,
        "gpu_memory_after_mb": mem_after,
        "avg_power_w": avg_power,
        "energy_per_inference_j": energy,
    }

    with open(os.path.join(OUT_DIR, "snpatchgan_full.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[6/6] Results saved")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()