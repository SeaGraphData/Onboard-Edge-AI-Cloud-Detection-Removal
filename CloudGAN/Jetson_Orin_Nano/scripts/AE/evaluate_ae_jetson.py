#!/usr/bin/env python3
import os, sys, json, time, subprocess, threading, re
import numpy as np
import tensorflow as tf

# ── TEGRASTATS MONITOR ──────────────────────────────────────
class TegraStatsMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.lines = []; self.proc = None; self._stop = False

    def start(self):
        self.lines = []; self._stop = False
        self.proc = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if self._stop: break
            self.lines.append(line.decode('utf-8', errors='ignore'))

    def stop(self):
        self._stop = True
        if self.proc: self.proc.terminate(); self.proc.wait()

    def _parse_mw(self, key):
        vals = [float(re.search(rf'{key} (\d+)mW', l).group(1)) / 1000.0
                for l in self.lines if re.search(rf'{key} (\d+)mW', l)]
        return float(sum(vals)/len(vals)) if vals else None

    def total_power_w(self):   return self._parse_mw('VDD_IN')
    def gpu_cpu_power_w(self): return self._parse_mw('VDD_CPU_GPU_CV')
    def ram_used_mb(self):
        for line in reversed(self.lines):
            m = re.search(r'RAM (\d+)/\d+MB', line)
            if m: return float(m.group(1))
        return None
# ────────────────────────────────────────────────────────────

PROJECT_ROOT = "/home/sigcomjetson/Desktop/CloudGAN"
WEIGHTS  = f"{PROJECT_ROOT}/weights/ae_checkpoint.h5"
OUT_DIR  = f"{PROJECT_ROOT}/evaluation/results"

sys.path.insert(0, f"{PROJECT_ROOT}/cloud_detection/networks")
from autoencoder import AE

def count_params(model):
    return int(np.sum([tf.keras.backend.count_params(w)
                       for w in model.trainable_weights]))

def compute_flops():
    graph = tf.Graph()
    with graph.as_default():
        fresh_model = AE(input_res=256, activation="relu",
                         activation_out="sigmoid", seed=42)
        inputs = tf.compat.v1.placeholder(tf.float32, shape=(1, 256, 256, 3))
        _ = fresh_model(inputs)
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            from tensorflow.python.profiler import model_analyzer, option_builder
            opts  = option_builder.ProfileOptionBuilder.float_operation()
            flops = model_analyzer.profile(graph, options=opts)
    return flops.total_float_ops if flops else 0

def evaluate_ae():
    print("=" * 60)
    print("AUTOENCODER EVALUATION — Jetson Orin Nano")
    print("=" * 60)

    size_mb = os.path.getsize(WEIGHTS) / (1024**2)
    print(f"[1/6] Size: {size_mb:.2f} MB")

    print("[2/6] Building model...")
    model = AE(input_res=256, activation="relu",
               activation_out="sigmoid", seed=42)
    _ = model(np.zeros((1, 256, 256, 3), dtype=np.float32), training=False)
    model.load_weights(WEIGHTS)
    print("  Model loaded")

    total_params = count_params(model)
    print(f"[3/6] Parameters: {total_params:,}")

    print("[4/6] Computing FLOPs...")
    total_flops = compute_flops()
    print(f"  FLOPs: {total_flops:,}")

    print("[5/6] Measuring inference (100 runs) + tegrastats...")
    dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        model.predict(dummy, verbose=0)

    monitor = TegraStatsMonitor(interval_ms=100)
    monitor.start()
    time.sleep(0.5)

    times = []
    for i in range(100):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    time.sleep(0.3)
    monitor.stop()

    mean_ms       = float(np.mean(times))
    fps           = float(1000.0 / mean_ms)
    total_power   = monitor.total_power_w()
    gpu_cpu_power = monitor.gpu_cpu_power_w()
    ram_used      = monitor.ram_used_mb()
    energy        = gpu_cpu_power * (mean_ms / 1000.0) if gpu_cpu_power else None

    print(f"  Latency: {mean_ms:.2f} ms  |  FPS: {fps:.2f}")
    print(f"  Total board power : {total_power:.2f} W"   if total_power   else "  Total power: N/A")
    print(f"  CPU+GPU+CV power  : {gpu_cpu_power:.2f} W" if gpu_cpu_power  else "  GPU power: N/A")
    print(f"  RAM used          : {ram_used:.0f} MB"      if ram_used      else "  RAM: N/A")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "model": "Autoencoder",
        "parameters": total_params,
        "size_mb": size_mb,
        "flops": int(total_flops),
        "latency_ms": mean_ms,
        "fps": fps,
        "ram_used_mb": ram_used,
        "total_board_power_w": total_power,
        "gpu_cpu_power_w": gpu_cpu_power,
        "energy_per_inference_j": energy,
    }
    out_path = os.path.join(OUT_DIR, "ae_jetson.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[6/6] Guardado en {out_path}")
    print("=" * 60)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    evaluate_ae()
