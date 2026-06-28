import json
import time
import numpy as np
import torch
from pathlib import Path
from omnicloudmask import predict_from_array
import omnicloudmask.cloud_mask as cm
import threading

# ── tegrastats power sampler ───────────────────────────────────────────────────
import subprocess

class TegraPowerSampler:
    def __init__(self, interval=0.1):
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
            try:
                result = subprocess.run(
                    ["cat", "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input"],
                    capture_output=True, text=True, timeout=1
                )
                # Intentar leer potencia total del sistema
                power_result = subprocess.run(
                    ["tegrastats", "--interval", "100"],
                    capture_output=True, text=True, timeout=0.5
                )
            except:
                pass
            time.sleep(self.interval)

    def avg_power_w(self):
        return 0.0  # tegrastats necesita parsing especial, reportamos 0 si no disponible

def get_tegrastats_power():
    """Lee potencia instantanea via tegrastats."""
    try:
        proc = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        line = proc.stdout.readline()
        proc.kill()
        # Buscar VDD_IN o SOC en la linea
        import re
        matches = re.findall(r'VDD_IN\s+(\d+)mW', line)
        if matches:
            return float(matches[0]) / 1000.0  # mW -> W
        matches = re.findall(r'(\d+)mW/\d+mW', line)
        if matches:
            return float(matches[0]) / 1000.0
    except:
        pass
    return None

OUT_DIR = Path.home() / "omnicloud_efficiency"
OUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device : {device}")
print(f"[INFO] torch  : {torch.__version__}")
if torch.cuda.is_available():
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

H, W = 384, 384
dummy_np = np.random.rand(3, H, W).astype(np.float32)

# ── Warmup ─────────────────────────────────────────────────────────────────────
print("[INFO] Warmup...")
_ = predict_from_array(dummy_np, inference_dtype='float16')
print("[INFO] Warmup done.")

# ── FLOPs ──────────────────────────────────────────────────────────────────────
gflops  = None
mparams = None

try:
    from thop import profile as thop_profile
    print("[INFO] Measuring FLOPs...")
    device_obj = torch.device(device)
    models = cm.collect_models(
        custom_models=None,
        inference_device=device_obj,
        inference_dtype=torch.float32,
        source="hugging_face",
    )
    dummy_t     = torch.randn(1, 3, H, W).to(device_obj)
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

# ── Latencia y Power: 100 inferencias ─────────────────────────────────────────
print("[INFO] Measuring latency and power (100 inferences)...")
N = 100
latencies = []
powers    = []
ram_usage = []

for i in range(N):
    # Potencia via tegrastats
    pw = get_tegrastats_power()
    if pw is None:
        pw = 0.0

    t0  = time.time()
    _   = predict_from_array(dummy_np, inference_dtype='float16')
    lat = time.time() - t0

    latencies.append(lat)
    powers.append(pw)

    # RAM usage
    try:
        import psutil
        ram_usage.append(psutil.Process().memory_info().rss / 1024**2)
    except:
        ram_usage.append(0.0)

    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{N}] lat={lat*1000:.1f}ms  power={pw:.1f}W")

avg_latency  = float(np.mean(latencies))
std_latency  = float(np.std(latencies))
fps          = 1.0 / avg_latency if avg_latency > 0 else 0
avg_power    = float(np.mean([p for p in powers if p > 0])) if any(p > 0 for p in powers) else 0.0
avg_energy   = avg_power * avg_latency * 1000 if avg_power > 0 else None
avg_ram      = float(np.mean(ram_usage)) if ram_usage else 0.0

print(f"\n[RESULTS]")
print(f"  GFLOPs          : {gflops}")
print(f"  Params          : {mparams} M")
print(f"  Avg latency     : {avg_latency*1000:.2f} ms")
print(f"  Std latency     : {std_latency*1000:.2f} ms")
print(f"  FPS             : {fps:.2f}")
print(f"  Avg power       : {avg_power:.2f} W")
print(f"  Avg energy      : {avg_energy} mJ/inference")
print(f"  Avg RAM usage   : {avg_ram:.1f} MB")

efficiency = {
    "model":          "OmniCloudMask v1.7.1",
    "platform":       "Jetson Orin Nano",
    "device":         device,
    "gpu":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "jetpack":        "6.x (L4T R36.4.7)",
    "cuda_version":   torch.version.cuda,
    "torch_version":  torch.__version__,
    "input_shape":    [3, H, W],
    "n_runs":         N,
    "gflops":         gflops,
    "params_M":       mparams,
    "avg_latency_ms": round(avg_latency * 1000, 3),
    "std_latency_ms": round(std_latency * 1000, 3),
    "fps":            round(fps, 2),
    "avg_power_w":    round(avg_power, 2),
    "avg_energy_mj":  round(avg_energy, 3) if avg_energy else None,
    "avg_ram_mb":     round(avg_ram, 1),
}

out_json = OUT_DIR / "efficiency_omnicloud_jetson.json"
out_txt  = OUT_DIR / "efficiency_omnicloud_jetson.txt"

with open(out_json, "w") as f:
    json.dump(efficiency, f, indent=2)

with open(out_txt, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("OmniCloudMask — Jetson Orin Nano Efficiency\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Platform       : Jetson Orin Nano\n")
    f.write(f"JetPack        : 6.x (L4T R36.4.7)\n")
    f.write(f"CUDA           : {torch.version.cuda}\n")
    f.write(f"Device         : {device}\n\n")
    f.write(f"GFLOPs         : {gflops}\n")
    f.write(f"Parameters     : {mparams} M\n")
    f.write(f"Avg latency    : {avg_latency*1000:.2f} ms\n")
    f.write(f"Std latency    : {std_latency*1000:.2f} ms\n")
    f.write(f"FPS            : {fps:.2f}\n")
    f.write(f"Avg power      : {avg_power:.2f} W\n")
    f.write(f"Avg energy     : {avg_energy} mJ\n")
    f.write(f"Avg RAM        : {avg_ram:.1f} MB\n")

print(f"\n[DONE] Saved to {OUT_DIR}")
