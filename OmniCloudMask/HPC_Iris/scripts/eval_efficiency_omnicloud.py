import json
import time
import numpy as np
import torch
from pathlib import Path
from omnicloudmask import predict_from_array
import omnicloudmask.cloud_mask as cm

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_OK = True
    print("[INFO] pynvml OK")
except Exception as e:
    PYNVML_OK = False
    GPU_HANDLE = None
    print(f"[WARN] pynvml not available: {e}")

try:
    from thop import profile as thop_profile
    THOP_OK = True
    print("[INFO] thop OK")
except Exception as e:
    THOP_OK = False
    print(f"[WARN] thop not available: {e}")

OUT_DIR = Path("/scratch/users/jfernandezmartinez/GeoAI/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_obj = torch.device(device)
print(f"[INFO] Device : {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

H, W = 384, 384
dummy_np = np.random.rand(3, H, W).astype(np.float32)

# ── Warmup ─────────────────────────────────────────────────────────────────────
print("[INFO] Warmup (loads + caches both models)...")
_ = predict_from_array(dummy_np, inference_dtype="float32")
print("[INFO] Warmup done.")

# ── Cargar modelos directamente via collect_models ────────────────────────────
print("[INFO] Loading models via collect_models...")
models = cm.collect_models(
    custom_models=None,
    inference_device=device_obj,
    inference_dtype=torch.float32,
    source="hugging_face",
)
print(f"[INFO] Loaded {len(models)} models in ensemble")
for i, m in enumerate(models):
    n = sum(p.numel() for p in m.parameters())
    print(f"  Model {i+1}: {type(m).__class__.__name__} — {n/1e6:.3f} M params")

# ── FLOPs con thop — suma de los dos modelos ───────────────────────────────────
gflops_total  = None
mparams_total = None
model_details = []

if THOP_OK and torch.cuda.is_available():
    print("[INFO] Measuring FLOPs for each model...")
    total_macs   = 0
    total_params = 0
    dummy_t = torch.randn(1, 3, H, W).to(device_obj)

    for i, model in enumerate(models):
        try:
            model.eval()
            macs, params = thop_profile(model, inputs=(dummy_t,), verbose=False)
            gf = round(macs * 2 / 1e9, 3)
            mp = round(params / 1e6, 3)
            total_macs   += macs
            total_params += params
            print(f"  Model {i+1}: {gf} GFLOPs, {mp} M params")
            model_details.append({"gflops": gf, "params_M": mp})
        except Exception as e:
            print(f"  Model {i+1}: FLOPs failed — {e}")

    gflops_total  = round(total_macs   * 2 / 1e9, 3)
    mparams_total = round(total_params / 1e6,      3)
    print(f"[INFO] Total ensemble GFLOPs : {gflops_total}")
    print(f"[INFO] Total ensemble Params : {mparams_total} M")

# ── Latencia y Power: 100 inferencias ─────────────────────────────────────────
print("[INFO] Measuring latency and power (100 inferences)...")
N = 100
latencies = []
powers    = []
gpu_mems  = []

for i in range(N):
    if PYNVML_OK:
        try:
            pw = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
        except:
            pw = 0.0
    else:
        pw = 0.0

    t0 = time.time()
    _ = predict_from_array(dummy_np, inference_dtype="float32")
    lat = time.time() - t0

    latencies.append(lat)
    powers.append(pw)

    if torch.cuda.is_available():
        gpu_mems.append(torch.cuda.memory_allocated() / 1024**2)

    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{N}] lat={lat*1000:.1f}ms  power={pw:.1f}W")

avg_latency  = float(np.mean(latencies))
std_latency  = float(np.std(latencies))
fps          = 1.0 / avg_latency if avg_latency > 0 else 0
avg_power    = float(np.mean(powers))
avg_energy   = avg_power * avg_latency * 1000
avg_gpu_mem  = float(np.mean(gpu_mems)) if gpu_mems else 0.0
peak_gpu_mem = float(np.max(gpu_mems))  if gpu_mems else 0.0

print(f"\n[RESULTS]")
print(f"  GFLOPs (ensemble)  : {gflops_total}")
print(f"  Params (ensemble)  : {mparams_total} M")
print(f"  Avg latency        : {avg_latency*1000:.2f} ms")
print(f"  Std latency        : {std_latency*1000:.2f} ms")
print(f"  FPS                : {fps:.2f}")
print(f"  Avg power          : {avg_power:.2f} W")
print(f"  Avg energy         : {avg_energy:.3f} mJ/inference")
print(f"  Avg GPU memory     : {avg_gpu_mem:.1f} MB")
print(f"  Peak GPU memory    : {peak_gpu_mem:.1f} MB")

# ── Guardar JSON ───────────────────────────────────────────────────────────────
efficiency = {
    "model":            "OmniCloudMask v1.7.1",
    "architecture":     "Ensemble: smp.Unet(regnety_004) + smp.Unet(edgenext_small)",
    "device":           device,
    "gpu":              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "input_shape":      [3, H, W],
    "n_measure_runs":   N,
    "ensemble_models":  model_details,
    "gflops_total":     gflops_total,
    "params_M_total":   mparams_total,
    "avg_latency_ms":   round(avg_latency * 1000, 3),
    "std_latency_ms":   round(std_latency * 1000, 3),
    "fps":              round(fps, 2),
    "avg_power_w":      round(avg_power, 2),
    "avg_energy_mj":    round(avg_energy, 3),
    "avg_gpu_mem_mb":   round(avg_gpu_mem, 1),
    "peak_gpu_mem_mb":  round(peak_gpu_mem, 1),
}

out_json = OUT_DIR / "efficiency_omnicloudmask.json"
with open(out_json, "w") as f:
    json.dump(efficiency, f, indent=2)

print(f"\n[DONE] Saved to {out_json}")
