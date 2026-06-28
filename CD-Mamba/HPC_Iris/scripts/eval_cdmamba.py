"""
CD-Mamba Performance Evaluation Script
Computes: Parameters, Size, FLOPs, Latency, FPS, GPU Memory, Power, Energy
No dataset required — uses a dummy input matching the expected 4-channel 384x384 input.
Evaluates all 4 pretrained checkpoints (one per fold) and reports per-fold + averaged results.
Output: console + .txt + .json
"""

import os
import sys
import json
import datetime
import numpy as np
import torch

# ── Add f01 to path so 'models.cloud' (and its sub-imports) are resolvable ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'f01'))

from models.cloud import cdMamba

# ── Optional: power measurement via pynvml ────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# ── Optional: FLOPs via fvcore ────────────────────────────────────────────────
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FVCORE_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'num_classes':    1,
    'input_channels': 4,
    'c_list':         [8, 16, 24, 32, 48, 64],
    'split_att':      'fc',
    'bridge':         True,
}

INPUT_CHANNELS = 4
INPUT_SIZE     = 384          # spatial size used during training: 384x384
NUM_WARMUP     = 10
NUM_TIMING     = 100

CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'pt_models')
CHECKPOINTS = [
    'cdm_01_0.82835.pth',
    'cdm_04_0.86710.pth',
    'cdm_07_0.89102.pth',
    'cdm_10_0.89449.pth',
]

RESULT_DIR = os.path.join(SCRIPT_DIR, 'output', 'cdmamba-eval')
os.makedirs(RESULT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_model():
    model = cdMamba(**MODEL_CONFIG)
    return model


def load_weights(model, ckpt_path):
    """
    Weights in pt_models/ are plain OrderedDicts (raw state dicts).
    No wrapper keys — load directly.
    """
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / (1024 ** 2)
    return total, trainable, size_mb


def compute_flops(model, dummy):
    if not FVCORE_AVAILABLE:
        return None
    try:
        analyzer = FlopCountAnalysis(model, dummy)
        analyzer.unsupported_ops_warnings(False)
        analyzer.uncalled_modules_warnings(False)
        return analyzer.total()
    except Exception as e:
        print(f"  [WARNING] FLOPs computation failed: {e}")
        return None


def measure_latency(model, dummy, num_warmup=NUM_WARMUP, num_timing=NUM_TIMING):
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_timing)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(num_timing)]
    with torch.no_grad():
        for i in range(num_timing):
            starts[i].record()
            _ = model(dummy)
            ends[i].record()
    torch.cuda.synchronize()

    latencies = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    avg_ms = float(np.mean(latencies))
    fps    = 1000.0 / avg_ms
    return avg_ms, fps


def measure_gpu_memory(model, dummy):
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() / (1024 ** 2)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() / (1024 ** 2)
    peak  = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return before, after, peak


def measure_power_energy(model, dummy, avg_latency_ms):
    if not PYNVML_AVAILABLE:
        return None, None
    try:
        handle  = pynvml.nvmlDeviceGetHandleByIndex(0)
        samples = []
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy)
                torch.cuda.synchronize()
                samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        avg_power  = float(np.mean(samples))
        energy_per = avg_power * (avg_latency_ms / 1000.0)
        return avg_power, energy_per
    except Exception as e:
        print(f"  [WARNING] Power measurement failed: {e}")
        return None, None


# ── Main Evaluation Loop ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CD-Mamba EFFICIENCY EVALUATION — ALL FOLDS")
print("=" * 70)
print(f"  Input:  {INPUT_CHANNELS} channels  |  {INPUT_SIZE}x{INPUT_SIZE} spatial")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print("=" * 70)

device = torch.device('cuda')
dummy  = torch.randn(1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE).to(device)

# Compute params & FLOPs once (architecture is identical across checkpoints)
print("\n  [1/5] Computing parameters and model size...")
model_probe = build_model().to(device).eval()
total_params, trainable_params, size_mb = count_params(model_probe)
print(f"        Parameters:  {total_params:,}")
print(f"        Size:        {size_mb:.4f} MB")

print("\n  [2/5] Computing FLOPs...")
flops_total = compute_flops(model_probe, dummy)
if flops_total:
    print(f"        FLOPs:       {flops_total:,}")
else:
    print("        FLOPs:       N/A (fvcore not available)")
del model_probe
torch.cuda.empty_cache()

# Per-checkpoint results
all_results = []

for idx, ckpt_name in enumerate(CHECKPOINTS):
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    fold_tag  = ckpt_name.split('_')[1]  # '01', '04', '07', '10'

    print(f"\n  --- Fold checkpoint: {ckpt_name} ---")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] File not found: {ckpt_path}")
        continue

    model = build_model()
    model = load_weights(model, ckpt_path)
    model = model.to(device).eval()

    print(f"  [3/5] Measuring latency & FPS...")
    avg_latency_ms, fps = measure_latency(model, dummy)
    print(f"        Latency:     {avg_latency_ms:.4f} ms  |  FPS: {fps:.2f}")

    print(f"  [4/5] Measuring GPU memory...")
    mem_before, mem_after, mem_peak = measure_gpu_memory(model, dummy)
    print(f"        Before: {mem_before:.2f} MB  |  After: {mem_after:.2f} MB  |  Peak: {mem_peak:.2f} MB")

    print(f"  [5/5] Measuring power & energy...")
    avg_power, energy_per_inf = measure_power_energy(model, dummy, avg_latency_ms)
    if avg_power:
        print(f"        Power:       {avg_power:.4f} W")
        print(f"        Energy/inf:  {energy_per_inf:.8f} J")
    else:
        print("        Power/Energy: N/A (pynvml not available)")

    fold_result = {
        "checkpoint":           ckpt_name,
        "fold":                 fold_tag,
        "latency_ms":           round(avg_latency_ms, 4),
        "fps":                  round(fps, 2),
        "gpu_memory_before_mb": round(mem_before, 2),
        "gpu_memory_after_mb":  round(mem_after, 2),
        "gpu_memory_peak_mb":   round(mem_peak, 2),
        "avg_power_w":          round(avg_power, 4)      if avg_power      is not None else "N/A",
        "energy_per_inf_j":     round(energy_per_inf, 8) if energy_per_inf is not None else "N/A",
    }
    all_results.append(fold_result)

    del model
    torch.cuda.empty_cache()


# ── Compute Averages ───────────────────────────────────────────────────────────
def safe_avg(key):
    vals = [r[key] for r in all_results if isinstance(r.get(key), (int, float))]
    return round(float(np.mean(vals)), 4) if vals else "N/A"

avg_result = {
    "checkpoint":           "AVERAGE (all folds)",
    "fold":                 "avg",
    "latency_ms":           safe_avg("latency_ms"),
    "fps":                  safe_avg("fps"),
    "gpu_memory_before_mb": safe_avg("gpu_memory_before_mb"),
    "gpu_memory_after_mb":  safe_avg("gpu_memory_after_mb"),
    "gpu_memory_peak_mb":   safe_avg("gpu_memory_peak_mb"),
    "avg_power_w":          safe_avg("avg_power_w"),
    "energy_per_inf_j":     safe_avg("energy_per_inf_j"),
}


# ── Final Summary ──────────────────────────────────────────────────────────────
sep = "=" * 70
print(f"\n{sep}")
print(f"  CD-Mamba FULL EVALUATION SUMMARY")
print(sep)
print(f"  Input size:           {INPUT_SIZE}x{INPUT_SIZE}  ({INPUT_CHANNELS} channels)")
print(f"  Parameters:           {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Size:                 {size_mb:.4f} MB")
print(f"  FLOPs:                {flops_total:,}" if flops_total else "  FLOPs:                N/A")
print()
print(f"  {'Checkpoint':<30}  {'Latency (ms)':>12}  {'FPS':>7}  {'Peak Mem (MB)':>13}  {'Power (W)':>10}  {'Energy (J)':>12}")
print(f"  {'-'*30}  {'-'*12}  {'-'*7}  {'-'*13}  {'-'*10}  {'-'*12}")
for r in all_results:
    pw  = f"{r['avg_power_w']:.4f}"   if isinstance(r['avg_power_w'],   float) else "N/A"
    en  = f"{r['energy_per_inf_j']:.6f}" if isinstance(r['energy_per_inf_j'], float) else "N/A"
    print(f"  {r['checkpoint']:<30}  {r['latency_ms']:>12.4f}  {r['fps']:>7.2f}  {r['gpu_memory_peak_mb']:>13.2f}  {pw:>10}  {en:>12}")
print(f"  {'-'*30}  {'-'*12}  {'-'*7}  {'-'*13}  {'-'*10}  {'-'*12}")
apw = f"{avg_result['avg_power_w']:.4f}"     if isinstance(avg_result['avg_power_w'],     float) else "N/A"
aen = f"{avg_result['energy_per_inf_j']:.6f}" if isinstance(avg_result['energy_per_inf_j'], float) else "N/A"
print(f"  {'AVERAGE':<30}  {avg_result['latency_ms']:>12.4f}  {avg_result['fps']:>7.2f}  {avg_result['gpu_memory_peak_mb']:>13.2f}  {apw:>10}  {aen:>12}")
print(sep)


# ── Save Results ───────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
now_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

final_data = {
    "model":              "CD-Mamba",
    "timestamp":          now_str,
    "input_channels":     INPUT_CHANNELS,
    "input_size":         f"{INPUT_SIZE}x{INPUT_SIZE}",
    "parameters":         total_params,
    "trainable_params":   trainable_params,
    "size_mb":            round(size_mb, 4),
    "flops":              flops_total,
    "per_fold_results":   all_results,
    "averaged_results":   avg_result,
}

json_path = os.path.join(RESULT_DIR, f"cdmamba_eval_{timestamp}.json")
txt_path  = os.path.join(RESULT_DIR, f"cdmamba_eval_{timestamp}.txt")

with open(json_path, 'w') as f:
    json.dump(final_data, f, indent=4)

with open(txt_path, 'w') as f:
    f.write("CD-Mamba EFFICIENCY EVALUATION\n")
    f.write(f"Timestamp:      {now_str}\n")
    f.write(f"Input:          {INPUT_CHANNELS} channels, {INPUT_SIZE}x{INPUT_SIZE}\n\n")
    f.write("--- Model Complexity ---\n")
    f.write(f"Parameters:           {total_params:,}\n")
    f.write(f"Trainable Parameters: {trainable_params:,}\n")
    f.write(f"Size:                 {size_mb:.4f} MB\n")
    f.write(f"FLOPs:                {flops_total:,}\n" if flops_total else "FLOPs:                N/A\n")
    f.write("\n--- Per-Fold Results ---\n")
    for r in all_results:
        f.write(f"\n  Checkpoint: {r['checkpoint']}\n")
        f.write(f"    Latency:      {r['latency_ms']} ms\n")
        f.write(f"    FPS:          {r['fps']}\n")
        f.write(f"    GPU Mem Peak: {r['gpu_memory_peak_mb']} MB\n")
        f.write(f"    Avg Power:    {r['avg_power_w']} W\n")
        f.write(f"    Energy/inf:   {r['energy_per_inf_j']} J\n")
    f.write("\n--- Averaged Results (all folds) ---\n")
    f.write(f"  Latency:      {avg_result['latency_ms']} ms\n")
    f.write(f"  FPS:          {avg_result['fps']}\n")
    f.write(f"  GPU Mem Peak: {avg_result['gpu_memory_peak_mb']} MB\n")
    f.write(f"  Avg Power:    {avg_result['avg_power_w']} W\n")
    f.write(f"  Energy/inf:   {avg_result['energy_per_inf_j']} J\n")

print(f"\n  Results saved to:")
print(f"    {json_path}")
print(f"    {txt_path}")
