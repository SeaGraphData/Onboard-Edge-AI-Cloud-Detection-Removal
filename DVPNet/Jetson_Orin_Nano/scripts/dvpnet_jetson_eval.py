"""
DVPNet Efficiency Evaluation Script — Jetson Orin Nano
Computes: Parameters, Size, FLOPs, Latency, FPS, GPU Memory, Power, Energy
No dataset required — uses dummy input only.
Runs for all 3 pretrained models: RICE1, RICE2, T-Cloud.

Usage:
    cd ~/Desktop/DVPNet
    conda activate DVPNet
    python dvpnet_jetson_eval.py
"""

import os
import sys
import json
import datetime
import subprocess
import threading
import time
import re
import numpy as np
import torch
import yaml

# ── Add DVPNet root to path ──────────────────────────────────────────────────
DVPNET_ROOT = os.path.expanduser('~/Desktop/DVPNet')
sys.path.insert(0, DVPNET_ROOT)
os.chdir(DVPNET_ROOT)

from basicsr.models.archs.DVPNet import DVPNet

# ── Optional: FLOPs via ptflops ──────────────────────────────────────────────
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False
    print('[WARNING] ptflops not available — install with: pip install ptflops')


# ── Power measurement via tegrastats ─────────────────────────────────────────
TEGRASTATS_BIN = '/usr/bin/tegrastats'

def parse_tegrastats_line(line):
    """
    Parse a tegrastats line and return a dict with power values in mW.
    Fields extracted:
        VDD_IN           — total board input power
        VDD_CPU_GPU_CV   — CPU + GPU + CV power
        VDD_SOC          — SoC power
    Each field reports 'current/average' mW; we take the current value.
    """
    result = {}
    for field in ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']:
        m = re.search(rf'{field}\s+(\d+)mW/(\d+)mW', line)
        if m:
            result[field] = int(m.group(1))   # current mW
    return result


def sample_tegrastats(samples_list, stop_event, interval_ms=200):
    """Background thread: run tegrastats and collect power samples."""
    try:
        proc = subprocess.Popen(
            ['sudo', TEGRASTATS_BIN, '--interval', str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if line:
                parsed = parse_tegrastats_line(line)
                if parsed:
                    samples_list.append(parsed)
        proc.terminate()
        proc.wait()
    except Exception as e:
        print(f'  [WARNING] tegrastats sampling failed: {e}')


# ── Config: models to evaluate ───────────────────────────────────────────────
MODELS = [
    {
        'name':    'RICE1',
        'weights': 'pretrained_models/rice1/net_g_best.pth',
        'opt':     'option/rice1-DVPNet.yml',
    },
    {
        'name':    'RICE2',
        'weights': 'pretrained_models/rice2/net_g_best.pth',
        'opt':     'option/rice2-DVPNet.yml',
    },
    {
        'name':    'T-Cloud',
        'weights': 'pretrained_models/t-cloud/net_g_best.pth',
        'opt':     'option/T-cloud-DVPNet.yml',
    },
]

INPUT_SIZE  = 256
NUM_WARMUP  = 10
NUM_TIMING  = 100
OUTPUT_DIR  = os.path.join(DVPNET_ROOT, 'output', 'jetson-eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')


def load_model(opt_path, weights_path):
    with open(opt_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader)
    net_cfg = dict(cfg['network_g'])
    net_cfg.pop('type', None)
    model = DVPNet(**net_cfg)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('params', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(name, opt_path, weights_path):
    print(f'\n{"="*60}')
    print(f'  Evaluating: DVPNet — {name}')
    print(f'{"="*60}')

    model = load_model(opt_path, weights_path)
    print(f'  Weights: {weights_path}')

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    # ── 1. Parameters & Size ─────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb          = total_params * 4 / (1024 ** 2)

    # ── 2. FLOPs via ptflops ─────────────────────────────────────────────────
    flops_total = None
    if PTFLOPS_AVAILABLE:
        try:
            with torch.no_grad():
                macs, _ = get_model_complexity_info(
                    model,
                    (3, INPUT_SIZE, INPUT_SIZE),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False
                )
            flops_total = int(2 * macs)   # MACs → FLOPs
        except Exception as e:
            print(f'  [WARNING] FLOPs computation failed: {e}')

    # ── 3. Warmup ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(dummy)
    torch.cuda.synchronize()

    # ── 4. Latency & FPS ─────────────────────────────────────────────────────
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING)]

    with torch.no_grad():
        for i in range(NUM_TIMING):
            start_events[i].record()
            _ = model(dummy)
            end_events[i].record()

    torch.cuda.synchronize()
    latencies_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency  = float(np.mean(latencies_ms))
    fps          = 1000.0 / avg_latency

    # ── 5. GPU Memory ─────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_peak  = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ── 6. Power via tegrastats ───────────────────────────────────────────────
    power_samples = []
    stop_event    = threading.Event()
    sampler       = threading.Thread(
        target=sample_tegrastats,
        args=(power_samples, stop_event, 200),
        daemon=True
    )
    sampler.start()

    # Run inference while sampling power
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)
            torch.cuda.synchronize()

    time.sleep(0.5)   # let sampler catch last samples
    stop_event.set()
    sampler.join(timeout=3)

    # Aggregate power
    avg_vdd_in         = None
    avg_vdd_cpu_gpu_cv = None
    avg_vdd_soc        = None
    energy_per_inf_j   = None

    if power_samples:
        avg_vdd_in         = float(np.mean([s['VDD_IN']           for s in power_samples if 'VDD_IN'           in s])) / 1000.0
        avg_vdd_cpu_gpu_cv = float(np.mean([s['VDD_CPU_GPU_CV']   for s in power_samples if 'VDD_CPU_GPU_CV'   in s])) / 1000.0
        avg_vdd_soc        = float(np.mean([s['VDD_SOC']          for s in power_samples if 'VDD_SOC'          in s])) / 1000.0
        # Energy = GPU+CPU power × latency per inference
        energy_per_inf_j   = avg_vdd_cpu_gpu_cv * (avg_latency / 1000.0)

    # ── Print results ─────────────────────────────────────────────────────────
    sep = '='*60
    print(f'\n{sep}')
    print(f'  DVPNet EFFICIENCY — {name} — Jetson Orin Nano')
    print(sep)
    print(f'  Input size:           {INPUT_SIZE}x{INPUT_SIZE}')
    print(f'  Parameters:           {total_params:,}')
    print(f'  Trainable Parameters: {trainable_params:,}')
    print(f'  Size:                 {size_mb:.4f} MB')
    print(f'  FLOPs:                {flops_total:,}' if flops_total else '  FLOPs:                N/A')
    print(f'  Latency:              {avg_latency:.4f} ms')
    print(f'  FPS:                  {fps:.2f}')
    print(f'  GPU Memory Before:    {mem_before:.2f} MB')
    print(f'  GPU Memory After:     {mem_after:.2f} MB')
    print(f'  GPU Memory Peak:      {mem_peak:.2f} MB')
    if power_samples:
        print(f'  VDD_IN (board):       {avg_vdd_in:.4f} W')
        print(f'  VDD_CPU_GPU_CV:       {avg_vdd_cpu_gpu_cv:.4f} W')
        print(f'  VDD_SOC:              {avg_vdd_soc:.4f} W')
        print(f'  Energy per inference: {energy_per_inf_j:.8f} J  (VDD_CPU_GPU_CV × latency)')
    else:
        print(f'  Power:                N/A (tegrastats unavailable)')
    print(sep)

    # ── Build result dict ─────────────────────────────────────────────────────
    result = {
        'model':              'DVPNet',
        'dataset':            name,
        'platform':           'Jetson Orin Nano — JetPack 6.1',
        'weights':            weights_path,
        'timestamp':          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_size':         f'{INPUT_SIZE}x{INPUT_SIZE}',
        'parameters':         total_params,
        'trainable_params':   trainable_params,
        'size_mb':            round(size_mb, 4),
        'flops':              flops_total,
        'latency_ms':         round(avg_latency, 4),
        'fps':                round(fps, 2),
        'gpu_memory_before_mb': round(mem_before, 2),
        'gpu_memory_after_mb':  round(mem_after, 2),
        'gpu_memory_peak_mb':   round(mem_peak, 2),
        'vdd_in_w':           round(avg_vdd_in, 4)           if avg_vdd_in           is not None else 'N/A',
        'vdd_cpu_gpu_cv_w':   round(avg_vdd_cpu_gpu_cv, 4)  if avg_vdd_cpu_gpu_cv   is not None else 'N/A',
        'vdd_soc_w':          round(avg_vdd_soc, 4)         if avg_vdd_soc          is not None else 'N/A',
        'energy_per_inf_j':   round(energy_per_inf_j, 8)    if energy_per_inf_j     is not None else 'N/A',
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    now_str   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag       = name.lower().replace('-', '')
    json_path = os.path.join(OUTPUT_DIR, f'dvpnet_{tag}_jetson_{now_str}.json')
    txt_path  = os.path.join(OUTPUT_DIR, f'dvpnet_{tag}_jetson_{now_str}.txt')

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)

    with open(txt_path, 'w') as f:
        f.write(f'DVPNet Efficiency Evaluation — {name} — Jetson Orin Nano\n')
        f.write(f'Timestamp:  {result["timestamp"]}\n')
        f.write(f'Platform:   {result["platform"]}\n')
        f.write(f'Weights:    {result["weights"]}\n\n')
        f.write(f'Input size:           {result["input_size"]}\n')
        f.write(f'Parameters:           {result["parameters"]:,}\n')
        f.write(f'Trainable Parameters: {result["trainable_params"]:,}\n')
        f.write(f'Size:                 {result["size_mb"]} MB\n')
        f.write(f'FLOPs:                {result["flops"]:,}\n' if result["flops"] else 'FLOPs:                N/A\n')
        f.write(f'Latency:              {result["latency_ms"]} ms\n')
        f.write(f'FPS:                  {result["fps"]}\n')
        f.write(f'GPU Memory Before:    {result["gpu_memory_before_mb"]} MB\n')
        f.write(f'GPU Memory After:     {result["gpu_memory_after_mb"]} MB\n')
        f.write(f'GPU Memory Peak:      {result["gpu_memory_peak_mb"]} MB\n')
        f.write(f'VDD_IN (board):       {result["vdd_in_w"]} W\n')
        f.write(f'VDD_CPU_GPU_CV:       {result["vdd_cpu_gpu_cv_w"]} W\n')
        f.write(f'VDD_SOC:              {result["vdd_soc_w"]} W\n')
        f.write(f'Energy per inference: {result["energy_per_inf_j"]} J\n')

    print(f'\n  Results saved to:')
    print(f'    {json_path}')
    print(f'    {txt_path}')

    del model
    torch.cuda.empty_cache()
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '='*60)
    print('  DVPNet Efficiency Evaluation — Jetson Orin Nano')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA:    {torch.cuda.is_available()} — {torch.cuda.get_device_name(0)}')
    print(f'  Output:  {OUTPUT_DIR}')
    print('='*60)

    all_results = []
    for m in MODELS:
        opt_path     = os.path.join(DVPNET_ROOT, m['opt'])
        weights_path = os.path.join(DVPNET_ROOT, m['weights'])

        if not os.path.exists(weights_path):
            print(f'\n  [SKIP] Weights not found: {weights_path}')
            continue
        if not os.path.exists(opt_path):
            print(f'\n  [SKIP] Config not found: {opt_path}')
            continue

        result = evaluate_model(m['name'], opt_path, weights_path)
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_results:
        print('\n' + '='*60)
        print('  SUMMARY — All Models')
        print('='*60)
        print(f'  {"Dataset":<12} {"Lat(ms)":>9} {"FPS":>6} {"PeakMem(MB)":>12} {"VDD_CPU_GPU(W)":>15} {"Energy(J)":>11}')
        print('  ' + '-'*58)
        for r in all_results:
            print(f'  {r["dataset"]:<12} {r["latency_ms"]:>9.2f} {r["fps"]:>6.2f} '
                  f'{r["gpu_memory_peak_mb"]:>12.2f} '
                  f'{str(r["vdd_cpu_gpu_cv_w"]):>15} '
                  f'{str(r["energy_per_inf_j"]):>11}')
        print('='*60)

        now_str      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(OUTPUT_DIR, f'dvpnet_all_jetson_{now_str}.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f'\n  Combined results: {summary_path}')
