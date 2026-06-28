"""
DVPNet Full Evaluation Script — T-Cloud (with image quality metrics)
Computes: Parameters, Size, FLOPs, Latency, FPS, GPU Memory, Power, Energy,
          PSNR, SSIM, MAE, MSE, RMSE, BRMSE, SAM
Requires T-Cloud dataset with cloud/ and reference/ subfolders.
"""

import os
import sys
import json
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# Add DVPNet root to path so basicsr is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basicsr.models.archs.DVPNet import DVPNet

# ── Optional: power measurement via pynvml ──────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# ── Optional: FLOPs via fvcore ──────────────────────────────────────────────
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FVCORE_AVAILABLE = False


# ── Argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DVPNet Full Evaluation on T-Cloud')
parser.add_argument('--opt',           required=True,  help='Path to .yml config')
parser.add_argument('--weights',       required=True,  help='Path to .pth weights')
parser.add_argument('--input_dir',     default='/scratch/users/jfernandezmartinez/DVPNet/datasets/T-Cloud/T-Cloud/test/cloud',
                    help='Cloudy images folder (default: test/cloud)')
parser.add_argument('--input_truth_dir', default='/scratch/users/jfernandezmartinez/DVPNet/datasets/T-Cloud/T-Cloud/test/reference',
                    help='Ground truth folder (default: test/reference)')
parser.add_argument('--result_dir',    default='output/tcloud-full-eval', help='Output folder for results')
parser.add_argument('--num_warmup',    type=int, default=10,  help='Warmup runs for latency')
parser.add_argument('--num_timing',    type=int, default=100, help='Timing runs for latency')
parser.add_argument('--input_size',    type=int, default=256, help='Spatial size for FLOPs/latency dummy input')
args = parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def save_img(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def ssim(img1, img2, window_size=11):
    channel = img1.size(1)
    gauss = torch.Tensor(
        [math.exp(-(x - window_size/2)**2 / (2*1.5**2)) for x in range(window_size)]
    )
    gauss /= gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    s1 = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    s2 = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    s12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean().item()

def sam(img1, img2):
    mat = torch.sum(img1 * img2, dim=1)
    norm1 = torch.sqrt(torch.sum(img1 * img1, dim=1))
    norm2 = torch.sqrt(torch.sum(img2 * img2, dim=1))
    mat = mat / (norm1 * norm2 + 1e-8)
    angle = torch.acos(torch.clamp(mat, -1, 1)) * 180.0 / math.pi
    v = torch.mean(angle)
    return float('nan') if torch.isnan(v) else v.item()


# ── Load model ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DVPNet Full Evaluation — T-Cloud")
print("="*60)

with open(args.opt, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.CLoader if hasattr(yaml, 'CLoader') else yaml.Loader)

net_cfg = dict(cfg['network_g'])
net_cfg.pop('type', None)

model = DVPNet(**net_cfg)
checkpoint = torch.load(args.weights, map_location='cpu')
model.load_state_dict(checkpoint['params'])
model.cuda()
model.eval()
print(f"  Weights loaded: {args.weights}")

device = torch.device('cuda')


# ── 1. Parameters & Size ────────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = total_params * 4 / (1024 ** 2)


# ── 2. FLOPs ────────────────────────────────────────────────────────────────
flops_total = None
if FVCORE_AVAILABLE:
    dummy = torch.randn(1, 3, args.input_size, args.input_size).cuda()
    try:
        flops_analyzer = FlopCountAnalysis(model, dummy)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        flops_total = flops_analyzer.total()
    except Exception as e:
        print(f"  [WARNING] FLOPs computation failed: {e}")


# ── 3. Latency & FPS ────────────────────────────────────────────────────────
dummy = torch.randn(1, 3, args.input_size, args.input_size).cuda()

# Warmup
with torch.no_grad():
    for _ in range(args.num_warmup):
        _ = model(dummy)
torch.cuda.synchronize()

# Timed runs
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_timing)]
end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(args.num_timing)]

with torch.no_grad():
    for i in range(args.num_timing):
        start_events[i].record()
        _ = model(dummy)
        end_events[i].record()

torch.cuda.synchronize()
latencies_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_latency_ms = float(np.mean(latencies_ms))
fps = 1000.0 / avg_latency_ms


# ── 4. GPU Memory ────────────────────────────────────────────────────────────
torch.cuda.reset_peak_memory_stats()
mem_before_mb = torch.cuda.memory_allocated() / (1024**2)

with torch.no_grad():
    _ = model(dummy)
torch.cuda.synchronize()

mem_after_mb  = torch.cuda.memory_allocated() / (1024**2)
mem_peak_mb   = torch.cuda.max_memory_allocated() / (1024**2)


# ── 5. Power & Energy ────────────────────────────────────────────────────────
avg_power_w      = None
energy_per_inf_j = None

if PYNVML_AVAILABLE:
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_samples = []
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy)
                torch.cuda.synchronize()
                power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        avg_power_w      = float(np.mean(power_samples))
        energy_per_inf_j = avg_power_w * (avg_latency_ms / 1000.0)
    except Exception as e:
        print(f"  [WARNING] Power measurement failed: {e}")


# ── 6. Image Quality Metrics on T-Cloud dataset ─────────────────────────────
extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = []
for ext in extensions:
    files.extend(glob(os.path.join(args.input_dir, '*.' + ext)))
files = natsorted(files)

files_truth = []
for ext in extensions:
    files_truth.extend(glob(os.path.join(args.input_truth_dir, '*.' + ext)))
files_truth = natsorted(files_truth)

if len(files) == 0:
    raise FileNotFoundError(f"No images found in {args.input_dir}")

# Ensure matching filenames (basename match)
# If lengths differ, try to match by basename
if len(files) != len(files_truth):
    print(f"  Warning: number of cloudy images ({len(files)}) != number of reference images ({len(files_truth)})")
    print("  Attempting to match by basename...")
    base2truth = {os.path.splitext(os.path.basename(f))[0]: f for f in files_truth}
    matched_files = []
    matched_truth = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base in base2truth:
            matched_files.append(f)
            matched_truth.append(base2truth[base])
    files = matched_files
    files_truth = matched_truth
    print(f"  Matched {len(files)} image pairs.")

os.makedirs(args.result_dir, exist_ok=True)

mae_list, mse_list, rmse_list, brmse_list = [], [], [], []
psnr_list, ssim_list, sam_list = [], [], []

print(f"\n  Running inference on {len(files)} images...")

with torch.no_grad():
    for file_, truth_file in tqdm(zip(files, files_truth), total=len(files)):
        torch.cuda.empty_cache()

        img       = load_img(file_)
        img_truth = load_img(truth_file)

        inp   = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
        truth = torch.from_numpy(img_truth).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        out = model(inp)
        out = torch.clamp(out, 0, 1)

        # Metrics
        diff = out - truth
        mae_list.append(torch.mean(torch.abs(diff)).item())
        mse_v = torch.mean(diff**2).item()
        mse_list.append(mse_v)
        rmse_list.append(math.sqrt(mse_v))
        brmse_list.append(torch.mean(torch.sqrt(torch.mean(diff**2, dim=[2,3]))).item())
        psnr_list.append(psnr(truth, out))
        ssim_list.append(ssim(out, truth))
        sam_v = sam(truth, out)
        if not math.isnan(sam_v):
            sam_list.append(sam_v)

        # Save images
        fname = os.path.splitext(os.path.basename(file_))[0]
        out_np    = out.permute(0,2,3,1).cpu().numpy()[0]
        inp_np    = inp.permute(0,2,3,1).cpu().numpy()[0]
        truth_np  = truth.permute(0,2,3,1).cpu().numpy()[0]
        from skimage import img_as_ubyte
        save_img(os.path.join(args.result_dir, fname + '_out.png'),   img_as_ubyte(out_np))
        save_img(os.path.join(args.result_dir, fname + '_input.png'), img_as_ubyte(inp_np))
        save_img(os.path.join(args.result_dir, fname + '_truth.png'), img_as_ubyte(truth_np))


# ── 7. Compile Results ───────────────────────────────────────────────────────
dataset_name = os.path.basename(os.path.dirname(args.input_dir.rstrip('/')))  # 'test' or 'train'
results = {
    "model":            "DVPNet",
    "dataset":          f"T-Cloud/{dataset_name}",
    "weights":          args.weights,
    "timestamp":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # Complexity
    "parameters":       total_params,
    "trainable_params": trainable_params,
    "size_mb":          round(model_size_mb, 4),
    "flops":            flops_total,
    "latency_ms":       round(avg_latency_ms, 4),
    "fps":              round(fps, 2),
    "gpu_memory_before_mb": round(mem_before_mb, 2),
    "gpu_memory_after_mb":  round(mem_after_mb, 2),
    "gpu_memory_peak_mb":   round(mem_peak_mb, 2),
    "avg_power_w":      round(avg_power_w, 4)      if avg_power_w      is not None else "N/A",
    "energy_per_inf_j": round(energy_per_inf_j, 8) if energy_per_inf_j is not None else "N/A",
    # Image quality
    "num_images":       len(files),
    "PSNR":             round(float(np.mean(psnr_list)),  4),
    "SSIM":             round(float(np.mean(ssim_list)),  4),
    "MAE":              round(float(np.mean(mae_list)),   6),
    "MSE":              round(float(np.mean(mse_list)),   6),
    "RMSE":             round(float(np.mean(rmse_list)),  6),
    "BRMSE":            round(float(np.mean(brmse_list)), 6),
    "SAM":              round(float(np.mean(sam_list)),   4),
}

# ── 8. Print Results ─────────────────────────────────────────────────────────
sep = "="*60
print(f"\n{sep}")
print(f"  DVPNet FULL EVALUATION — T-Cloud ({dataset_name})")
print(sep)
print(f"  Parameters:           {results['parameters']:,}")
print(f"  Trainable Parameters: {results['trainable_params']:,}")
print(f"  Size:                 {results['size_mb']} MB")
print(f"  FLOPs:                {results['flops']:,}" if results['flops'] else "  FLOPs:                N/A")
print(f"  Latency:              {results['latency_ms']} ms")
print(f"  FPS:                  {results['fps']}")
print(f"  GPU Memory Before:    {results['gpu_memory_before_mb']} MB")
print(f"  GPU Memory After:     {results['gpu_memory_after_mb']} MB")
print(f"  GPU Memory Peak:      {results['gpu_memory_peak_mb']} MB")
print(f"  Avg Power:            {results['avg_power_w']} W")
print(f"  Energy per inference: {results['energy_per_inf_j']} J")
print(sep)
print(f"  Images evaluated:     {results['num_images']}")
print(f"  PSNR:                 {results['PSNR']} dB")
print(f"  SSIM:                 {results['SSIM']}")
print(f"  MAE:                  {results['MAE']}")
print(f"  MSE:                  {results['MSE']}")
print(f"  RMSE:                 {results['RMSE']}")
print(f"  BRMSE:                {results['BRMSE']}")
print(f"  SAM:                  {results['SAM']} deg")
print(sep)

# ── 9. Save to files ─────────────────────────────────────────────────────────
now_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = os.path.join(args.result_dir, f"dvpnet_tcloud_full_eval_{now_str}.json")
txt_path  = os.path.join(args.result_dir, f"dvpnet_tcloud_full_eval_{now_str}.txt")

with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)

with open(txt_path, 'w') as f:
    f.write(f"DVPNet FULL EVALUATION — T-Cloud ({dataset_name})\n")
    f.write(f"Timestamp: {results['timestamp']}\n")
    f.write(f"Weights:   {results['weights']}\n\n")
    f.write(f"--- Model Complexity ---\n")
    f.write(f"Parameters:           {results['parameters']:,}\n")
    f.write(f"Trainable Parameters: {results['trainable_params']:,}\n")
    f.write(f"Size:                 {results['size_mb']} MB\n")
    f.write(f"FLOPs:                {results['flops']:,}\n" if results['flops'] else "FLOPs:                N/A\n")
    f.write(f"Latency:              {results['latency_ms']} ms\n")
    f.write(f"FPS:                  {results['fps']}\n")
    f.write(f"GPU Memory Before:    {results['gpu_memory_before_mb']} MB\n")
    f.write(f"GPU Memory After:     {results['gpu_memory_after_mb']} MB\n")
    f.write(f"GPU Memory Peak:      {results['gpu_memory_peak_mb']} MB\n")
    f.write(f"Avg Power:            {results['avg_power_w']} W\n")
    f.write(f"Energy per inference: {results['energy_per_inf_j']} J\n\n")
    f.write(f"--- Image Quality Metrics ({results['num_images']} images) ---\n")
    f.write(f"PSNR:  {results['PSNR']} dB\n")
    f.write(f"SSIM:  {results['SSIM']}\n")
    f.write(f"MAE:   {results['MAE']}\n")
    f.write(f"MSE:   {results['MSE']}\n")
    f.write(f"RMSE:  {results['RMSE']}\n")
    f.write(f"BRMSE: {results['BRMSE']}\n")
    f.write(f"SAM:   {results['SAM']} deg\n")

print(f"\n  Results saved to:")
print(f"    {json_path}")
print(f"    {txt_path}")
print(f"  Restored images saved to: {args.result_dir}")