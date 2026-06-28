#!/bin/bash -l
#SBATCH --job-name=unet_quality
#SBATCH --output=unet_quality_%j.out
#SBATCH --error=unet_quality_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ── Environment ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "  UNet Quality Evaluation — SLURM Job"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Started:    $(date)"
echo "=========================================="

source /scratch/users/jfernandezmartinez/miniconda3/etc/profile.d/conda.sh
conda activate cloudgan

echo "Python:     $(which python)"
echo "Python ver: $(python --version 2>&1)"
echo "TF ver:     $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "h5py ver:   $(python -c 'import h5py; print(h5py.__version__)')"
echo "=========================================="

# ── Run evaluation ───────────────────────────────────────────────────────────
cd /scratch/users/jfernandezmartinez/CloudGAN/evaluation
python evaluate_unet_quality.py

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
