#!/bin/bash -l
#SBATCH --job-name=sngan_v2
#SBATCH --output=sngan_v2_%j.out
#SBATCH --error=sngan_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00

echo "=========================================="
echo "  SN-PatchGAN Quality v2 — SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Started: $(date)"

source /scratch/users/jfernandezmartinez/miniconda3/etc/profile.d/conda.sh
conda activate cloudgan

cd /scratch/users/jfernandezmartinez/CloudGAN/evaluation
python evaluate_snpatchgan_quality_v2.py

echo "Finished: $(date)"
