#!/bin/bash -l
#SBATCH --job-name=CDMamba_patch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_patch_%j.out
#SBATCH --error=/scratch/users/jfernandezmartinez/CDMamba/logs/cdmamba_patch_%j.err

source ~/.bashrc
conda activate cdmamba_backup

cd /scratch/users/jfernandezmartinez/CDMamba

echo "========================================"
echo "  CD-Mamba TIF Patching Job"
echo "  Started: $(date)"
echo "========================================"

python patch_tif_bands.py

echo "========================================"
echo "  Patching Done: $(date)"
echo "========================================"
